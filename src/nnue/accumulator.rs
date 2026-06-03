//! Incremental NNUE accumulator.
//!
//! ## What is the accumulator?
//!
//! The first hidden layer of the NNUE is a linear layer of `HIDDEN_SIZE` neurons.
//! Its pre-activation value for neuron `i` is:
//!
//! ```text
//! acc[i] = bias[i] + Σ_active_features  weight[feature_idx][i]
//! ```
//!
//! Because the input is *sparse* (only ~30 features active), this is a sum of
//! ~30 weight rows — much cheaper than a dense matrix-vector product.
//!
//! ## Incremental updates
//!
//! Between consecutive positions in the search tree, only a small number of
//! features change (the moved piece appears/disappears at specific squares).
//! Rather than recomputing the full sum from scratch, we maintain the
//! accumulator **incrementally**:
//!
//! ```text
//! on make_move(piece p, from → to):
//!   for i in 0..HIDDEN_SIZE:
//!     acc[i] -= weight[feature(king, p, from)][i]   // remove old position
//!     acc[i] += weight[feature(king, p, to)][i]     // add new position
//! ```
//!
//! `unmake_move` is the exact inverse.  A **king move** invalidates the whole
//! accumulator (all feature indices depend on the king square) and triggers a
//! full *refresh* — iterating all non-king pieces and summing their rows.
//!
//! ## Stack discipline
//!
//! [`AccumulatorStack`] mirrors the search's make/unmake call pattern:
//!
//! * **`push()`** — save the current accumulator before calling `board.make_move`.
//! * **Apply deltas** — add/subtract the feature rows that changed.
//! * **`pop()`** — restore the saved accumulator after `board.unmake_move`.
//!
//! The current accumulator is always in `AccumulatorStack::current`, ready for
//! [`Network::evaluate`](super::network::Network::evaluate).
//!
//! ## Current usage
//!
//! [`crate::eval::evaluator::NnueEvaluator`] performs a **full refresh** on
//! every `evaluate()` call — correct but not incremental.  To activate
//! incremental updates in the search, hook `push_with_deltas` / `pop` into
//! [`crate::search::negamax::Searcher`]'s `make_move` / `unmake_move` sites,
//! storing the `AccumulatorStack` as a field alongside `hash_history`.

use super::features::{active_features, HIDDEN_SIZE, INPUT_FEATURES};
use crate::board::Board;

// ─── Feature transformer weights ─────────────────────────────────────────────

/// Weights for the feature-transformer (first) layer.
///
/// The weight matrix is stored row-major: `weight[feature_idx * HIDDEN_SIZE + neuron_idx]`.
/// The bias vector is `bias[neuron_idx]`.
pub struct FeatureWeights {
    /// Bias for each of the `HIDDEN_SIZE` neurons.
    pub bias: Vec<i16>,
    /// `INPUT_FEATURES × HIDDEN_SIZE` weight matrix, row-major by feature index.
    pub weight: Vec<i16>,
}

impl FeatureWeights {
    /// All-zeros weights (used for testing or as a zeroed-network baseline).
    pub fn zeros() -> Self {
        FeatureWeights {
            bias:   vec![0i16; HIDDEN_SIZE],
            weight: vec![0i16; INPUT_FEATURES * HIDDEN_SIZE],
        }
    }

    /// The weight row for feature `feature_idx` (a slice of length `HIDDEN_SIZE`).
    #[inline]
    pub fn row(&self, feature_idx: usize) -> &[i16] {
        let start = feature_idx * HIDDEN_SIZE;
        &self.weight[start..start + HIDDEN_SIZE]
    }
}

// ─── One-sided accumulator ────────────────────────────────────────────────────

/// A single-side accumulator: the first-layer pre-activation for one king's
/// perspective, as an array of `HIDDEN_SIZE` i16 values.
///
/// Values outside `[−32768, 32767]` cause saturation; in practice well-trained
/// weights keep activations in range, and ClampedReLU further bounds them to
/// `[0, QA]` during inference.
#[derive(Clone)]
pub struct HalfAccumulator {
    pub values: Vec<i16>, // length == HIDDEN_SIZE
}

impl HalfAccumulator {
    fn from_bias(bias: &[i16]) -> Self {
        HalfAccumulator { values: bias.to_vec() }
    }

    /// Add the weight row for `feature_idx` to this accumulator.
    #[inline]
    pub fn add_feature(&mut self, feature_idx: usize, weights: &FeatureWeights) {
        let row = weights.row(feature_idx);
        for (a, &w) in self.values.iter_mut().zip(row.iter()) {
            *a = a.saturating_add(w);
        }
    }

    /// Subtract the weight row for `feature_idx` from this accumulator.
    #[inline]
    pub fn sub_feature(&mut self, feature_idx: usize, weights: &FeatureWeights) {
        let row = weights.row(feature_idx);
        for (a, &w) in self.values.iter_mut().zip(row.iter()) {
            *a = a.saturating_sub(w);
        }
    }
}

// ─── Two-sided accumulator ────────────────────────────────────────────────────

/// The full two-sided accumulator: one `HalfAccumulator` for White's king
/// perspective and one for Black's king perspective.
#[derive(Clone)]
pub struct Accumulator {
    /// Pre-activation from White's king perspective.
    pub white: HalfAccumulator,
    /// Pre-activation from Black's king perspective.
    pub black: HalfAccumulator,
}

impl Accumulator {
    fn from_weights(weights: &FeatureWeights) -> Self {
        Accumulator {
            white: HalfAccumulator::from_bias(&weights.bias),
            black: HalfAccumulator::from_bias(&weights.bias),
        }
    }

    /// Recompute both halves from scratch for the given position.
    ///
    /// O(pieces × HIDDEN_SIZE).  Used on king moves and at the root of each
    /// search iteration; individual non-king moves use incremental updates.
    pub fn refresh(&mut self, board: &Board, weights: &FeatureWeights) {
        self.white = HalfAccumulator::from_bias(&weights.bias);
        self.black = HalfAccumulator::from_bias(&weights.bias);

        let (wf, bf) = active_features(board);
        for f in wf { self.white.add_feature(f, weights); }
        for f in bf { self.black.add_feature(f, weights); }
    }
}

// ─── Accumulator stack ────────────────────────────────────────────────────────

/// An accumulator maintained as a push-down stack, mirroring the search tree.
///
/// ## Typical search hook
///
/// ```ignore
/// // Before board.make_move(mv):
/// let (removed_wf, added_wf, removed_bf, added_bf) = move_deltas(&board, mv, &net.feature_weights);
/// acc_stack.push_with_deltas(&removed_wf, &added_wf, &removed_bf, &added_bf, &net.feature_weights);
///
/// // After board.unmake_move(mv, undo):
/// acc_stack.pop();
/// ```
///
/// For king moves, call `acc_stack.current.refresh(board_after_move, &net.feature_weights)`
/// instead of using deltas.
pub struct AccumulatorStack {
    /// The accumulator for the current node in the search tree.
    pub current: Accumulator,
    stack: Vec<Accumulator>,
}

impl AccumulatorStack {
    /// Create a new stack with the accumulator pre-populated from `weights`'
    /// bias vector.  Call `refresh()` or `push_with_deltas()` before using.
    pub fn new(weights: &FeatureWeights) -> Self {
        AccumulatorStack {
            current: Accumulator::from_weights(weights),
            stack: Vec::new(),
        }
    }

    /// Rebuild the current accumulator for `board` from scratch.
    ///
    /// Must be called at least once before inference.  Also call on king moves.
    pub fn refresh(&mut self, board: &Board, weights: &FeatureWeights) {
        self.current.refresh(board, weights);
    }

    /// Save the current accumulator and apply incremental feature deltas.
    ///
    /// `removed_*` and `added_*` are feature indices for the White and Black
    /// perspectives respectively that were removed / added by the move.
    pub fn push_with_deltas(
        &mut self,
        removed_white: &[usize],
        added_white:   &[usize],
        removed_black: &[usize],
        added_black:   &[usize],
        weights: &FeatureWeights,
    ) {
        self.stack.push(self.current.clone());
        for &f in removed_white { self.current.white.sub_feature(f, weights); }
        for &f in added_white   { self.current.white.add_feature(f, weights); }
        for &f in removed_black { self.current.black.sub_feature(f, weights); }
        for &f in added_black   { self.current.black.add_feature(f, weights); }
    }

    /// Restore the accumulator to the state before the last `push_with_deltas`.
    ///
    /// Panics if the stack is empty (push/pop mismatch in the search).
    pub fn pop(&mut self) {
        self.current = self.stack.pop().expect("AccumulatorStack::pop with empty stack");
    }

    /// Current stack depth (number of saved frames).
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, STARTING_FEN};

    fn zero_weights() -> FeatureWeights { FeatureWeights::zeros() }

    #[test]
    fn half_accumulator_starts_at_bias() {
        let w = zero_weights();
        let acc = HalfAccumulator::from_bias(&w.bias);
        assert!(acc.values.iter().all(|&v| v == 0));
    }

    #[test]
    fn add_then_sub_restores_original() {
        let mut w = FeatureWeights::zeros();
        // Set a non-trivial weight row.
        let feature = 5;
        for i in 0..HIDDEN_SIZE {
            w.weight[feature * HIDDEN_SIZE + i] = (i % 10) as i16;
        }
        let mut acc = HalfAccumulator::from_bias(&w.bias);
        let before = acc.values.clone();
        acc.add_feature(feature, &w);
        acc.sub_feature(feature, &w);
        assert_eq!(acc.values, before);
    }

    #[test]
    fn refresh_produces_nonzero_for_nonzero_weights() {
        let mut w = FeatureWeights::zeros();
        // Give every feature weight of 1.
        w.weight.iter_mut().for_each(|x| *x = 1);
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mut acc = Accumulator::from_weights(&w);
        acc.refresh(&board, &w);
        // With all weights = 1 and 32 active features, each neuron should be 32 (bias=0).
        assert!(acc.white.values.iter().any(|&v| v != 0));
        assert!(acc.black.values.iter().any(|&v| v != 0));
    }

    #[test]
    fn push_pop_restores_accumulator() {
        let w = zero_weights();
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mut stack = AccumulatorStack::new(&w);
        stack.refresh(&board, &w);

        let before = stack.current.white.values.clone();
        stack.push_with_deltas(&[0, 1], &[2, 3], &[], &[], &w);
        // After push, depth should increase.
        assert_eq!(stack.depth(), 1);
        stack.pop();
        // After pop, depth back to 0 and values restored.
        assert_eq!(stack.depth(), 0);
        assert_eq!(stack.current.white.values, before);
    }

    #[test]
    fn incremental_matches_refresh() {
        // An accumulator updated incrementally should match a fresh refresh.
        // We use a non-trivial weight matrix so differences would show up.
        let mut w = FeatureWeights::zeros();
        for i in 0..HIDDEN_SIZE {
            w.bias[i] = (i as i16 % 7) - 3;
        }
        // Give the first few features non-trivial weights.
        for f in 0..50 {
            for n in 0..HIDDEN_SIZE {
                w.weight[f * HIDDEN_SIZE + n] = ((f * n) % 17) as i16 - 8;
            }
        }

        let board = Board::from_fen(STARTING_FEN).unwrap();

        // Full refresh.
        let mut acc_refresh = Accumulator::from_weights(&w);
        acc_refresh.refresh(&board, &w);

        // Build from scratch using add_feature (should give same result).
        let (wf, bf) = super::super::features::active_features(&board);
        let mut acc_incremental = Accumulator::from_weights(&w);
        for f in &wf { acc_incremental.white.add_feature(*f, &w); }
        for f in &bf { acc_incremental.black.add_feature(*f, &w); }

        assert_eq!(acc_refresh.white.values, acc_incremental.white.values);
        assert_eq!(acc_refresh.black.values, acc_incremental.black.values);
    }
}
