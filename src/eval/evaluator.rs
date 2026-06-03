//! The [`Evaluator`] trait and its two implementations.
//!
//! ## Why a trait?
//!
//! Every static evaluation call in the search funnels through a single method:
//! `self.evaluator.evaluate(board)`. Because the search only sees this trait, you
//! can swap the handcrafted function for a neural network without touching a
//! single line inside the search — just construct a [`Searcher`] with a different
//! evaluator type and everything else is automatic.
//!
//! [`Searcher`]: crate::search::Searcher
//!
//! ## Selecting an evaluator at runtime
//!
//! `Box<dyn Evaluator>` satisfies the `Evaluator` trait itself (see the blanket
//! impl below), so you can create a `Searcher<Box<dyn Evaluator>>` and switch
//! between [`HandcraftedEvaluator`] and [`NnueEvaluator`] without changing
//! any search code:
//!
//! ```ignore
//! let eval: Box<dyn Evaluator> = if let Some(net) = &engine.nnue {
//!     Box::new(NnueEvaluator::new(Arc::clone(net)))
//! } else {
//!     Box::new(HandcraftedEvaluator)
//! };
//! let mut searcher = Searcher::new(deadline, stop, nodes, eval);
//! ```
//!
//! ## NNUE — what it is and how it fits here
//!
//! NNUE stands for **Efficiently Updatable Neural Network** (originally developed
//! for Shogi engines; popularized in chess by Stockfish 12 in 2020). The core
//! insight is that the most expensive step of neural-network inference — the
//! multiplication of a huge sparse input vector by a wide first weight matrix —
//! can be maintained *incrementally* across successive positions in the search
//! tree, rather than recomputed from scratch at every leaf.
//!
//! See [`crate::nnue`] for the full implementation: feature extraction,
//! accumulator stack, binary weight file format, and quantized inference.
//!
//! ### Training pipeline
//!
//! To train a network for this engine you need:
//!
//! 1. **Positions** — tens of millions of non-trivial chess positions. Generated
//!    with [`crate::tune::datagen`] or from strong-engine games.
//! 2. **Labels** — for each position, a blended score:
//!    `score = λ · (WDL from game result) + (1−λ) · (shallow eval)`.
//!    Stockfish uses λ ≈ 0.4.
//! 3. **Loss** — cross-entropy on the WDL blend, or MSE on the eval.
//! 4. **Quantization** — weights stored as i8/i16 for SIMD inference.
//! 5. **File format** — Checksmith `.nnue` (see [`crate::nnue::network`]).
//!
//! Open-source trainers: [`bullet`](https://github.com/jnlt3/blackmarlin-bullet)
//! and Stockfish's `nnue-pytorch`.

use crate::board::Board;
use crate::nnue::{network::Network, AccumulatorStack};
use std::sync::Arc;

/// Assign a static score (centipawns, side-to-move perspective) to a position.
///
/// The search calls `evaluator.evaluate(board)` at every leaf node. Both the
/// classical handcrafted function and a neural-network function satisfy this
/// interface — switching evaluators requires no changes inside the search.
pub trait Evaluator: Send {
    fn evaluate(&self, board: &Board) -> i32;
}

/// Blanket impl so `Box<dyn Evaluator>` can itself be used as an evaluator.
///
/// This allows `Searcher<Box<dyn Evaluator>>`, enabling runtime evaluator
/// selection (e.g. switching between HCE and NNUE via `setoption EvalFile`).
impl Evaluator for Box<dyn Evaluator + Send> {
    fn evaluate(&self, board: &Board) -> i32 {
        (**self).evaluate(board)
    }
}

// ─── Handcrafted evaluator ────────────────────────────────────────────────────

/// The classical hand-crafted evaluator (the engine's current default).
///
/// Delegates directly to [`super::evaluate`], which computes tapered material
/// plus piece-square tables, pawn structure (isolated / doubled / passed),
/// mobility, king safety, bishop pair, and rook-file bonuses.
#[derive(Clone, Copy, Default, Debug)]
pub struct HandcraftedEvaluator;

impl Evaluator for HandcraftedEvaluator {
    #[inline]
    fn evaluate(&self, board: &Board) -> i32 {
        super::evaluate(board)
    }
}

// ─── NNUE evaluator ───────────────────────────────────────────────────────────

/// An NNUE evaluator backed by a loaded (or zeroed) weight file.
///
/// Construct with [`NnueEvaluator::new`] (from a pre-loaded `Arc<Network>`) or
/// [`NnueEvaluator::load`] (from a `.nnue` file path).
///
/// ## Current behaviour
///
/// Each call to [`evaluate`] performs a **full accumulator refresh** from the
/// current board position, which is O(pieces × HIDDEN_SIZE) ≈ O(7,680).
/// This is correct but not fully incremental.
///
/// For production performance, maintain an [`AccumulatorStack`] inside
/// [`crate::search::negamax::Searcher`] and call `push_with_deltas`/`pop`
/// around each `make_move`/`unmake_move`.  See [`crate::nnue`] for details.
///
/// ## Zero network
///
/// Use `NnueEvaluator::new(Arc::new(Network::zeros()))` for testing — it
/// evaluates every position as 0 (valid but strategically blind).
///
/// [`evaluate`]: NnueEvaluator::evaluate
/// [`AccumulatorStack`]: crate::nnue::AccumulatorStack
pub struct NnueEvaluator {
    network: Arc<Network>,
}

impl NnueEvaluator {
    /// Wrap a pre-loaded network.
    pub fn new(network: Arc<Network>) -> Self {
        NnueEvaluator { network }
    }

    /// Load a network from a Checksmith `.nnue` binary file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let network = Network::load(path)?;
        Ok(NnueEvaluator { network: Arc::new(network) })
    }

    /// Return a clone of the internal `Arc<Network>` (cheap — increments ref-count).
    pub fn network(&self) -> Arc<Network> {
        Arc::clone(&self.network)
    }
}

impl Evaluator for NnueEvaluator {
    fn evaluate(&self, board: &Board) -> i32 {
        // Full refresh (non-incremental). Correct for any position.
        let mut stack = AccumulatorStack::new(&self.network.feature_weights);
        stack.refresh(board, &self.network.feature_weights);
        self.network.evaluate(&stack.current, board.side_to_move)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, STARTING_FEN};

    /// A trivial evaluator that always returns a constant, used to verify
    /// that any type implementing the trait can drive the search correctly.
    struct ConstEvaluator(i32);
    impl Evaluator for ConstEvaluator {
        fn evaluate(&self, _board: &Board) -> i32 { self.0 }
    }

    #[test]
    fn handcrafted_matches_direct_call() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let he = HandcraftedEvaluator;
        assert_eq!(he.evaluate(&board), crate::eval::evaluate(&board));
    }

    #[test]
    fn handcrafted_start_position_is_zero() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(HandcraftedEvaluator.evaluate(&board), 0);
    }

    #[test]
    fn evaluator_trait_is_object_safe() {
        let eval: Box<dyn Evaluator + Send> = Box::new(HandcraftedEvaluator);
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(eval.evaluate(&board), 0);
    }

    #[test]
    fn boxed_evaluator_impl_works() {
        // Box<dyn Evaluator + Send> itself satisfies Evaluator, enabling
        // Searcher<Box<dyn Evaluator + Send>>.
        let boxed: Box<dyn Evaluator + Send> = Box::new(HandcraftedEvaluator);
        let board = Board::from_fen(STARTING_FEN).unwrap();
        // Call through the blanket impl.
        assert_eq!(Evaluator::evaluate(&boxed, &board), 0);
    }

    #[test]
    fn custom_evaluator_drives_search() {
        // A search driven by a constant evaluator should still return a legal
        // move (scores are meaningless, but the search mechanics are correct).
        use crate::search::negamax::Searcher;
        use crate::search::tt::TranspositionTable;
        use std::sync::{atomic::AtomicBool, Arc};

        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let tt = TranspositionTable::new(1);
        let mut searcher = Searcher::new(
            None,
            Arc::new(AtomicBool::new(false)),
            Some(500),
            ConstEvaluator(0),
        );
        let (mv, _score) = searcher.search_root(&mut board, 2, None, &tt, &[], -30_000, 30_000);
        assert!(mv.is_some(), "should return a legal move even with a trivial evaluator");
    }

    #[test]
    fn nnue_constants_are_correct() {
        assert_eq!(crate::nnue::INPUT_FEATURES, 40_960);
        assert_eq!(crate::nnue::HIDDEN_SIZE, 512);
    }

    #[test]
    fn nnue_zero_network_evaluates_without_panic() {
        let net = Arc::new(Network::zeros());
        let eval = NnueEvaluator::new(Arc::clone(&net));
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let score = eval.evaluate(&board);
        assert_eq!(score, 0, "zero network must return 0 for any position");
    }

    #[test]
    fn nnue_evaluator_is_send() {
        // NnueEvaluator must be Send so it can be passed to search threads.
        fn assert_send<T: Send>() {}
        assert_send::<NnueEvaluator>();
    }
}
