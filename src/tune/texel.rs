//! Texel tuning: minimise MSE between sigmoid(eval) and game outcome.
//!
//! ## Overview
//!
//! Peter Österlund's Texel tuning method (2012) treats chess evaluation as a
//! regression problem:
//!
//! 1. Record a large set of positions with known outcomes (1 = White wins,
//!    0.5 = draw, 0 = Black wins).
//! 2. Convert the static evaluation score to a "win probability" via a
//!    sigmoid: `P = 1 / (1 + 10^(-K * score / 400))`.
//! 3. Minimise the **mean squared error** (MSE) between `P` and the actual
//!    game result over the dataset.
//!
//! The optimiser is **coordinate descent**: for each parameter in turn, try
//! incrementing (+1) or decrementing (−1) by one centipawn, keep the change
//! if it reduces the loss, and move on.  Because the evaluation is linear in
//! its parameters, changing one parameter only shifts each position's score by
//! a constant `delta = feature_count × taper_factor` — so we can update all
//! scores in O(N) without recomputing the full evaluation.
//!
//! ## Sigmoid scaling constant K
//!
//! The K constant converts centipawns to a probability.  A score of 400 cp
//! (≈ a pawn up) should give roughly P ≈ 0.73 in practice.  The exact optimal
//! K depends on the dataset and the evaluation scale; [`find_k`] finds it via
//! golden-section search on the validation set.
//!
//! ## Overfitting prevention
//!
//! The dataset is split 90 / 10 into training and validation sets.  The tuner
//! minimises the training loss but checks the validation loss at the end of
//! each sweep.  If validation loss stops improving, tuning halts early.
//!
//! ## Why coordinate descent (not gradient descent)?
//!
//! The evaluation parameters are integers.  Gradient descent would require
//! floating-point parameters and a separate rounding step.  Coordinate descent
//! naturally works with integers, requires no learning-rate tuning, and
//! converges to a local minimum in typically 2-5 sweeps.

use crate::tune::dataset::Entry;
use crate::tune::params::{param_delta, EvalParams, N_PARAMS};

// ─── Sigmoid ──────────────────────────────────────────────────────────────────

/// Win-probability sigmoid: `1 / (1 + 10^(-K * score / 400))`.
///
/// Equivalent to `1 / (1 + exp(-K * score * ln(10) / 400))`.
/// At score=0 this returns 0.5 (equal).
/// At score=400 and K=1.0 this returns ≈ 0.909.
#[inline]
pub fn sigmoid(score: f64, k: f64) -> f64 {
    1.0 / (1.0 + 10_f64.powf(-k * score / 400.0))
}

// ─── Loss ─────────────────────────────────────────────────────────────────────

/// Mean squared error: `mean((sigmoid(score_i) − outcome_i)²)`.
///
/// `scores` and `entries` must have the same length.
pub fn mse_loss(scores: &[f64], entries: &[Entry], k: f64) -> f64 {
    let n = scores.len() as f64;
    if n == 0.0 { return f64::INFINITY; }
    scores
        .iter()
        .zip(entries.iter())
        .map(|(&s, e)| {
            let p = sigmoid(s, k);
            let diff = p - e.outcome;
            diff * diff
        })
        .sum::<f64>()
        / n
}

// ─── Score computation ────────────────────────────────────────────────────────

/// Compute the White-relative score for each entry (from trace + params).
///
/// The result is positive when White is better, regardless of which side
/// is to move.  This is the convention expected by [`mse_loss`]: the
/// outcome is also stored as `1.0 = White wins`.
pub fn compute_scores(entries: &[Entry], params: &EvalParams) -> Vec<f64> {
    entries
        .iter()
        .map(|e| {
            let raw = crate::tune::params::score_from_trace(&e.trace, params) as f64;
            // score_from_trace gives a White-relative score (positive = White better).
            // If it was Black's move, the search would have flipped the sign, but for
            // training we always compare against the White-perspective outcome.
            raw
        })
        .collect()
}

// ─── find_k ───────────────────────────────────────────────────────────────────

/// Find the K constant that minimises MSE on `entries` via golden-section search.
///
/// Search range: `[k_lo, k_hi]`, typical good range is `[0.5, 2.0]`.
pub fn find_k(entries: &[Entry], params: &EvalParams, k_lo: f64, k_hi: f64) -> f64 {
    let scores = compute_scores(entries, params);
    let phi = (5_f64.sqrt() - 1.0) / 2.0; // golden ratio
    let mut a = k_lo;
    let mut b = k_hi;

    for _ in 0..50 {
        let c = b - phi * (b - a);
        let d = a + phi * (b - a);
        if mse_loss(&scores, entries, c) < mse_loss(&scores, entries, d) {
            b = d;
        } else {
            a = c;
        }
        if (b - a).abs() < 1e-6 { break; }
    }
    (a + b) / 2.0
}

// ─── Coordinate descent ───────────────────────────────────────────────────────

/// Run coordinate-descent Texel tuning.
///
/// # Arguments
/// * `train` — training positions (90% of dataset).
/// * `val` — validation positions (10% of dataset).
/// * `params` — initial parameter set (modified in place).
/// * `k` — sigmoid scaling constant (call [`find_k`] first).
/// * `max_sweeps` — stop after this many sweeps regardless of convergence.
///
/// # Returns
/// Number of sweeps actually performed.
pub fn tune(
    train:      &[Entry],
    val:        &[Entry],
    params:     &mut EvalParams,
    k:          f64,
    max_sweeps: usize,
) -> usize {
    // Pre-compute current scores (White-relative).
    let mut train_scores = compute_scores(train, params);
    let mut best_train   = mse_loss(&train_scores, train, k);
    let mut best_val     = mse_loss(&compute_scores(val, params), val, k);

    eprintln!(
        "initial train MSE = {:.6}  val MSE = {:.6}",
        best_train, best_val
    );

    let mut param_vec = params.to_vec();
    let mut sweeps = 0usize;

    for sweep in 0..max_sweeps {
        let mut improved = 0usize;

        for pi in 0..N_PARAMS {
            // Pre-compute the per-position delta for param pi.
            let deltas: Vec<f64> = train.iter()
                .map(|e| param_delta(&e.trace, pi))
                .collect();

            // Candidate: param[pi] += 1
            let scores_up: Vec<f64> = train_scores.iter()
                .zip(deltas.iter())
                .map(|(s, d)| s + d)
                .collect();
            let loss_up = mse_loss(&scores_up, train, k);

            if loss_up < best_train - 1e-9 {
                param_vec[pi] += 1;
                train_scores = scores_up;
                best_train   = loss_up;
                improved    += 1;
                continue;
            }

            // Candidate: param[pi] -= 1
            let scores_dn: Vec<f64> = train_scores.iter()
                .zip(deltas.iter())
                .map(|(s, d)| s - d)
                .collect();
            let loss_dn = mse_loss(&scores_dn, train, k);

            if loss_dn < best_train - 1e-9 {
                param_vec[pi] -= 1;
                train_scores = scores_dn;
                best_train   = loss_dn;
                improved    += 1;
            }
        }

        sweeps = sweep + 1;

        // Update params from the mutated vector for validation scoring.
        *params = EvalParams::from_vec(&param_vec);
        let val_loss = mse_loss(&compute_scores(val, params), val, k);

        eprintln!(
            "sweep {:3}: {} changes  train={:.6}  val={:.6}",
            sweeps, improved, best_train, val_loss
        );

        // Early stop: no improvement this sweep.
        if improved == 0 {
            eprintln!("converged (no changes in sweep {})", sweeps);
            break;
        }

        // Overfitting guard: stop if validation loss is rising significantly.
        if val_loss > best_val + 5e-5 {
            eprintln!("stopping: validation loss rising ({:.6} > {:.6})", val_loss, best_val);
            break;
        }
        best_val = best_val.min(val_loss);
    }

    sweeps
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Split a dataset into training (fraction `ratio`) and validation sets.
/// Uses a deterministic interleaved split (no randomness needed for tuning).
pub fn train_val_split(entries: Vec<Entry>, val_ratio: f64) -> (Vec<Entry>, Vec<Entry>) {
    let val_every = (1.0 / val_ratio).round() as usize;
    let mut train = Vec::new();
    let mut val   = Vec::new();
    for (i, e) in entries.into_iter().enumerate() {
        if (i + 1) % val_every == 0 { val.push(e); } else { train.push(e); }
    }
    eprintln!("split: {} train, {} val", train.len(), val.len());
    (train, val)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0, 1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sigmoid_is_monotone() {
        for &k in &[0.5, 1.0, 2.0] {
            assert!(sigmoid(100.0, k) > sigmoid(0.0, k));
            assert!(sigmoid(-100.0, k) < sigmoid(0.0, k));
        }
    }

    #[test]
    fn sigmoid_bounds() {
        assert!(sigmoid(f64::MAX, 1.0) <= 1.0);
        assert!(sigmoid(f64::MIN, 1.0) >= 0.0);
    }

    #[test]
    fn mse_is_zero_for_perfect_prediction() {
        // If sigmoid(score) == outcome exactly for all entries, MSE = 0.
        // Construct a trivial dataset: outcome=0.5, score=0 (sigmoid(0)=0.5).
        use crate::board::Board;
        use crate::tune::params::compute_trace;
        let board  = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let entry  = Entry { trace: compute_trace(&board), outcome: 0.5, white_to_move: true };
        let scores = vec![0.0_f64]; // sigmoid(0) = 0.5 = outcome
        let loss   = mse_loss(&scores, &[entry], 1.0);
        assert!(loss < 1e-10, "MSE should be ~0 for perfect prediction, got {}", loss);
    }

    #[test]
    fn find_k_returns_plausible_value() {
        use crate::board::Board;
        use crate::tune::params::{compute_trace, EvalParams};
        let params = EvalParams::from_engine();
        let board  = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let entries = vec![
            Entry { trace: compute_trace(&board), outcome: 0.5, white_to_move: true },
        ];
        let k = find_k(&entries, &params, 0.1, 3.0);
        assert!(k > 0.1 && k < 3.0, "K={} out of expected range", k);
    }
}
