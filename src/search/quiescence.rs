//! Quiescence search.
//!
//! Stopping the main search at a fixed depth creates the **horizon effect**: if
//! depth runs out in the middle of a capture sequence, the engine evaluates a
//! position where, say, it has just taken a queen — without seeing that the
//! queen is recaptured next move. It thinks it's winning when it's losing.
//!
//! The fix is to not stop at a "loud" position. At depth 0, instead of returning
//! a static evaluation, we keep searching — but only *tactical* moves (captures
//! and promotions) — until the position is **quiet** (no captures left worth
//! making). Then we evaluate. This makes leaf scores trustworthy.
//!
//! ## Stand-pat
//!
//! The side to move is never forced to capture; it can "stand pat" on the static
//! evaluation. So that score becomes a lower bound (`alpha`): we only look at
//! captures that might *improve* on simply doing nothing. If standing pat already
//! reaches `beta`, the opponent won't allow this line and we cut off immediately.
//!
//! When in check there is no standing pat — we must answer the check — so we
//! search *all* legal moves (evasions), not just captures.

use super::negamax::Searcher;
use super::{INFINITY, MATE};
use crate::board::Board;
use crate::eval::evaluator::Evaluator;
use crate::search::ordering::{capture_score, MAX_PLY};
use crate::search::see::{see, SEE_VALUES};
use crate::board::PieceType;

impl<E: Evaluator> Searcher<E> {
    /// Search captures (and, when in check, all evasions) until the position is
    /// quiet, returning a trustworthy score for `board`.
    pub(crate) fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        if self.nodes & 4095 == 0 && self.should_stop() {
            return 0;
        }
        self.nodes += 1;
        if ply > self.max_ply { self.max_ply = ply; }

        // Bound recursion (some positions have very long forcing sequences).
        if ply as usize >= MAX_PLY {
            return self.evaluator.evaluate(board);
        }

        let in_check = board.is_in_check();
        let mut best = -INFINITY;

        if !in_check {
            // Stand-pat: the option of making no capture at all.
            let stand_pat = self.evaluator.evaluate(board);
            best = stand_pat;
            if stand_pat >= beta {
                return stand_pat;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }

        let mut moves = board.legal_moves();
        if moves.is_empty() {
            // Terminal at the horizon: checkmate, or stalemate (a draw).
            return if in_check { -MATE + ply } else { 0 };
        }

        // Order moves so captures (and promotions) come first; quiets last.
        let n = moves.len();
        let mut scores = [0i32; 256];
        {
            let slice = moves.as_mut_slice();
            for i in 0..n {
                scores[i] = capture_score(board, slice[i]);
            }
        }

        // Delta pruning margin: a capture must be able to beat alpha by at least
        // this much to be worth searching.  Set to 0 to only skip SEE-negative
        // captures; increase to prune further.
        const DELTA_MARGIN: i32 = 0;

        // Maximum material a single capture can win (queen).
        let max_gain = SEE_VALUES[PieceType::Queen.index()];

        let slice = moves.as_mut_slice();
        for i in 0..n {
            // Selection sort: pull the best-scoring remaining move to the front.
            let mut best_idx = i;
            for j in (i + 1)..n {
                if scores[j] > scores[best_idx] {
                    best_idx = j;
                }
            }
            slice.swap(i, best_idx);
            scores.swap(i, best_idx);
            let mv = slice[i];

            // When not in check, only tactical moves are searched. Since moves are
            // sorted by tactical score, the first quiet means all the rest are too.
            if !in_check && !(mv.is_capture() || mv.is_promotion()) {
                break;
            }

            // SEE pruning (not in check): skip captures that lose material.
            // Even in the best case (we win a queen on top of this capture) the
            // position can't improve enough to be worth searching.
            if !in_check && mv.is_capture() {
                let see_val = see(board, mv);
                // Hard prune: capture is statically losing and even the best
                // follow-up can't recover alpha.
                if see_val < 0 && best + max_gain + DELTA_MARGIN < alpha {
                    break; // all remaining moves score ≤ this one (sorted)
                }
                // Skip this specific losing capture.
                if see_val < 0 {
                    continue;
                }
            }

            let undo = board.make_move(mv);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1);
            board.unmake_move(mv, undo);

            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break; // beta cutoff
                }
            }
        }

        best
    }
}
