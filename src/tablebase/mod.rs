//! Endgame tablebase interface.
//!
//! ## What are tablebases?
//!
//! Endgame tablebases are pre-computed databases that give the exact game-theoretic
//! result (Win/Draw/Loss) and distance-to-mate for every legal position with a
//! small number of pieces.  They were pioneered by Ken Thompson (Thompson's KQK
//! table, 1977) and are now most commonly distributed in **Syzygy** format, which
//! covers all positions up to 7 pieces.
//!
//! Unlike search-based play, tablebases are *perfect*: given any 5-piece position,
//! the result and optimal move can be looked up in microseconds without any search
//! at all.
//!
//! ## Syzygy format
//!
//! Syzygy tables (by Ronald de Man / Bojun Guo) are split into two file types:
//!
//! | Extension | Content |
//! |-----------|---------|
//! | `.rtbw`   | WDL tables — Win/Draw/Loss at distance ≤ 2 from the 50-move rule horizon |
//! | `.rtbz`   | DTZ tables — Distance to Zero (distance to irreversible move that converts) |
//!
//! ### How engines use Syzygy
//!
//! 1. **At the leaf**: when the search reaches a position with ≤ N pieces, probe
//!    the WDL table.  If the result is **Win**, return a large positive score; if
//!    **Loss**, return a large negative score; if **Draw**, return 0.  This
//!    replaces the static evaluator entirely.
//!
//! 2. **Root move ordering**: before iterative deepening, probe the DTZ table to
//!    identify moves that preserve the win (or avoid the loss) while minimising
//!    the DTZ counter.  This prevents the engine from stalling near the 50-move
//!    rule.
//!
//! 3. **Fifty-move rule interaction**: the `CursedWin`/`BlessedLoss` results
//!    indicate positions that are wins in theory but drawn due to the 50-move
//!    rule before mate can be forced — these must be treated as draws.
//!
//! ## How to add full Syzygy support
//!
//! 1. Add the `shakmaty-syzygy` or `pyrrhic` crate (C FFI wrapper) to Cargo.toml.
//! 2. Initialise the tablebase paths via the `SyzygyPath` UCI option.
//! 3. Call `probe_wdl` (and `probe_dtz` at the root) wherever the current stub
//!    returns `None`.
//! 4. Integrate the result into `negamax()`: replace the `evaluator.evaluate()`
//!    call at `depth == 0` with a tablebase lookup when piece count ≤ `tb_piece_limit`.
//!
//! The [`probe_wdl`] and [`probe_dtz`] stubs below are already wired into the
//! correct positions; implementing full Syzygy support is therefore a drop-in
//! replacement of the function bodies.

use crate::board::Board;

// ── Result types ──────────────────────────────────────────────────────────────

/// Win/Draw/Loss result from a tablebase probe.
///
/// The `Cursed` and `Blessed` variants represent positions that are wins or
/// losses in perfect play but are drawn under the 50-move rule before mate
/// can be forced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WdlScore {
    /// The side to move wins with correct play, well within the 50-move limit.
    Win,
    /// The side to move wins in theory but cannot force mate before the
    /// 50-move counter expires — effectively a draw.
    CursedWin,
    /// The position is exactly drawn with best play.
    Draw,
    /// The side to move loses in theory but the opponent cannot force mate
    /// before the 50-move counter expires — effectively a draw.
    BlessedLoss,
    /// The side to move loses with correct play.
    Loss,
}

impl WdlScore {
    /// True when the result should be treated as a draw (50-move-rule cursed wins/losses).
    pub fn is_effective_draw(self) -> bool {
        matches!(self, WdlScore::CursedWin | WdlScore::Draw | WdlScore::BlessedLoss)
    }

    /// Convert to a centipawn score for the search (relative to side to move).
    ///
    /// Uses a fixed large constant rather than a mate score so that search can
    /// still prefer shorter wins / longer losses within the tablebase.
    pub fn to_search_score(self) -> i32 {
        match self {
            WdlScore::Win        =>  20_000,
            WdlScore::CursedWin  =>  0,
            WdlScore::Draw       =>  0,
            WdlScore::BlessedLoss => 0,
            WdlScore::Loss       => -20_000,
        }
    }
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Running count of successful tablebase probes, for UCI `info` output.
#[derive(Default)]
pub struct TbHits(pub u64);

impl TbHits {
    pub fn increment(&mut self) { self.0 += 1; }
    pub fn reset(&mut self)     { self.0 = 0; }
}

// ── Probe functions ───────────────────────────────────────────────────────────

/// Probe the WDL (Win/Draw/Loss) tablebase for `board`.
///
/// Returns `Some(result)` if the position is in the tablebase, `None` if it
/// is not (too many pieces, or tablebase files are not loaded).
///
/// # Current status
///
/// This is a **stub**.  It always returns `None`.  To enable real Syzygy
/// probing, add a Syzygy crate to `Cargo.toml` and replace the body of this
/// function.  The call site in `negamax()` is already in place.
#[allow(unused_variables)]
pub fn probe_wdl(board: &Board) -> Option<WdlScore> {
    None
}

/// Probe the DTZ (Distance-to-Zero) tablebase for `board`.
///
/// Returns `Some(dtz)` where `dtz` is the number of half-moves until the next
/// irreversible move that preserves the WDL result.  Used at the root to
/// choose moves that avoid the 50-move-rule horizon.
///
/// This is also a **stub** and always returns `None`.
#[allow(unused_variables)]
pub fn probe_dtz(board: &Board) -> Option<i32> {
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;

    #[test]
    fn probe_returns_none_for_any_position() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(probe_wdl(&board), None);
        assert_eq!(probe_dtz(&board), None);
    }

    #[test]
    fn wdl_draw_variants_are_effective_draws() {
        assert!(WdlScore::Draw.is_effective_draw());
        assert!(WdlScore::CursedWin.is_effective_draw());
        assert!(WdlScore::BlessedLoss.is_effective_draw());
        assert!(!WdlScore::Win.is_effective_draw());
        assert!(!WdlScore::Loss.is_effective_draw());
    }

    #[test]
    fn wdl_scores_have_correct_sign() {
        assert!(WdlScore::Win.to_search_score() > 0);
        assert_eq!(WdlScore::Draw.to_search_score(), 0);
        assert!(WdlScore::Loss.to_search_score() < 0);
    }
}
