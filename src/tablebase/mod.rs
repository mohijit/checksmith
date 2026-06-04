//! Endgame tablebase interface — Syzygy probing via shakmaty-syzygy.
//!
//! ## What are tablebases?
//!
//! Endgame tablebases are pre-computed databases that give the exact game-theoretic
//! result (Win/Draw/Loss) and distance-to-mate for every legal position with a
//! small number of pieces.  They were pioneered by Ken Thompson (Thompson's KQK
//! table, 1977) and are now most commonly distributed in **Syzygy** format, which
//! covers all positions up to 7 pieces.
//!
//! ## Syzygy format
//!
//! Syzygy tables (by Ronald de Man / Bojun Guo) are split into two file types:
//!
//! | Extension | Content |
//! |-----------|---------|
//! | `.rtbw`   | WDL tables — Win/Draw/Loss, exact game-theoretic outcome |
//! | `.rtbz`   | DTZ tables — Distance to Zero (plies to next pawn move/capture) |
//!
//! ## How the engine uses Syzygy
//!
//! **WDL probe (in-search):** when a position has ≤ `piece_limit` pieces and
//! `halfmove_clock == 0`, we probe the WDL table.  A Win/Loss score replaces the
//! static evaluator entirely.  A Draw tightens the alpha-beta window to 0.
//! We use `probe_wdl_after_zeroing`, which assumes the 50-move clock has just
//! been reset — exactly our condition.
//!
//! **DTZ probe (at root):** finds the move that preserves the win while advancing
//! the DTZ counter fastest, preventing 50-move-rule horizon stalls.
//!
//! ## 50-move rule interaction
//!
//! WDL is only probed when `halfmove_clock == 0`.  This is conservative but
//! always correct: after any irreversible move (capture or pawn push) the clock
//! resets to 0 and `probe_wdl_after_zeroing` results are unambiguous.
//!
//! ## Thread safety
//!
//! The loaded `Tablebase` is stored behind an `Arc<RwLock<…>>`.  Concurrent
//! probes clone the inner `Arc<Tablebase>` under a read lock — one reference
//! count increment per probe, negligible overhead.  Writes (re-initialisation
//! via `setoption`) take the write lock between searches.
//!
//! ## UCI options
//!
//! | Option | Type | Default | Notes |
//! |--------|------|---------|-------|
//! | `SyzygyPath` | string | `<empty>` | Directory containing `.rtbw`/`.rtbz` files |
//! | `SyzygyPieces` | spin 3..7 | 6 | Maximum piece count for probing |

use crate::board::{Board, Color};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use shakmaty_syzygy::{MaybeRounded, Syzygy, Tablebase, Wdl};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, LazyLock, RwLock};

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
    pub fn to_search_score(self) -> i32 {
        match self {
            WdlScore::Win         =>  20_000,
            WdlScore::CursedWin   =>  0,
            WdlScore::Draw        =>  0,
            WdlScore::BlessedLoss =>  0,
            WdlScore::Loss        => -20_000,
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

// ── Module-level state ────────────────────────────────────────────────────────

struct TbState {
    tablebases:  Option<Arc<Tablebase<Chess>>>,
    piece_limit: u32,
}

static TB: LazyLock<RwLock<TbState>> = LazyLock::new(|| {
    RwLock::new(TbState { tablebases: None, piece_limit: 6 })
});

// ── Public initialisation / configuration ─────────────────────────────────────

/// Load Syzygy tables from `path`.  Returns the number of files found.
///
/// Call this when the GUI sends `setoption name SyzygyPath value <path>`.
/// Passing an empty string or a path with no `.rtbw` files disables probing.
pub fn init(path: &str) -> usize {
    let mut state = TB.write().expect("TB lock poisoned");
    if path.is_empty() {
        state.tablebases = None;
        return 0;
    }
    let mut tb = Tablebase::<Chess>::new();
    let added = tb.add_directory(Path::new(path)).unwrap_or(0);
    state.tablebases = if added > 0 { Some(Arc::new(tb)) } else { None };
    added
}

/// Set the maximum piece count for which TB probing is attempted.
pub fn set_piece_limit(n: u32) {
    TB.write().expect("TB lock poisoned").piece_limit = n.clamp(3, 7);
}

/// The current piece limit for probing.
pub fn piece_limit() -> u32 {
    TB.read().expect("TB lock poisoned").piece_limit
}

/// True if tablebase files have been loaded.
pub fn is_available() -> bool {
    TB.read().expect("TB lock poisoned").tablebases.is_some()
}

// ── Probe functions ───────────────────────────────────────────────────────────

/// Probe the WDL tablebase for `board`.
///
/// Uses `probe_wdl_after_zeroing`, which is correct when `halfmove_clock == 0`
/// (i.e., after the most recent irreversible move).  Only call under that
/// condition.
///
/// Returns `Some(result)` if the position is in a loaded table, `None` otherwise.
pub fn probe_wdl(board: &Board) -> Option<WdlScore> {
    let pos = board_to_shakmaty(board)?;
    let tb = {
        let state = TB.read().expect("TB lock poisoned");
        Arc::clone(state.tablebases.as_ref()?)
    };
    match tb.probe_wdl_after_zeroing(&pos) {
        Ok(Wdl::Win)         => Some(WdlScore::Win),
        Ok(Wdl::CursedWin)   => Some(WdlScore::CursedWin),
        Ok(Wdl::Draw)        => Some(WdlScore::Draw),
        Ok(Wdl::BlessedLoss) => Some(WdlScore::BlessedLoss),
        Ok(Wdl::Loss)        => Some(WdlScore::Loss),
        Err(_)               => None,
    }
}

/// Probe the DTZ tablebase for `board`.
///
/// Returns the raw DTZ value (positive = winning side to move, negative =
/// losing).  Used at the root to pick moves that advance toward a decisive
/// result within the 50-move rule.
pub fn probe_dtz(board: &Board) -> Option<i32> {
    let pos = board_to_shakmaty(board)?;
    let tb = {
        let state = TB.read().expect("TB lock poisoned");
        Arc::clone(state.tablebases.as_ref()?)
    };
    match tb.probe_dtz(&pos) {
        Ok(MaybeRounded::Rounded(dtz)) => Some(dtz.0),
        Ok(MaybeRounded::Precise(dtz)) => Some(dtz.0),
        Err(_)                          => None,
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Convert our `Board` to a shakmaty `Chess` position via its FEN string.
fn board_to_shakmaty(board: &Board) -> Option<Chess> {
    let fen_str = board.to_fen();
    let fen = Fen::from_str(&fen_str).ok()?;
    fen.into_position(CastlingMode::Standard).ok()
}

/// Count the total number of pieces on the board (both colors).
pub fn piece_count(board: &Board) -> u32 {
    let all = board.color(Color::White) | board.color(Color::Black);
    all.0.count_ones()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;

    #[test]
    fn probe_returns_none_when_no_tb_loaded() {
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
        assert_eq!(WdlScore::CursedWin.to_search_score(), 0);
        assert_eq!(WdlScore::BlessedLoss.to_search_score(), 0);
    }

    #[test]
    fn piece_count_starting_position() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(piece_count(&board), 32);
    }

    #[test]
    fn piece_count_endgame() {
        let board = Board::from_fen("7k/8/8/8/8/8/8/K6Q w - - 0 1").unwrap();
        assert_eq!(piece_count(&board), 3);
    }

    #[test]
    fn piece_limit_clamps_to_valid_range() {
        set_piece_limit(99);
        assert_eq!(piece_limit(), 7);
        set_piece_limit(1);
        assert_eq!(piece_limit(), 3);
        set_piece_limit(6);
        assert_eq!(piece_limit(), 6);
    }

    #[test]
    fn init_empty_path_disables_probing() {
        let count = init("");
        assert_eq!(count, 0);
    }

    #[test]
    fn board_conversion_does_not_panic() {
        let fens = [
            STARTING_FEN,
            "7k/8/8/8/8/8/8/K6Q w - - 0 1",
            "8/8/8/8/8/8/4k3/4K2R w - - 0 1",
            "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
        ];
        for fen in &fens {
            let board = Board::from_fen(fen).unwrap();
            let _ = board_to_shakmaty(&board);
        }
    }
}
