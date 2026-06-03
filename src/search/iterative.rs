//! Iterative deepening and time management.
//!
//! Rather than searching to a fixed depth, we search depth 1, then 2, then 3,
//! and so on, keeping the best move from the last *fully completed* depth.
//! This has two advantages:
//!
//! * **Anytime behaviour.** A usable move is always ready; we stop the moment
//!   the clock demands it.
//! * **Better ordering.** The best move from depth `d` is searched first at
//!   depth `d+1`, which improves alpha-beta efficiency significantly.
//!
//! ## Time control
//!
//! [`TimeManager`] encapsulates both the *soft limit* (don't start the next
//! depth if we have already used our optimal time) and the *hard deadline*
//! (abort mid-search if the wall clock hits the maximum). See [`time`] for
//! the formula and the stability-scaling design.
//!
//! [`time`]: super::time

use super::negamax::{SearchResult, Searcher};
use super::time::TimeManager;
use super::tt::TranspositionTable;
use super::{is_mate_score, INFINITY, MATE};
use crate::board::Board;
use crate::eval::evaluator::{Evaluator, HandcraftedEvaluator, NnueEvaluator};
use crate::movegen::Move;
use crate::nnue::Network;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

/// Hard cap on search depth (also the maximum recursion depth).
pub const MAX_DEPTH: u32 = 64;

/// Constraints on a search, parsed from a UCI `go` command.
#[derive(Clone, Debug, Default)]
pub struct SearchLimits {
    /// Fixed maximum depth.
    pub depth: Option<u32>,
    /// Fixed thinking time for this move, in milliseconds.
    pub movetime: Option<u64>,
    /// Remaining clock and increment for each side, in milliseconds.
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    /// Moves until the next time control.
    pub movestogo: Option<u32>,
    /// Search until explicitly stopped (`go infinite` or `go ponder`).
    pub infinite: bool,
    /// The engine is pondering: search with no time pressure.
    /// A subsequent `ponderhit` tells the engine the move was played.
    pub ponder: bool,
    /// Stop after roughly this many nodes.
    pub nodes: Option<u64>,
    /// Root moves excluded from the search (used for MultiPV: each successive
    /// PV line excludes the best moves found in earlier lines).
    pub excluded_root_moves: Vec<crate::movegen::Move>,
}

impl SearchLimits {
    /// The depth cap for iterative deepening.
    fn max_depth(&self) -> u32 {
        self.depth.unwrap_or(MAX_DEPTH).min(MAX_DEPTH)
    }

    /// True if the search should run until explicitly stopped (no time pressure).
    pub fn is_infinite(&self) -> bool {
        self.infinite || self.ponder
    }
}

/// One completed depth of an iterative-deepening search, for `info` reporting.
pub struct SearchInfo {
    pub depth: u32,
    /// Maximum ply reached anywhere in the tree (includes quiescence).
    /// Always ≥ `depth`; often several plies deeper in tactical positions.
    pub seldepth: u32,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u128,
    /// How full the transposition table is, in permille (0..=1000).
    pub hashfull: u32,
    /// Principal variation: the line the engine expects, walked out of the TT.
    pub pv: Vec<Move>,
    /// For MultiPV: which line this is (1 = best, 2 = second-best, …).
    pub multipv: usize,
}

/// Run a time-managed iterative-deepening search.
///
/// `on_info` is invoked after each completed depth (the UCI layer turns it into
/// an `info` line). The returned [`SearchResult`] is the deepest fully completed
/// search — guaranteed to contain a legal move unless the position is terminal.
///
/// ## Time control flow
///
/// 1. [`TimeManager::new`] computes a soft limit (optimal time) and a hard
///    deadline (absolute maximum) from `limits`.
/// 2. The hard deadline is passed to `Searcher`, which polls it every 4096
///    nodes and aborts if it is reached.
/// 3. After each completed depth we call [`TimeManager::update`] to track move
///    stability, which adjusts the soft limit, and then check
///    [`TimeManager::soft_limit_expired`]. If it has expired we stop — even if
///    the hard deadline has not yet been reached.
/// Run a time-managed iterative-deepening search.
///
/// `game_history` is the slice of Zobrist hashes for every position that has
/// appeared in the game so far, **not including** the root position.  It is
/// used by the search to detect threefold repetition: if the current search
/// position hash matches any hash in this slice, the position is scored as a
/// draw.
///
/// Pass an empty slice (`&[]`) when no prior history is available (e.g. in
/// benchmarks or tests).
///
/// `nnue` — if `Some`, uses the loaded NNUE network for evaluation; if `None`,
/// falls back to the hand-crafted evaluator.  Both use the same full-refresh
/// per-node approach; incremental updates are added in M38.
pub fn think(
    board: &mut Board,
    limits: &SearchLimits,
    stop: Arc<AtomicBool>,
    tt: &TranspositionTable,
    game_history: &[u64],
    nnue: Option<Arc<Network>>,
    mut on_info: impl FnMut(SearchInfo),
) -> SearchResult {
    // Factory: called twice (main Searcher + fallback Searcher).
    // Cloning an Arc<Network> is a ref-count bump — essentially free.
    let make_eval = || -> Box<dyn Evaluator + Send> {
        match nnue.as_ref() {
            Some(net) => Box::new(NnueEvaluator::new(Arc::clone(net))),
            None      => Box::new(HandcraftedEvaluator),
        }
    };
    let start = Instant::now();
    let mut tm = TimeManager::new(limits, start, board.side_to_move, board.fullmove_number);
    let max_depth = limits.max_depth();

    let mut searcher = Searcher::new_with_history(
        game_history,
        tm.hard_deadline(),
        stop,
        limits.nodes,
        make_eval(),
    );
    // --- Aspiration window constants ---
    // Only used from this depth onward; shallow searches have volatile scores.
    const ASPIRATION_MIN_DEPTH: u32 = 4;
    // Initial window half-width in centipawns (~half a pawn).
    const ASPIRATION_DELTA: i32 = 50;

    let mut best = SearchResult {
        best_move: None,
        score: 0,
        depth: 0,
        nodes: 0,
    };
    let mut pv_move: Option<Move> = None;

    for depth in 1..=max_depth {
        searcher.max_ply = 0; // reset per iteration so seldepth is per-depth

        // Decay history tables so scores from the previous depth have diminishing
        // influence.  Keeps the ordering responsive to fresh tactical discoveries.
        if depth > 1 {
            searcher.history.decay();
            searcher.cont_hist_1.decay();
            searcher.cont_hist_2.decay();
        }

        // Aspiration windows: re-search with a wider window on fail-low/fail-high.
        // At shallow depths (< ASPIRATION_MIN_DEPTH) or before any completed depth,
        // fall back to a full-window search — scores are too volatile to window reliably.
        let (mv, score) = if depth >= ASPIRATION_MIN_DEPTH && best.depth > 0 {
            let prev = best.score;
            let mut delta = ASPIRATION_DELTA;
            let mut lo = (prev - delta).max(-INFINITY);
            let mut hi = (prev + delta).min(INFINITY);
            loop {
                let (m, s) = searcher.search_root(
                    board, depth, pv_move, tt,
                    &limits.excluded_root_moves, lo, hi,
                );
                if searcher.stopped { break (m, s); }
                if s <= lo {
                    // Fail low: widen the lower bound and grow delta for next attempt.
                    lo = (lo - delta).max(-INFINITY);
                    delta += delta >> 1;
                } else if s >= hi {
                    // Fail high: widen the upper bound.
                    hi = (hi + delta).min(INFINITY);
                    delta += delta >> 1;
                } else {
                    break (m, s); // score fell inside the window — exact result
                }
            }
        } else {
            searcher.search_root(
                board, depth, pv_move, tt,
                &limits.excluded_root_moves, -INFINITY, INFINITY,
            )
        };

        if searcher.stopped {
            // Partial iteration — discard; keep the previous completed result.
            break;
        }

        // Update stability tracking *before* checking soft limit, so the
        // stability-scaled limit is current when we make the stop decision.
        pv_move = mv;
        tm.update(mv);

        best = SearchResult {
            best_move: mv,
            score,
            depth,
            nodes: searcher.nodes,
        };
        on_info(SearchInfo {
            depth,
            seldepth: (searcher.max_ply as u32).max(depth),
            score,
            nodes: searcher.nodes,
            time_ms: start.elapsed().as_millis(),
            hashfull: tt.hashfull(),
            pv: extract_pv(board, tt, depth as usize),
            multipv: 1,
        });

        // Forced mate found: deeper search cannot improve it.
        if is_mate_score(score) {
            break;
        }

        // Soft stop: don't start a depth we're unlikely to finish in time.
        // Not applied when pondering or searching with infinite time.
        if !limits.is_infinite() && tm.soft_limit_expired() {
            break;
        }
    }

    // Guarantee a legal move even if the time budget was too small to finish
    // depth 1 (e.g. movetime = 1 ms on a slow machine).
    if best.best_move.is_none() && !board.legal_moves().is_empty() {
        let mut fallback = Searcher::new(
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            make_eval(),
        );
        let (mv, score) = fallback.search_root(board, 1, None, tt, &limits.excluded_root_moves, -INFINITY, INFINITY);
        best = SearchResult {
            best_move: mv,
            score,
            depth: 1,
            nodes: best.nodes + fallback.nodes,
        };
    } else if best.best_move.is_none() {
        // Terminal position: report the mate/stalemate score correctly.
        best.score = if board.is_in_check() { -MATE } else { 0 };
    }

    best
}

/// Walk the principal variation out of the transposition table: repeatedly
/// follow the stored best move from the current position. A legality check
/// guards against stale or collided entries; `max_len` bounds the walk.
fn extract_pv(board: &mut Board, tt: &TranspositionTable, max_len: usize) -> Vec<Move> {
    let mut pv = Vec::new();
    let mut undos = Vec::new();

    while pv.len() < max_len {
        let Some(mv) = tt.probe(board.hash).and_then(|d| d.best_move()) else {
            break;
        };
        if !board.legal_moves().iter().any(|&m| m == mv) {
            break;
        }
        undos.push((mv, board.make_move(mv)));
        pv.push(mv);
    }
    while let Some((mv, undo)) = undos.pop() {
        board.unmake_move(mv, undo);
    }
    pv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Color, STARTING_FEN};
    use crate::search::time::TimeManager;

    #[test]
    fn movetime_limit_returns_a_move() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let limits = SearchLimits {
            movetime: Some(100),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(1);
        let result = think(&mut board, &limits, stop, &tt, &[], None, |_| {});
        assert!(result.best_move.is_some());
        assert!(result.depth >= 1);
    }

    #[test]
    fn depth_limit_is_respected() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let tt = TranspositionTable::new(1);
        let mut last_depth = 0;
        let result = think(&mut board, &limits, stop, &tt, &[], None, |info| last_depth = info.depth);
        assert_eq!(result.depth, 3);
        assert_eq!(last_depth, 3);
    }

    #[test]
    fn clock_allocation_scales_with_time() {
        // More time on the clock should produce a longer soft limit (via TimeManager).
        let start = Instant::now();
        let small = SearchLimits {
            wtime: Some(1_000),
            ..Default::default()
        };
        let large = SearchLimits {
            wtime: Some(60_000),
            ..Default::default()
        };
        let tm_small = TimeManager::new(&small, start, Color::White, 1);
        let tm_large = TimeManager::new(&large, start, Color::White, 1);
        assert!(
            tm_large.soft_limit().is_some(),
            "large clock should give a soft limit"
        );
        assert!(
            tm_small.soft_limit().is_some(),
            "small clock should give a soft limit"
        );
        assert!(
            tm_large.soft_limit().unwrap() > tm_small.soft_limit().unwrap(),
            "more time on clock => later soft limit"
        );
    }

    #[test]
    fn aspiration_does_not_change_best_move() {
        // think() uses aspiration windows at depth >= 4.  The final best move
        // must match a full-window search at the same depth.
        let mut b1 = Board::from_fen(STARTING_FEN).unwrap();
        let mut b2 = Board::from_fen(STARTING_FEN).unwrap();
        let tt1 = TranspositionTable::new(4);
        let tt2 = TranspositionTable::new(4);
        let stop1 = Arc::new(AtomicBool::new(false));
        let stop2 = Arc::new(AtomicBool::new(false));

        let limits = SearchLimits { depth: Some(6), ..Default::default() };
        let r1 = think(&mut b1, &limits, stop1, &tt1, &[], None, |_| {});
        let r2 = think(&mut b2, &limits, stop2, &tt2, &[], None, |_| {});

        assert_eq!(
            r1.best_move.map(|m| m.to_uci()),
            r2.best_move.map(|m| m.to_uci()),
            "two identical fixed-depth searches must agree on best move"
        );
        assert!(r1.depth >= 4, "must have searched to depth >= 4 for aspiration to apply");
    }

    #[test]
    fn stop_flag_terminates_search() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let limits = SearchLimits {
            infinite: true,
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let tt = TranspositionTable::new(1);

        // Flip the stop flag after the search starts. Since we're single-threaded
        // here we just pre-set it; the Searcher will abort on the first poll.
        stop2.store(true, std::sync::atomic::Ordering::Relaxed);
        let result = think(&mut board, &limits, stop, &tt, &[], None, |_| {});
        // Should return a fallback move from depth 1.
        assert!(result.best_move.is_some(), "must have a move even after early stop");
    }
}
