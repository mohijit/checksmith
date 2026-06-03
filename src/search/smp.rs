//! Lazy SMP — Symmetric Multi-Processing parallel search.
//!
//! ## Design overview
//!
//! Lazy SMP is the simplest effective parallel search strategy. Rather than
//! partitioning the search tree across threads (which requires complex
//! synchronisation and is prone to load imbalance), every thread independently
//! runs a full iterative-deepening search from the root.
//!
//! The **shared transposition table** is the coordination mechanism:
//! * When thread A stores a result at depth D, thread B at the same depth may
//!   find it and cut off earlier than it would have alone.
//! * Over time, threads effectively guide each other toward promising lines
//!   without explicit communication.
//!
//! ## Thread roles
//!
//! | Thread | Role |
//! |--------|------|
//! | Main (thread 0) | Full iterative deepening with real time limits; reports `info`; determines `bestmove`. |
//! | Helpers (1..N-1) | Iterative deepening with `infinite` time; stop only when the shared stop flag is set. |
//!
//! ## Stop coordination
//!
//! All threads share one `Arc<AtomicBool>` stop flag:
//! * **External stop** (UCI `stop`/`quit`): set by the UCI main thread → all
//!   search threads see it within 4096 nodes and terminate.
//! * **Time limit**: when the main thread's soft or hard limit expires, it sets
//!   the flag before returning (see `spawn_helpers` below). Helpers stop within
//!   one `should_stop` poll cycle.
//!
//! ## Root move splitting
//!
//! Basic Lazy SMP does **not** split root moves — each thread independently
//! orders and searches all root moves. The benefit comes entirely from TT cross-
//! pollution. Future techniques (YBWC, ABDADA) distribute root moves but add
//! significant coordination complexity and are rarely worth it below 16 threads.
//!
//! ## Race conditions and safety
//!
//! Concurrent TT reads and writes can cause benign data races (see `tt.rs`):
//! * A torn write produces a garbage key that fails the key check → TT miss.
//! * In the worst case the race wastes work; it never produces an illegal move.
//!
//! ## Scaling characteristics (empirical)
//!
//! | Threads | Typical Elo gain |
//! |---------|-----------------|
//! | 2       | +40–60 Elo      |
//! | 4       | +70–90 Elo      |
//! | 8       | +80–100 Elo     |
//! | 16+     | Diminishing; TT contention limits gains |
//!
//! Lazy SMP scales well to ~8 threads and remains effective to ~16. Beyond that,
//! more sophisticated work-splitting (or a larger TT) is needed.

use super::iterative::{think, SearchLimits};
use super::tt::TranspositionTable;
use crate::board::Board;
use crate::nnue::Network;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Spawn `count` helper search threads that independently search from `board`.
///
/// Each helper:
/// * Runs `think()` with `infinite = true` (no time pressure).
/// * Shares `tt` via `Arc::clone` — filling it for the benefit of all threads.
/// * Stops when `stop` is set to `true`.
///
/// The caller is responsible for:
/// 1. Running the main search (with real time limits) after this call.
/// 2. Setting `stop.store(true, Ordering::Relaxed)` when the main search ends.
/// 3. Joining the returned handles to ensure helpers have stopped.
///
/// ## Example
///
/// ```ignore
/// let helpers = spawn_helpers(&board, stop.clone(), Arc::clone(&tt), &history, 3);
/// let result = think(&mut board, &limits, stop.clone(), &tt, &history, on_info);
/// stop.store(true, Ordering::Relaxed);  // signal helpers
/// for h in helpers { let _ = h.join(); }
/// ```
/// `nnue` — if `Some`, each helper uses the same loaded NNUE network for
/// evaluation (via `Arc::clone`); if `None`, helpers use the hand-crafted evaluator.
pub fn spawn_helpers(
    board: &Board,
    stop: Arc<AtomicBool>,
    tt: Arc<TranspositionTable>,
    game_history: &[u64],
    count: usize,
    nnue: Option<Arc<Network>>,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::with_capacity(count);

    for _ in 0..count {
        let mut helper_board = board.clone();
        let helper_stop = Arc::clone(&stop);
        let helper_tt = Arc::clone(&tt);
        let helper_history = game_history.to_vec();
        let helper_nnue = nnue.as_ref().map(Arc::clone);

        let h = thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                // Helpers run with infinite time — they stop via the stop flag.
                let infinite_limits = SearchLimits {
                    infinite: true,
                    ..Default::default()
                };
                think(
                    &mut helper_board,
                    &infinite_limits,
                    helper_stop,
                    &helper_tt,
                    &helper_history,
                    helper_nnue,
                    |_| {}, // helpers don't report info
                );
            })
            .expect("failed to spawn SMP helper thread");

        handles.push(h);
    }

    handles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;
    use std::sync::atomic::Ordering;

    #[test]
    fn helpers_stop_when_flag_set() {
        let board = crate::board::Board::from_fen(STARTING_FEN).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let tt = Arc::new(TranspositionTable::new(1));

        // Spawn 2 helper threads, immediately set stop, then join.
        let helpers = spawn_helpers(&board, Arc::clone(&stop), Arc::clone(&tt), &[], 2, None);
        stop.store(true, Ordering::Relaxed);
        for h in helpers {
            h.join().expect("helper thread panicked");
        }
        // Test passes if helpers terminate promptly (within the test timeout).
    }

    #[test]
    fn helpers_fill_shared_tt() {
        // After helpers run briefly, the TT should have some entries.
        let board = crate::board::Board::from_fen(STARTING_FEN).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let tt = Arc::new(TranspositionTable::new(4));
        tt.new_generation();

        let helpers = spawn_helpers(&board, Arc::clone(&stop), Arc::clone(&tt), &[], 1, None);

        // Give the helper ~100ms of search time to fill some TT entries.
        // Use a generous budget: Windows thread startup can consume 20-50ms,
        // leaving less than 10ms of actual search time under parallel test load.
        std::thread::sleep(std::time::Duration::from_millis(100));
        stop.store(true, Ordering::Relaxed);
        for h in helpers { let _ = h.join(); }

        assert!(tt.hashfull() > 0, "helper must have written at least one TT entry");
    }
}
