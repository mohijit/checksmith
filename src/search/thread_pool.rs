//! Persistent search thread pool for Lazy SMP.
//!
//! Rather than spawning `N-1` helper threads on every `go` command and joining
//! them after each search (non-trivial startup cost at fast time controls),
//! this module keeps `count` worker threads alive for the engine's lifetime.
//!
//! ## Lifecycle
//!
//! 1. [`SearchPool::new`] — called once at engine startup.
//! 2. [`SearchPool::start_helpers`] — called from `start_search()` in `uci/mod.rs`
//!    just before the main search thread is launched.  Each worker receives a
//!    clone of the board, the shared TT, and the shared stop flag, then runs
//!    `think()` with infinite time.
//! 3. **Stop**: when the main search ends it calls
//!    `stop.store(true, Ordering::Relaxed)`.  Workers poll the flag every
//!    ~4096 nodes and return from `think()`, then go back to waiting on the
//!    channel.  No explicit join is needed between searches.
//! 4. [`SearchPool::resize`] — called on `setoption name Threads value N`
//!    (pass `N - 1` as `count`).  Excess workers receive `Quit` and are
//!    joined; new workers are spawned.
//! 5. `drop(pool)` — sends `Quit` to every worker and joins them.
//!
//! ## Per-thread history
//!
//! Each worker creates its own [`Searcher`] (and thus its own killer/history
//! tables) every search iteration inside `think()`.  These tables are
//! independent across threads — no cross-thread data races — which is the
//! standard Lazy SMP design.  Future work can persist the history across
//! searches within the same worker once `think()` is refactored to accept
//! pre-built `Searcher` state.
//!
//! ## Channel design
//!
//! We use an unbounded `mpsc::channel`.  If the previous search's stop flag
//! takes a few milliseconds to propagate, the next `Search` message is simply
//! queued and the worker picks it up once it returns from `think()`.  No
//! backpressure is needed because `start_helpers` sends at most one message
//! per worker per search.

use super::iterative::{think, SearchLimits};
use super::tt::TranspositionTable;
use crate::board::Board;
use crate::nnue::Network;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread;

// ── Messages ──────────────────────────────────────────────────────────────────

enum WorkerMsg {
    Search(SearchWork),
    Quit,
}

struct SearchWork {
    board:        Board,
    stop:         Arc<AtomicBool>,
    tt:           Arc<TranspositionTable>,
    game_history: Vec<u64>,
    nnue:         Option<Arc<Network>>,
}

// ── Worker handle ─────────────────────────────────────────────────────────────

struct WorkerHandle {
    tx:     Sender<WorkerMsg>,
    handle: thread::JoinHandle<()>,
}

// ── SearchPool ────────────────────────────────────────────────────────────────

/// Persistent pool of Lazy SMP helper threads.
///
/// Create once at engine startup.  Helper searches are dispatched via
/// [`start_helpers`](SearchPool::start_helpers) and stop automatically when
/// the main search sets the shared stop flag.
pub struct SearchPool {
    workers: Vec<WorkerHandle>,
}

impl SearchPool {
    /// Create a pool with `count` idle worker threads (0 = single-threaded).
    pub fn new(count: usize) -> Self {
        let mut pool = SearchPool { workers: Vec::with_capacity(count) };
        for i in 0..count {
            pool.workers.push(spawn_worker(i));
        }
        pool
    }

    /// Number of helper workers (excluding the main search thread).
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Dispatch a helper search to every worker.
    ///
    /// Workers run `think()` with `infinite = true` and stop when `stop` is
    /// set.  Call this just before the main search; the main search sets `stop`
    /// when it finishes, causing workers to return and wait for the next cmd.
    pub fn start_helpers(
        &self,
        board:        &Board,
        stop:         Arc<AtomicBool>,
        tt:           Arc<TranspositionTable>,
        game_history: &[u64],
        nnue:         Option<Arc<Network>>,
    ) {
        for w in &self.workers {
            let work = SearchWork {
                board:        board.clone(),
                stop:         Arc::clone(&stop),
                tt:           Arc::clone(&tt),
                game_history: game_history.to_vec(),
                nnue:         nnue.as_ref().map(Arc::clone),
            };
            let _ = w.tx.send(WorkerMsg::Search(work));
        }
    }

    /// Resize the pool to exactly `count` helper workers.
    ///
    /// Shrinking: excess workers receive `Quit` and are joined synchronously.
    /// Growing: new workers are spawned immediately.
    ///
    /// Call after `setoption name Threads value N` with `count = N - 1`.
    pub fn resize(&mut self, count: usize) {
        let current = self.workers.len();
        if count < current {
            let excess = self.workers.drain(count..).collect::<Vec<_>>();
            for w in excess {
                let _ = w.tx.send(WorkerMsg::Quit);
                let _ = w.handle.join();
            }
        } else {
            for i in current..count {
                self.workers.push(spawn_worker(i));
            }
        }
    }
}

impl Drop for SearchPool {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = w.tx.send(WorkerMsg::Quit);
        }
        for w in self.workers.drain(..) {
            let _ = w.handle.join();
        }
    }
}

// ── Worker implementation ─────────────────────────────────────────────────────

fn spawn_worker(id: usize) -> WorkerHandle {
    let (tx, rx) = mpsc::channel::<WorkerMsg>();
    let handle = thread::Builder::new()
        .name(format!("cs-smp-{}", id))
        .stack_size(8 * 1024 * 1024)
        .spawn(move || worker_loop(rx))
        .expect("failed to spawn SMP worker thread");
    WorkerHandle { tx, handle }
}

fn worker_loop(rx: mpsc::Receiver<WorkerMsg>) {
    while let Ok(msg) = rx.recv() {
        let WorkerMsg::Search(work) = msg else { break };
        let SearchWork { mut board, stop, tt, game_history, nnue } = work;
        let limits = SearchLimits { infinite: true, ..Default::default() };
        think(&mut board, &limits, stop, &tt, &game_history, nnue, |_| {});
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[test]
    fn pool_zero_workers_is_a_no_op() {
        let pool = SearchPool::new(0);
        assert_eq!(pool.len(), 0);
        let stop = Arc::new(AtomicBool::new(false));
        let tt = Arc::new(TranspositionTable::new(1));
        let board = Board::from_fen(STARTING_FEN).unwrap();
        pool.start_helpers(&board, stop, tt, &[], None); // must not panic
    }

    #[test]
    fn workers_stop_when_flag_is_set() {
        let pool = SearchPool::new(2);
        let stop = Arc::new(AtomicBool::new(false));
        let tt = Arc::new(TranspositionTable::new(1));
        tt.new_generation();
        let board = Board::from_fen(STARTING_FEN).unwrap();

        pool.start_helpers(&board, Arc::clone(&stop), Arc::clone(&tt), &[], None);
        thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        // Workers should return from think() quickly and go back to recv().
        // Verify by dropping the pool cleanly (sends Quit + joins).
        thread::sleep(Duration::from_millis(30));
        drop(pool);
    }

    #[test]
    fn pool_resize_adjusts_worker_count() {
        let mut pool = SearchPool::new(2);
        assert_eq!(pool.len(), 2);
        pool.resize(4);
        assert_eq!(pool.len(), 4);
        pool.resize(1);
        assert_eq!(pool.len(), 1);
        pool.resize(0);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn pool_handles_multiple_consecutive_searches() {
        let pool = SearchPool::new(1);
        let tt = Arc::new(TranspositionTable::new(4));

        for _ in 0..3 {
            let stop = Arc::new(AtomicBool::new(false));
            let board = Board::from_fen(STARTING_FEN).unwrap();
            tt.new_generation();

            pool.start_helpers(&board, Arc::clone(&stop), Arc::clone(&tt), &[], None);
            thread::sleep(Duration::from_millis(40));
            stop.store(true, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(30));
        }

        drop(pool); // must join cleanly
    }
}
