//! The UCI (Universal Chess Interface) protocol loop.
//!
//! UCI is the text protocol that chess GUIs (Cute Chess, Arena, BanksiaGUI, …)
//! use to talk to engines over stdin/stdout.  The GUI sends commands; the engine
//! replies with lines like `info ...` and `bestmove ...`.  A minimal session:
//!
//! ```text
//! > uci
//! < id name Checksmith 0.9.0
//! < id author Checksmith contributors
//! < option name Hash type spin default 256 min 1 max 1024
//! < option name Move Overhead type spin default 30 min 0 max 5000
//! < option name Threads type spin default 1 min 1 max <cpu_count>
//! < option name Clear Hash type button
//! < option name Ponder type check default false
//! < option name MultiPV type spin default 1 min 1 max 500
//! < uciok
//! > isready
//! < readyok
//! > position startpos moves e2e4
//! > go wtime 300000 btime 300000
//! < info depth 1 seldepth 2 score cp 30 nodes 21 nps 21000 time 1 hashfull 0 pv e7e5
//! < bestmove e7e5 ponder c7c5
//! ```
//!
//! ## Options supported
//!
//! | Name | Type | Default | Notes |
//! |------|------|---------|-------|
//! | Hash | spin | 16 MB | Transposition table size |
//! | Move Overhead | spin | 30 ms | Subtracted from time budget each move |
//! | Threads | spin | 1 | Number of search threads (Lazy SMP) |
//! | Clear Hash | button | — | Clears the transposition table |
//! | Ponder | check | false | Enables ponder mode (bestmove includes ponder hint) |
//! | MultiPV | spin | 1 | Number of principal variations to search |
//!
//! ## Parallel search (Lazy SMP)
//!
//! When `Threads > 1`, the engine uses Lazy SMP (see [`crate::search::smp`]):
//! * Thread 0 (main): iterative deepening with real time limits; reports info.
//! * Threads 1..N-1 (helpers): infinite search, filling the shared TT.
//! * All threads share one `Arc<TranspositionTable>`.
//!
//! The TT is allocated once and shared for the engine's lifetime. Setting a new
//! `Hash` value replaces the `Arc` (old one is dropped when all threads finish).
//!
//! ## Background thread and ponder
//!
//! `go` spawns the search on its own thread so the main loop can read `stop`
//! without blocking.  `stop` / `quit` flip the shared atomic flag.
//!
//! **Ponder** (`go ponder`): the engine searches with no time pressure while
//! waiting for the opponent's move.  If the GUI sends `ponderhit`, we know the
//! opponent played our predicted move; we stop the ponder search and start a
//! fresh timed search.  If the GUI sends `stop`, the opponent played something
//! else; we discard the result.
//!
//! **MultiPV**: each PV line runs a fresh search excluding the best moves found
//! in earlier lines.  The `info` line for each includes `multipv N`.

pub mod parser;

use crate::board::{Board, STARTING_FEN};
use crate::book::OpeningBook;
use crate::movegen::Move;
use crate::nnue::network::Network;
use crate::search::{
    is_mate_score, mate_distance_plies, think, SearchInfo, SearchLimits, TranspositionTable,
};
use crate::search::thread_pool::SearchPool;
use crate::tablebase;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

const NAME: &str = concat!("Checksmith ", env!("CARGO_PKG_VERSION"));
const AUTHOR: &str = "Checksmith contributors";

// ── Option ranges ─────────────────────────────────────────────────────────────

const DEFAULT_HASH_MB: usize  = 256;
const MIN_HASH_MB: usize      = 1;
const MAX_HASH_MB: usize      = 1024;

const DEFAULT_OVERHEAD_MS: u64 = 30;
const MIN_OVERHEAD_MS: u64     = 0;
const MAX_OVERHEAD_MS: u64     = 5_000;

const MIN_MULTIPV: usize = 1;
const MAX_MULTIPV: usize = 500;

const DEFAULT_OWN_BOOK: bool  = true;
const DEFAULT_BOOK_FILE: &str = "";

const DEFAULT_SYZYGY_PIECES: u32 = 6;
const MIN_SYZYGY_PIECES: u32     = 3;
const MAX_SYZYGY_PIECES: u32     = 7;

// ── Engine state ──────────────────────────────────────────────────────────────

/// All mutable engine state shared across UCI commands.
struct Engine {
    board: Board,
    stop:  Arc<AtomicBool>,

    /// Running search thread (`None` when idle). Returns `()` — the TT stays
    /// alive via the `Arc` on `Engine`.
    search: Option<JoinHandle<()>>,

    /// Shared transposition table. Wrapped in `Arc` so search threads can hold
    /// a clone without the main thread losing ownership.
    tt: Arc<TranspositionTable>,

    /// Persistent pool of Lazy SMP helper threads.
    ///
    /// Holds `num_threads - 1` workers, each waiting on an `mpsc` channel.
    /// `start_search` dispatches helper searches via `pool.start_helpers()`
    /// before launching the main search thread.  Workers stop automatically
    /// when the main search sets the shared stop flag and re-enter their wait
    /// loop, ready for the next search — no per-search spawn/join overhead.
    pool: SearchPool,

    // ── Tunable options ────────────────────────────────────────────────────
    /// Milliseconds subtracted from the clock each move to cover GUI overhead.
    move_overhead_ms: u64,
    /// Number of search threads. 1 = single-threaded; >1 = Lazy SMP.
    num_threads: usize,
    /// Number of principal variations to report (≥ 1).
    multi_pv: usize,
    /// Whether ponder mode is enabled (controls `bestmove … ponder …` output).
    ponder_enabled: bool,
    /// True while the engine is running a ponder search.
    pondering: bool,
    /// The position that was set just before `go ponder` — needed to restart
    /// a timed search after `ponderhit`.
    ponder_board: Option<Board>,
    /// The time limits from the last `go ponder` command.
    ponder_limits: Option<SearchLimits>,

    // ── Book options ───────────────────────────────────────────────────────
    /// Whether book play is enabled.
    own_book: bool,
    /// Path to a Polyglot `.bin` file, or empty string for the internal book.
    book_file: String,
    /// The opening book (internal or external).
    book: OpeningBook,

    // ── NNUE evaluator ─────────────────────────────────────────────────────
    /// Loaded NNUE network, or `None` when the handcrafted evaluator is active.
    /// Set via `setoption name EvalFile value <path>`.
    nnue_network: Option<Arc<Network>>,

    // ── Repetition detection ───────────────────────────────────────────────
    /// Zobrist hashes of all positions that have appeared in the game so far,
    /// excluding the current board position.  Passed to `think()` so the
    /// search can detect threefold repetition and avoid perpetual-check traps.
    game_history: Vec<u64>,
}

impl Engine {
    fn new() -> Engine {
        Engine {
            board: Board::from_fen(STARTING_FEN).expect("valid start FEN"),
            stop:  Arc::new(AtomicBool::new(false)),
            search: None,
            tt:     Arc::new(TranspositionTable::new(DEFAULT_HASH_MB)),
            pool:   SearchPool::new(0), // 1 thread = 0 helpers; resized on setoption Threads
            move_overhead_ms: DEFAULT_OVERHEAD_MS,
            num_threads: 1,
            multi_pv: 1,
            ponder_enabled: false,
            pondering: false,
            ponder_board: None,
            ponder_limits: None,
            own_book:  DEFAULT_OWN_BOOK,
            book_file: DEFAULT_BOOK_FILE.to_string(),
            book:      OpeningBook::new_internal(),
            nnue_network: None,
            game_history: Vec::new(),
        }
    }

    // ── Command dispatch ───────────────────────────────────────────────────

    /// Handle one UCI command line.  Returns `false` to exit the loop.
    fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&command) = tokens.first() else { return true; };

        match command {
            "uci"        => self.cmd_uci(),
            "isready"    => send("readyok"),
            "setoption"  => self.cmd_setoption(&tokens[1..]),
            "ucinewgame" => self.cmd_ucinewgame(),
            "position"   => self.cmd_position(&tokens[1..]),
            "go"         => self.cmd_go(&tokens[1..]),
            "stop"       => self.cmd_stop(),
            "ponderhit"  => self.cmd_ponderhit(),
            "quit"       => { self.reclaim_search(); return false; }
            "d"          => {
                println!("{}", self.board);
                println!("FEN: {}", self.board.to_fen());
            }
            _ => {} // per spec: ignore unknown commands silently
        }
        true
    }

    // ── Individual command handlers ────────────────────────────────────────

    fn cmd_uci(&self) {
        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);

        send(&format!("id name {}", NAME));
        send(&format!("id author {}", AUTHOR));
        send(&format!(
            "option name Hash type spin default {} min {} max {}",
            DEFAULT_HASH_MB, MIN_HASH_MB, MAX_HASH_MB
        ));
        send(&format!(
            "option name Move Overhead type spin default {} min {} max {}",
            DEFAULT_OVERHEAD_MS, MIN_OVERHEAD_MS, MAX_OVERHEAD_MS
        ));
        send(&format!(
            "option name Threads type spin default 1 min 1 max {}",
            max_threads
        ));
        send("option name Clear Hash type button");
        send("option name Ponder type check default false");
        send(&format!(
            "option name MultiPV type spin default 1 min {} max {}",
            MIN_MULTIPV, MAX_MULTIPV
        ));
        send(&format!(
            "option name OwnBook type check default {}",
            DEFAULT_OWN_BOOK
        ));
        send(&format!(
            "option name BookFile type string default {}",
            if DEFAULT_BOOK_FILE.is_empty() { "<empty>" } else { DEFAULT_BOOK_FILE }
        ));
        send("option name EvalFile type string default <empty>");
        send("option name SyzygyPath type string default <empty>");
        send(&format!(
            "option name SyzygyPieces type spin default {} min {} max {}",
            DEFAULT_SYZYGY_PIECES, MIN_SYZYGY_PIECES, MAX_SYZYGY_PIECES
        ));
        // SPSA-tunable search parameters.
        for p in crate::search::params::ALL_PARAMS {
            send(&format!(
                "option name {} type spin default {} min {} max {}",
                p.name, p.default, p.min, p.max
            ));
        }
        send("uciok");
    }

    fn cmd_ucinewgame(&mut self) {
        self.reclaim_search();
        self.board = Board::from_fen(STARTING_FEN).unwrap();
        self.pondering = false;
        self.game_history.clear();
        self.tt.clear();
    }

    fn cmd_position(&mut self, args: &[&str]) {
        self.reclaim_search();
        if let Some(board) = parser::parse_position(args) {
            self.game_history = build_game_history(args);
            self.board = board;
        }
    }

    fn cmd_go(&mut self, args: &[&str]) {
        self.reclaim_search();
        self.stop.store(false, Ordering::Relaxed);

        let mut limits = parser::parse_go(args);

        // Try the opening book first — skip search if we get a hit.
        // Do not query the book during `go ponder` (the position may not be in
        // book, and ponder expects us to keep thinking).
        if self.own_book && !limits.ponder {
            let mut rng = rand::thread_rng();
            if let Some(mv) = self.book.pick_move(&self.board, &mut rng) {
                send(&format!("bestmove {}", mv.to_uci()));
                return;
            }
        }

        // Subtract move overhead from clock times so we never flag.
        if let Some(t) = limits.wtime.as_mut() {
            *t = t.saturating_sub(self.move_overhead_ms);
        }
        if let Some(t) = limits.btime.as_mut() {
            *t = t.saturating_sub(self.move_overhead_ms);
        }

        // Ponder search: run with no time pressure; save state for ponderhit.
        if limits.ponder {
            self.pondering = true;
            self.ponder_board = Some(self.board.clone());
            self.ponder_limits = Some(limits.clone());
        } else {
            self.pondering = false;
        }

        self.start_search(self.board.clone(), limits, self.multi_pv, self.ponder_enabled);
    }

    fn cmd_stop(&mut self) {
        self.pondering = false;
        self.stop.store(true, Ordering::Relaxed);
        // The search thread will print bestmove when it sees the flag.
    }

    fn cmd_ponderhit(&mut self) {
        if !self.pondering {
            return;
        }
        // The opponent played our predicted move.  Stop the ponder search;
        // the GUI is expected to send a proper `go` command with real time
        // limits, which we will service in cmd_go.
        self.pondering = false;
        self.stop.store(true, Ordering::Relaxed);
        self.reclaim_search();

        // Restart with real time limits from the saved ponder command.
        if let (Some(board), Some(mut limits)) =
            (self.ponder_board.take(), self.ponder_limits.take())
        {
            limits.ponder = false;
            self.board = board.clone();
            self.stop.store(false, Ordering::Relaxed);
            self.start_search(board, limits, self.multi_pv, self.ponder_enabled);
        }
    }

    fn cmd_setoption(&mut self, tokens: &[&str]) {
        // Syntax: name <words…> value <words…>   (or just: name <button>)
        let name_pos  = tokens.iter().position(|&t| t == "name");
        let value_pos = tokens.iter().position(|&t| t == "value");
        let Some(np) = name_pos else { return; };

        let name_end = value_pos.unwrap_or(tokens.len());
        let name = tokens[np + 1..name_end].join(" ");

        if name.eq_ignore_ascii_case("Clear Hash") {
            self.reclaim_search();
            self.tt.clear();
            return;
        }

        let Some(vp) = value_pos else { return; };
        let value = tokens[vp + 1..].join(" ");

        if name.eq_ignore_ascii_case("Hash") {
            if let Ok(mb) = value.parse::<usize>() {
                let mb = mb.clamp(MIN_HASH_MB, MAX_HASH_MB);
                self.reclaim_search();
                self.tt = Arc::new(TranspositionTable::new(mb));
            }
        } else if name.eq_ignore_ascii_case("Move Overhead") {
            if let Ok(ms) = value.parse::<u64>() {
                self.move_overhead_ms = ms.clamp(MIN_OVERHEAD_MS, MAX_OVERHEAD_MS);
            }
        } else if name.eq_ignore_ascii_case("Threads") {
            if let Ok(n) = value.parse::<usize>() {
                let max = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .max(1);
                self.num_threads = n.clamp(1, max);
                // Resize the persistent helper pool: main thread + (N-1) helpers.
                self.pool.resize(self.num_threads.saturating_sub(1));
            }
        } else if name.eq_ignore_ascii_case("Ponder") {
            self.ponder_enabled = value.eq_ignore_ascii_case("true");
        } else if name.eq_ignore_ascii_case("MultiPV") {
            if let Ok(n) = value.parse::<usize>() {
                self.multi_pv = n.clamp(MIN_MULTIPV, MAX_MULTIPV);
            }
        } else if name.eq_ignore_ascii_case("OwnBook") {
            self.own_book = value.eq_ignore_ascii_case("true");
        } else if name.eq_ignore_ascii_case("BookFile") {
            self.book_file = value.trim().to_string();
            // Reload the book immediately so future `go` commands use it.
            self.book = if self.book_file.is_empty() {
                OpeningBook::new_internal()
            } else {
                OpeningBook::load(&self.book_file)
            };
        } else if name.eq_ignore_ascii_case("EvalFile") {
            let path = value.trim();
            if path.is_empty() || path == "<empty>" {
                self.nnue_network = None;
                send("info string NNUE disabled; using handcrafted evaluator");
            } else {
                match Network::load(std::path::Path::new(path)) {
                    Ok(net) => {
                        self.nnue_network = Some(Arc::new(net));
                        send(&format!("info string NNUE loaded from {}", path));
                    }
                    Err(e) => {
                        send(&format!("info string NNUE load failed ({}); using handcrafted evaluator", e));
                    }
                }
            }
        } else if name.eq_ignore_ascii_case("SyzygyPath") {
            let path = value.trim();
            if path.is_empty() || path == "<empty>" {
                tablebase::init("");
                send("info string Syzygy tablebases disabled");
            } else {
                let n = tablebase::init(path);
                if n > 0 {
                    send(&format!("info string Syzygy: loaded {} table(s) from {}", n, path));
                } else {
                    send(&format!("info string Syzygy: no tables found in {}", path));
                }
            }
        } else if name.eq_ignore_ascii_case("SyzygyPieces") {
            if let Ok(n) = value.parse::<u32>() {
                tablebase::set_piece_limit(n);
            }
        } else if let Ok(v) = value.parse::<i32>() {
            // SPSA-tunable search parameters: match by name.
            for p in crate::search::params::ALL_PARAMS {
                if name.eq_ignore_ascii_case(p.name) {
                    (p.set)(v);
                    break;
                }
            }
        }
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Launch the search on a background thread.
    ///
    /// The TT is shared via `Arc::clone` — the main thread retains its own
    /// Arc so `clear()` and `hashfull()` remain accessible while searching.
    fn start_search(
        &mut self,
        board: Board,
        limits: SearchLimits,
        multi_pv: usize,
        ponder_enabled: bool,
    ) {
        let stop = self.stop.clone();
        let game_history = self.game_history.clone();
        let tt = Arc::clone(&self.tt);
        let nnue = self.nnue_network.as_ref().map(Arc::clone);

        self.tt.new_generation();

        // Dispatch helper searches to the persistent thread pool before the
        // main search starts.  Helpers fill the shared TT with speculative
        // results; they stop automatically when stop is set to true.
        if self.pool.len() > 0 {
            self.pool.start_helpers(
                &board,
                Arc::clone(&stop),
                Arc::clone(&tt),
                &game_history,
                nnue.as_ref().map(Arc::clone),
            );
        }

        let builder = thread::Builder::new().stack_size(16 * 1024 * 1024);
        self.search = Some(
            builder
                .spawn(move || {
                    run_search_thread(
                        board,
                        limits,
                        stop,
                        tt,
                        game_history,
                        multi_pv,
                        ponder_enabled,
                        nnue,
                    );
                })
                .expect("failed to spawn search thread"),
        );
    }

    /// Stop any running search and wait for it to finish.
    ///
    /// Unlike the previous design the TT is not "reclaimed" — it lives on the
    /// `Engine` via `Arc` and is always accessible.
    fn reclaim_search(&mut self) {
        if let Some(handle) = self.search.take() {
            self.stop.store(true, Ordering::Relaxed);
            let _ = handle.join();
        }
    }
}

// ── Search thread entry point ─────────────────────────────────────────────────

/// Drives one or more PV searches and emits `bestmove` on completion.
///
/// When `num_threads > 1`, Lazy SMP helper threads are spawned at the start
/// and share `tt`. Helpers run with infinite time and are stopped once all PV
/// iterations have completed (or the external stop flag is set).
///
/// For MultiPV > 1 the function runs `multi_pv` successive searches, each
/// time excluding the best moves from earlier lines.  The `info` output of
/// each search includes `multipv N` so GUIs can distinguish the lines.
fn run_search_thread(
    mut board:      Board,
    limits:         SearchLimits,
    stop:           Arc<AtomicBool>,
    tt:             Arc<TranspositionTable>,
    game_history:   Vec<u64>,
    multi_pv:       usize,
    ponder_enabled: bool,
    nnue:           Option<Arc<Network>>,
) {
    let multi_pv = multi_pv.max(1);

    // Helper threads are managed by the persistent SearchPool in Engine and
    // were already dispatched before this function was called.  We only need
    // to run the main (timed) search here.

    let mut excluded: Vec<Move> = Vec::new();
    let mut first_best_move: Option<Move> = None;
    let mut first_pv: Vec<Move> = Vec::new();

    for pv_num in 1..=multi_pv {
        if stop.load(Ordering::Relaxed) { break; }

        let pv_num_cap = pv_num;
        let multi_pv_cap = multi_pv;

        let mut pv_limits = limits.clone();
        pv_limits.excluded_root_moves = excluded.clone();

        let result = think(
            &mut board,
            &pv_limits,
            stop.clone(),
            &tt,
            &game_history,
            nnue.as_ref().map(Arc::clone),
            move |info| print_info(&info, pv_num_cap, multi_pv_cap),
        );

        match result.best_move {
            Some(mv) => {
                if pv_num == 1 {
                    first_best_move = Some(mv);
                    first_pv = extract_pv_from_tt(&mut board, &tt, result.depth as usize);
                }
                excluded.push(mv);
            }
            None => break,
        }
    }

    // Signal helpers to stop.  Pool workers poll the stop flag every ~4096
    // nodes and will return to their wait loop within a few milliseconds.
    stop.store(true, Ordering::Relaxed);

    // Emit `bestmove`, optionally followed by a ponder hint.
    let best_str = first_best_move
        .map(|m| m.to_uci())
        .unwrap_or_else(|| "0000".into());

    if ponder_enabled {
        // The second move of the PV is the engine's predicted opponent response.
        let ponder_hint = first_pv.get(1).map(|m| m.to_uci());
        match ponder_hint {
            Some(pm) => send(&format!("bestmove {} ponder {}", best_str, pm)),
            None     => send(&format!("bestmove {}", best_str)),
        }
    } else {
        send(&format!("bestmove {}", best_str));
    }
}

/// Extract up to `max_len` moves from the TT starting from `board`'s position.
/// Restores the board to its original state.
fn extract_pv_from_tt(board: &mut Board, tt: &TranspositionTable, max_len: usize) -> Vec<Move> {
    let mut pv = Vec::new();
    let mut undos = Vec::new();

    while pv.len() < max_len {
        let Some(mv) = tt.probe(board.hash).and_then(|d| d.best_move()) else { break; };
        if !board.legal_moves().iter().any(|&m| m == mv) { break; }
        undos.push((mv, board.make_move(mv)));
        pv.push(mv);
    }
    while let Some((mv, undo)) = undos.pop() {
        board.unmake_move(mv, undo);
    }
    pv
}

// ── Output helpers ────────────────────────────────────────────────────────────

/// Format and emit an `info` line.
///
/// When `multi_pv_total > 1`, includes `multipv N` in the output so GUIs
/// can assign each line to the correct PV slot.
fn print_info(info: &SearchInfo, pv_num: usize, multi_pv_total: usize) {
    let score = if is_mate_score(info.score) {
        let plies = mate_distance_plies(info.score);
        let moves = (plies + 1) / 2;
        format!("mate {}", if info.score > 0 { moves } else { -moves })
    } else {
        format!("cp {}", info.score)
    };

    let nps = if info.time_ms > 0 {
        info.nodes as u128 * 1000 / info.time_ms
    } else {
        0
    };

    let pv: Vec<String> = info.pv.iter().map(|m| m.to_uci()).collect();

    let multipv_field = if multi_pv_total > 1 {
        format!(" multipv {}", pv_num)
    } else {
        String::new()
    };

    let tbhits_field = if info.tbhits > 0 {
        format!(" tbhits {}", info.tbhits)
    } else {
        String::new()
    };

    send(&format!(
        "info depth {} seldepth {}{} score {} nodes {} nps {} time {} hashfull {}{} pv {}",
        info.depth,
        info.seldepth,
        multipv_field,
        score,
        info.nodes,
        nps,
        info.time_ms,
        info.hashfull,
        tbhits_field,
        if pv.is_empty() { "(none)".into() } else { pv.join(" ") },
    ));
}

/// Write a line to stdout and flush immediately.
fn send(msg: &str) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{}", msg);
    let _ = out.flush();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine { Engine::new() }

    #[test]
    fn default_options_match_constants() {
        let e = engine();
        assert_eq!(e.move_overhead_ms, DEFAULT_OVERHEAD_MS);
        assert_eq!(e.num_threads, 1);
        assert_eq!(e.multi_pv, 1);
        assert!(!e.ponder_enabled);
        assert_eq!(e.own_book, DEFAULT_OWN_BOOK);
    }

    #[test]
    fn setoption_move_overhead_stores_value() {
        let mut e = engine();
        e.handle("setoption name Move Overhead value 50");
        assert_eq!(e.move_overhead_ms, 50);
    }

    #[test]
    fn setoption_move_overhead_clamps_to_range() {
        let mut e = engine();
        e.handle("setoption name Move Overhead value 99999");
        assert_eq!(e.move_overhead_ms, MAX_OVERHEAD_MS);
    }

    #[test]
    fn setoption_multipv_stores_value() {
        let mut e = engine();
        e.handle("setoption name MultiPV value 3");
        assert_eq!(e.multi_pv, 3);
    }

    #[test]
    fn setoption_ponder_enables_flag() {
        let mut e = engine();
        e.handle("setoption name Ponder value true");
        assert!(e.ponder_enabled);
        e.handle("setoption name Ponder value false");
        assert!(!e.ponder_enabled);
    }

    #[test]
    fn setoption_threads_stores_value() {
        let mut e = engine();
        // Setting Threads = 2 is accepted; value is clamped to available CPUs.
        e.handle("setoption name Threads value 2");
        assert!(e.num_threads >= 1, "num_threads must be at least 1");
        // Setting Threads = 1 always works.
        e.handle("setoption name Threads value 1");
        assert_eq!(e.num_threads, 1);
    }

    #[test]
    fn setoption_clear_hash_does_not_panic() {
        let mut e = engine();
        e.handle("setoption name Clear Hash");
        // TT is always accessible via Arc; no panic expected.
    }

    #[test]
    fn setoption_unknown_name_is_silently_ignored() {
        let mut e = engine();
        e.handle("setoption name NonExistentOption value 99");
        // Nothing should change; no panic.
        assert_eq!(e.move_overhead_ms, DEFAULT_OVERHEAD_MS);
    }

    #[test]
    fn unknown_command_returns_true_and_does_not_quit() {
        let mut e = engine();
        assert!(e.handle("xyzzy"));
        assert!(e.handle(""));
        assert!(e.handle("  "));
    }

    #[test]
    fn quit_command_returns_false() {
        let mut e = engine();
        assert!(!e.handle("quit"));
    }

    #[test]
    fn ucinewgame_resets_board_and_pondering() {
        let mut e = engine();
        e.pondering = true;
        e.handle("ucinewgame");
        assert!(!e.pondering);
        assert_eq!(
            e.board.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    #[test]
    fn setoption_own_book_toggles() {
        let mut e = engine();
        e.handle("setoption name OwnBook value false");
        assert!(!e.own_book);
        e.handle("setoption name OwnBook value true");
        assert!(e.own_book);
    }

    #[test]
    fn setoption_book_file_empty_uses_internal() {
        let mut e = engine();
        // Setting a non-existent file path should fall back to internal book.
        e.handle("setoption name BookFile value nonexistent_file.bin");
        // Engine should still have a book (the internal fallback).
        assert!(e.own_book); // default remains unchanged
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Replay the moves in a `position` command to collect position hashes.
///
/// Returns the Zobrist hash of every position from the start up to (but NOT
/// including) the final position.  The resulting slice is passed to `think()`
/// as the game history for repetition detection.
fn build_game_history(tokens: &[&str]) -> Vec<u64> {
    let mut board;
    let moves_start;

    match tokens.first() {
        Some(&"startpos") => {
            board = Board::from_fen(STARTING_FEN).unwrap();
            moves_start = 1;
        }
        Some(&"fen") if tokens.len() >= 7 => {
            let fen = tokens[1..7].join(" ");
            board = match Board::from_fen(&fen) {
                Ok(b) => b,
                Err(_) => return Vec::new(),
            };
            moves_start = 7;
        }
        _ => return Vec::new(),
    }

    let mut history = Vec::new();
    if let Some(pos) = tokens[moves_start..].iter().position(|&t| t == "moves") {
        let first_move = moves_start + pos + 1;
        for &mv_str in &tokens[first_move..] {
            history.push(board.hash);
            if let Some(mv) = parser::parse_uci_move(&board, mv_str) {
                board.make_move(mv);
            } else {
                break;
            }
        }
    }
    history
}

/// Run the UCI loop, reading commands from stdin until `quit` or EOF.
pub fn run() {
    let mut engine = Engine::new();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if !engine.handle(line.trim()) { break; }
    }
    engine.reclaim_search();
}
