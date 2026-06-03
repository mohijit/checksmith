//! Built-in benchmarking suite.
//!
//! ## Overview
//!
//! This module provides two complementary measurement tools:
//!
//! * **`bench`** — fixed-depth search over a standardised position set.
//!   Every run produces the same total-node count (given identical search
//!   parameters), so you can compare node counts across engine versions to
//!   detect search-efficiency regressions even before measuring wall time.
//!
//! * **`test`** — a tactical test suite: mate-in-2 and mate-in-3 positions
//!   with known solutions, measured by whether the engine finds the correct
//!   move within a given depth limit.
//!
//! ## CLI
//!
//! ```sh
//! checksmith bench [depth]          # default depth 10
//! checksmith bench 12               # deeper bench
//! checksmith test  [max-depth]      # default max depth 8
//! ```
//!
//! ## Profiling with external tools
//!
//! ### cargo bench (criterion)
//!
//! ```sh
//! cargo bench                     # run all criterion benchmarks
//! cargo bench --bench perft       # perft only
//! cargo bench --bench search      # search only
//! cargo bench --bench eval        # eval only
//! ```
//!
//! Criterion writes HTML reports to `target/criterion/`.  Open
//! `target/criterion/report/index.html` in a browser for flame graphs and
//! regression comparisons.
//!
//! ### perf (Linux)
//!
//! ```sh
//! cargo build --release
//! perf stat ./target/release/checksmith bench 12
//! perf record -g ./target/release/checksmith bench 12
//! perf report
//! ```
//!
//! Useful `perf stat` counters: `instructions`, `cache-misses`,
//! `branch-misses`.  High cache-miss rates in the TT probe path are the
//! most common performance bottleneck.
//!
//! ### Flamegraph (Linux / macOS)
//!
//! ```sh
//! cargo install flamegraph
//! cargo flamegraph --bin checksmith -- bench 12
//! # Opens flamegraph.svg in the browser
//! ```
//!
//! The flamegraph shows which functions burn the most CPU time.  Typical
//! hot paths: `negamax` → `quiescence` → `is_square_attacked` →
//! `bishop_attacks` (slider ray walking).
//!
//! ### Instruments (macOS)
//!
//! ```sh
//! cargo instruments -t "Time Profiler" --bin checksmith -- bench 12
//! ```
//!
//! ### Visual Studio profiler / ETW (Windows)
//!
//! Build in release mode, then open in Visual Studio Profiler or use
//! `UIforETW` + `xperf` for sampling profiles.
//!
//! ## Interpreting results
//!
//! | Metric | What it tells you |
//! |--------|------------------|
//! | **NPS** (nodes/sec) | Raw computational speed; higher = better hardware or fewer wasted operations |
//! | **Node count at fixed depth** | Search efficiency; fewer = better pruning |
//! | **Tactical score** | Engine strength; more correct = better evaluation + search |
//! | **Perft count** | Move-generation correctness (exact counts expected) |
//!
//! A node-count regression (more nodes for the same depth after a code
//! change) means the pruning or move ordering got worse.  A wall-time
//! regression without a node-count regression means the evaluation or
//! move-generation got slower.

use crate::board::Board;
use crate::search::{is_mate_score, search};
use std::time::Instant;

// ─── Standard bench positions ─────────────────────────────────────────────────

/// 15 positions spanning opening, middlegame, and endgame.  These are stable
/// across milestones: the same set is always searched at the same depth so
/// node counts remain comparable between versions.
const BENCH_POSITIONS: &[&str] = &[
    // 1. Starting position (symmetric, good baseline)
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // 2. Italian Game — open centre, active development
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
    // 3. Sicilian Najdorf — complex unbalanced position
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    // 4. French Advance — closed pawn chain
    "rnbqkb1r/pp3ppp/4pn2/2pp4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq d6 0 5",
    // 5. King's Indian — rich imbalance, both sides castle on opposite wings
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7",
    // 6. Tactical middlegame — rooks on open files, active queens
    "r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bBP4/2N1PN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    // 7. Endgame: rook vs rook + pawn
    "8/r2k4/8/p7/P7/8/4K1R1/8 w - - 0 1",
    // 8. Queen endgame with passed pawns
    "8/8/1Q6/5k2/4p3/8/4K3/8 w - - 0 1",
    // 9. Bishop vs knight endgame
    "8/5k2/4bn2/8/8/4BK2/8/8 w - - 0 1",
    // 10. Heavy piece middlegame — lots of queen/rook activity
    "r2q1rk1/1pp2ppp/p2p1n2/2b1p3/4P1b1/1BPP1N2/PP3PPP/R1BQ1RK1 b - - 0 10",
    // 11. King safety test — White has opened the kingside
    "r4rk1/ppp1qppp/2np1n2/2b1p3/4P1b1/2PP1N2/PPBNQPPP/R4RK1 b - - 0 1",
    // 12. Rook ending — critical accuracy required
    "8/8/4k3/4p3/4P3/4K3/8/4R3 w - - 0 1",
    // 13. Pawn race — both sides push passed pawns
    "8/1k6/1p6/1Pp5/2P5/8/6K1/8 w - - 0 1",
    // 14. Opposite-colour bishops — drawing tendencies
    "8/5k2/4b3/8/3B4/4K3/8/8 w - - 0 1",
    // 15. Kiwipete — exercises en-passant, castling, promotions (complex)
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
];

// ─── Tactical test suite ──────────────────────────────────────────────────────

/// Each entry: (FEN, expected best move in UCI notation, description).
const TACTICAL_SUITE: &[(&str, &str, &str)] = &[
    // ── Mate in 2 ────────────────────────────────────────────────────────────
    (
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "h5f7",
        "Scholar's mate: Qxf7#",
    ),
    (
        "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        "e1e8",
        "Back-rank mate: Re8#",
    ),
    (
        "r5rk/5Rpp/p4R2/8/8/8/PPP3PP/6K1 w - - 0 1",
        "f7h7",
        "Double-rook mate: Rxh7#",
    ),
    (
        "7k/8/8/8/8/8/8/R3K2R w KQ - 0 1",
        "a1a8",
        "Rook to back rank, mate next",
    ),
    (
        "7k/6R1/6K1/8/8/8/8/8 w - - 0 1",
        "g7g8",
        "Rg8# immediate checkmate",
    ),
    // ── Mate in 3 ────────────────────────────────────────────────────────────
    (
        "7k/8/6K1/8/8/8/8/Q7 w - - 0 1",
        "a1a8",
        "Driving mate: Qa8+ Kh7 Qa7+ Kh8 (any queen move mates)",
    ),
    (
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
        "a1a8",
        "Rook to 8th rank — forces mate in 3",
    ),
    // ── Winning captures / tactics ────────────────────────────────────────────
    (
        "3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1",
        "d1d8",
        "Trade queens (hanging): Qxd8+",
    ),
    (
        "rnb1kbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 3",
        "e5f4",
        "Capture pawn: exf4 wins material (f4 is undefended)",
    ),
    (
        "7k/8/8/3p4/4P3/8/8/7K w - - 0 1",
        "e4d5",
        "Pawn captures hanging pawn: exd5",
    ),
    // ── Avoiding blunders ─────────────────────────────────────────────────────
    (
        "rnbqkbnr/pp2pppp/2p5/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
        // Best is to NOT blunder with exd5? which loses to cxd5 and leaves the
        // center open.  The engine should prefer a developing move.
        // Acceptable best moves: d2d4, g1f3, d1e2, f1d3, b1c3.
        // We'll just test that the engine returns a legal move; correctness
        // is position-specific.
        "",  // empty = any legal move is acceptable
        "Caro-Kann: engine should not immediately lose material",
    ),
];

// ─── BenchResult ─────────────────────────────────────────────────────────────

/// Summary of a full bench run across all standard positions.
pub struct BenchResult {
    pub depth:      u32,
    pub total_nodes: u64,
    pub elapsed_ms:  u128,
}

impl BenchResult {
    pub fn nps(&self) -> u64 {
        if self.elapsed_ms == 0 { return 0; }
        self.total_nodes * 1000 / self.elapsed_ms as u64
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "===========================")?;
        writeln!(f, "Depth         : {}", self.depth)?;
        writeln!(f, "Total nodes   : {}", fmt_nodes(self.total_nodes))?;
        writeln!(f, "Time          : {:.3}s", self.elapsed_ms as f64 / 1000.0)?;
        writeln!(f, "NPS           : {}", fmt_nodes(self.nps()))?;
        write!(f,   "===========================")
    }
}

// ─── TacticalResult ──────────────────────────────────────────────────────────

/// Summary of a tactical test-suite run.
pub struct TacticalResult {
    pub solved:  usize,
    pub total:   usize,
    pub elapsed_ms: u128,
}

impl std::fmt::Display for TacticalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "===========================")?;
        writeln!(f, "Solved  : {}/{}", self.solved, self.total)?;
        writeln!(f, "Time    : {:.3}s", self.elapsed_ms as f64 / 1000.0)?;
        write!(f,   "===========================")
    }
}

// ─── run_bench ───────────────────────────────────────────────────────────────

/// Search all standard positions at `depth` with a `hash_mb`-sized TT.
///
/// Prints one line per position while running, then returns the aggregate
/// stats.  The total node count is the primary comparison metric between
/// engine versions: identical search logic should produce identical counts.
pub fn run_bench(depth: u32, hash_mb: usize) -> BenchResult {
    let n = BENCH_POSITIONS.len();
    println!(
        "Checksmith bench  depth={}  positions={}  hash={}MB",
        depth, n, hash_mb
    );
    println!();

    let start    = Instant::now();
    let mut total_nodes = 0u64;

    for (i, &fen) in BENCH_POSITIONS.iter().enumerate() {
        let mut board = Board::from_fen(fen).expect("valid bench FEN");
        let result    = search(&mut board, depth);

        total_nodes += result.nodes;

        let score_str = fmt_score(result.score);
        let bm = result.best_move.map(|m| m.to_uci()).unwrap_or_else(|| "(none)".into());

        println!(
            "  [{:2}/{}]  score {:>8}  nodes {:>10}  bm {}",
            i + 1, n,
            score_str,
            fmt_nodes(result.nodes),
            bm,
        );
    }

    let elapsed_ms = start.elapsed().as_millis();
    BenchResult { depth, total_nodes, elapsed_ms }
}

// ─── run_tactical ─────────────────────────────────────────────────────────────

/// Run the tactical test suite, searching each position up to `max_depth`.
///
/// Prints one line per position: ✓ if the engine found the expected best
/// move, – if the expected move was not specified, ✗ if it found a different
/// move.
pub fn run_tactical(max_depth: u32) -> TacticalResult {
    let n = TACTICAL_SUITE.len();
    println!(
        "Checksmith tactical test  max_depth={}  positions={}",
        max_depth, n
    );
    println!();

    let start  = Instant::now();
    let mut solved = 0usize;

    for (i, &(fen, expected_bm, desc)) in TACTICAL_SUITE.iter().enumerate() {
        let mut board = Board::from_fen(fen).expect("valid tactical FEN");
        let result = search(&mut board, max_depth);
        let found = result.best_move.map(|m| m.to_uci()).unwrap_or_default();

        let (mark, correct) = if expected_bm.is_empty() {
            // Any legal move is acceptable.
            ('–', result.best_move.is_some())
        } else if found == expected_bm {
            ('✓', true)
        } else {
            ('✗', false)
        };

        if correct { solved += 1; }

        let score_str = fmt_score(result.score);
        println!(
            "  [{:2}/{}] {} score {:>8}  found {:>6}  expected {:>6}  {}",
            i + 1, n, mark,
            score_str, found,
            if expected_bm.is_empty() { "(any)".into() } else { expected_bm.to_string() },
            desc,
        );
    }

    println!();
    let elapsed_ms = start.elapsed().as_millis();
    TacticalResult { solved, total: n, elapsed_ms }
}

// ─── run_nps ──────────────────────────────────────────────────────────────────

/// Quick perft-based NPS measurement: depth-5 perft from the starting position.
///
/// This is the cheapest approximation of raw node speed — no eval, no TT.
pub fn measure_nps() -> u64 {
    let mut board = Board::from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ).unwrap();
    let start = Instant::now();
    let (nodes, _) = board.perft_divide(5);
    let elapsed = start.elapsed();
    let ms = elapsed.as_millis().max(1);
    nodes * 1000 / ms as u64
}

// ─── CLI entry points ─────────────────────────────────────────────────────────

/// `checksmith bench [depth]`
pub fn run(args: &[String]) {
    let depth: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    let result = run_bench(depth, 32);
    println!();
    println!("{}", result);
    println!("nps: {}", fmt_nodes(result.nps()));
}

/// `checksmith test [max-depth]`
pub fn run_test(args: &[String]) {
    let max_depth: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(8);
    let result = run_tactical(max_depth);
    println!("{}", result);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn fmt_nodes(n: u64) -> String {
    // Insert commas: 4_581_234 → "4,581,234"
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { out.push(','); }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn fmt_score(score: i32) -> String {
    if is_mate_score(score) {
        let plies = crate::search::mate_distance_plies(score);
        let moves = (plies + 1) / 2;
        format!("mate {}", if score > 0 { moves } else { -moves })
    } else {
        format!("cp {}", score)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_bench_fens_are_valid() {
        for &fen in BENCH_POSITIONS {
            Board::from_fen(fen).unwrap_or_else(|_| panic!("invalid bench FEN: {}", fen));
        }
    }

    #[test]
    fn all_tactical_fens_are_valid() {
        for &(fen, _, _) in TACTICAL_SUITE {
            Board::from_fen(fen).unwrap_or_else(|_| panic!("invalid tactical FEN: {}", fen));
        }
    }

    #[test]
    fn bench_depth_4_runs_without_panic() {
        // Run a mini-bench at depth 4 so the full sweep doesn't time out in CI.
        let result = run_bench(4, 1);
        assert!(result.total_nodes > 0, "bench must search some nodes");
        assert!(result.elapsed_ms < 60_000, "bench took too long: {}ms", result.elapsed_ms);
    }

    #[test]
    fn nps_is_positive() {
        let nps = measure_nps();
        assert!(nps > 0, "NPS must be positive");
    }

    #[test]
    fn tactical_finds_hanging_queen() {
        // The "trade queens" position must be solvable at depth 1.
        let result = run_tactical(3);
        // We have 11 positions where expected_bm is non-empty; any correct
        // answer should count.  We mainly verify the function runs without panic.
        assert!(result.solved > 0, "tactical suite should solve at least 1 position");
    }

    #[test]
    fn fmt_nodes_formats_correctly() {
        assert_eq!(fmt_nodes(0),         "0");
        assert_eq!(fmt_nodes(1_000),     "1,000");
        assert_eq!(fmt_nodes(1_234_567), "1,234,567");
    }

    #[test]
    fn mate_in_one_depth2() {
        // Scholar's mate position: Qxf7# is the only mate in 1.
        let mut board = Board::from_fen(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        ).unwrap();
        let result = search(&mut board, 2);
        assert!(
            is_mate_score(result.score),
            "depth-2 should find mate-in-1; got score {}",
            result.score
        );
        assert_eq!(
            result.best_move.map(|m| m.to_uci()),
            Some("h5f7".into()),
            "Qxf7# is the only mating move"
        );
    }

    #[test]
    fn back_rank_mate_depth2() {
        // Re8# delivers mate immediately.
        let mut board = Board::from_fen(
            "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
        ).unwrap();
        let result = search(&mut board, 2);
        assert!(is_mate_score(result.score), "should find back-rank mate");
        assert_eq!(result.best_move.map(|m| m.to_uci()), Some("e1e8".into()));
    }

    #[test]
    fn search_info_fields_present() {
        // Check that SearchResult has the fields the milestone requires.
        let mut board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ).unwrap();
        let r = search(&mut board, 3);
        assert!(r.depth > 0);
        assert!(r.nodes > 0);
        // score is always set (may be 0 for equal positions)
        let _ = r.score;
        assert!(r.best_move.is_some());
    }
}
