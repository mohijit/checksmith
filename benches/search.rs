//! Search speed benchmarks.
//!
//! These measure the full search pipeline (negamax + alpha-beta + PVS +
//! NMP + LMR + quiescence + TT) at fixed depths.  Unlike perft, the node
//! count here depends on the evaluation and move-ordering quality.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench search
//! ```
//!
//! ## Interpretation
//!
//! * **depth 6** is a warm-up: only a few hundred milliseconds at typical
//!   engine speeds.  A regression here shows up immediately.
//! * **depth 8** exercises the TT and history tables more deeply.  Expect
//!   ~50–300 Mnps depending on hardware and TT hit rate.
//!
//! Comparing `search/d6/start` across commits is the fastest way to detect
//! search-efficiency regressions (fewer nodes = better ordering = faster).

use checksmith::board::Board;
use checksmith::search::search;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const STARTING_FEN: &str =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// A tactical middlegame rich in queen and rook activity.
const TACTICAL_FEN: &str =
    "r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bBP4/2N1PN2/PPQ2PPP/R1B2RK1 w - - 0 10";

fn bench_search_start_d6(c: &mut Criterion) {
    let mut board = Board::from_fen(STARTING_FEN).unwrap();
    c.bench_function("search/d6/start", |b| {
        b.iter(|| search(black_box(&mut board), black_box(6)))
    });
}

fn bench_search_tactical_d6(c: &mut Criterion) {
    let mut board = Board::from_fen(TACTICAL_FEN).unwrap();
    c.bench_function("search/d6/tactical", |b| {
        b.iter(|| search(black_box(&mut board), black_box(6)))
    });
}

criterion_group!(benches, bench_search_start_d6, bench_search_tactical_d6);
criterion_main!(benches);
