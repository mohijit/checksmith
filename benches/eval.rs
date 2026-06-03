//! Evaluation speed benchmarks.
//!
//! The static evaluator is called at every leaf node of the search tree.
//! Even a 10% speedup here compounds into a meaningful overall improvement.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench eval
//! ```
//!
//! ## Interpretation
//!
//! A typical evaluation call should take 50–200 ns on modern hardware (the
//! handcrafted evaluator is intentionally simple; NNUE is faster still due to
//! SIMD inference).  A regression above 300 ns suggests an accidentally
//! quadratic loop or a missing `#[inline]` on an inner function.

use checksmith::board::Board;
use checksmith::eval;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const STARTING_FEN: &str =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Complex middlegame with many active pieces.
const COMPLEX_FEN: &str =
    "r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bBP4/2N1PN2/PPQ2PPP/R1B2RK1 w - - 0 10";

/// Sparse endgame (fewer pieces → different code paths).
const ENDGAME_FEN: &str =
    "8/5pk1/8/2p5/2P5/8/5PK1/8 w - - 0 1";

fn bench_eval_start(c: &mut Criterion) {
    let board = Board::from_fen(STARTING_FEN).unwrap();
    c.bench_function("eval/start", |b| {
        b.iter(|| eval::evaluate(black_box(&board)))
    });
}

fn bench_eval_complex(c: &mut Criterion) {
    let board = Board::from_fen(COMPLEX_FEN).unwrap();
    c.bench_function("eval/complex", |b| {
        b.iter(|| eval::evaluate(black_box(&board)))
    });
}

fn bench_eval_endgame(c: &mut Criterion) {
    let board = Board::from_fen(ENDGAME_FEN).unwrap();
    c.bench_function("eval/endgame", |b| {
        b.iter(|| eval::evaluate(black_box(&board)))
    });
}

criterion_group!(benches, bench_eval_start, bench_eval_complex, bench_eval_endgame);
criterion_main!(benches);
