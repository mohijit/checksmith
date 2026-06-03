//! Perft speed benchmarks.
//!
//! `perft(depth)` counts all leaf nodes in the move tree to a given depth.
//! Its speed measures raw move-generation throughput — the floor of any
//! engine's performance.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench perft
//! ```
//!
//! ## Interpretation
//!
//! * **Starting position perft(4)** — 197,281 nodes.  A reference machine
//!   doing 10 Mnps processes this in ~20 ms; 100 Mnps in ~2 ms.
//! * **Kiwipete perft(4)** — 4,085,603 nodes from a complex position that
//!   exercises en-passant, castling, and promotions.
//!
//! If either count diverges from the expected values, there is a bug in move
//! generation (and the `perft_startpos` / `perft_kiwipete` unit tests in
//! `movegen/legal.rs` will already have caught it).

use checksmith::board::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const STARTING_FEN: &str =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// A position rich in tactics: en-passant, castling, promotions, and pins.
/// Known as "Kiwipete" in the engine community.
const KIWIPETE_FEN: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

fn bench_perft_start_4(c: &mut Criterion) {
    let mut board = Board::from_fen(STARTING_FEN).unwrap();
    c.bench_function("perft/start/4", |b| {
        b.iter(|| {
            let (n, _) = black_box(&mut board).perft_divide(black_box(4));
            n
        })
    });
}

fn bench_perft_kiwipete_3(c: &mut Criterion) {
    let mut board = Board::from_fen(KIWIPETE_FEN).unwrap();
    c.bench_function("perft/kiwipete/3", |b| {
        b.iter(|| {
            let (n, _) = black_box(&mut board).perft_divide(black_box(3));
            n
        })
    });
}

criterion_group!(benches, bench_perft_start_4, bench_perft_kiwipete_3);
criterion_main!(benches);
