//! Hardcoded internal opening book.
//!
//! Positions are stored as (FEN, UCI-move, weight) triples and converted
//! to `PolyEntry` format at startup using the same `poly_hash` function
//! that processes external Polyglot files.  This keeps the internal and
//! external books on exactly the same code path.

use super::polyglot::{encode_move, poly_hash, PolyEntry};
use crate::board::Board;
use crate::uci::parser::parse_move;

/// Raw internal book data: (position FEN, UCI move, weight).
///
/// Weights follow the same convention as Polyglot: higher weight = played
/// more often.  Equal weights produce uniform selection.
static INTERNAL_LINES: &[(&str, &str, u16)] = &[
    // ── Starting position ─────────────────────────────────────────────────
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", 10),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4", 10),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3",  5),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "c2c4",  5),

    // ── After 1.e4 (black responses) ──────────────────────────────────────
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5", 10),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "c7c5",  8),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e6",  6),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "c7c6",  4),

    // ── After 1.d4 (black responses) ──────────────────────────────────────
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "d7d5",  8),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "g8f6",  8),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "e7e6",  5),
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "c7c5",  4),

    // ── After 1.e4 e5 (white second moves) ────────────────────────────────
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "g1f3", 10),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "f1c4",  5),

    // ── After 1.e4 e5 2.Nf3 (black) ──────────────────────────────────────
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b8c6", 10),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "g8f6",  5),

    // ── After 1.e4 c5 Sicilian: 2.Nf3 ────────────────────────────────────
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "g1f3", 10),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "b1c3",  5),

    // ── After 1.d4 d5 2.c4 Queen's Gambit ────────────────────────────────
    ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", "c2c4",  8),
    ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", "g1f3",  6),
];

/// Build a sorted `Vec<PolyEntry>` from the hardcoded opening lines.
/// Lines whose FEN cannot be parsed or whose move is not legal are silently
/// skipped (they represent a bug in the table, not a runtime error).
pub fn build_internal_entries() -> Vec<PolyEntry> {
    let mut entries: Vec<PolyEntry> = Vec::new();

    for &(fen, uci, weight) in INTERNAL_LINES {
        let board = match Board::from_fen(fen) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let legal = board.legal_moves();
        let mv = match parse_move(uci, &legal) {
            Some(m) => m,
            None => continue,
        };
        entries.push(PolyEntry {
            key:    poly_hash(&board),
            mv:     encode_move(mv),
            weight,
            learn:  0,
        });
    }

    entries.sort_unstable_by_key(|e| e.key);
    entries
}
