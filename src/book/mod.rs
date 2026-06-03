//! Opening book support.
//!
//! [`OpeningBook`] can load entries from an external Polyglot `.bin` file,
//! fall back to the hardcoded internal book, or disable book play entirely.
//! Entry points:
//!
//! * [`OpeningBook::new_internal`]  — use only the built-in lines.
//! * [`OpeningBook::load`]          — try a Polyglot file; fall back to internal.
//! * [`OpeningBook::probe`]         — return all entries for a position hash.
//! * [`OpeningBook::pick_move`]     — weighted-random selection from those entries.

pub mod internal;
pub mod polyglot;
pub mod polyglot_random;

use polyglot::{decode_move, load_polyglot, poly_hash, probe_entries, PolyEntry};

use crate::board::Board;
use crate::movegen::Move;
use rand::Rng;
use std::path::Path;


// ── OpeningBook ────────────────────────────────────────────────────────────────

/// An opening book backed by either an external Polyglot file or the built-in
/// hardcoded lines (or both merged together, with the Polyglot file winning on
/// overlapping positions).
pub struct OpeningBook {
    /// All entries, sorted by `key` so binary-search probing is O(log n).
    entries: Vec<PolyEntry>,
}

impl OpeningBook {
    /// Build a book from the internal hardcoded lines only.
    pub fn new_internal() -> OpeningBook {
        OpeningBook {
            entries: internal::build_internal_entries(),
        }
    }

    /// Try to load a Polyglot `.bin` file.  On failure, print a warning to
    /// stderr and fall back to the internal book silently.
    pub fn load<P: AsRef<Path>>(path: P) -> OpeningBook {
        match load_polyglot(&path) {
            Ok(mut entries) => {
                entries.sort_unstable_by_key(|e| e.key);
                OpeningBook { entries }
            }
            Err(e) => {
                eprintln!(
                    "info string book file {:?} not found: {} — using internal book",
                    path.as_ref(),
                    e
                );
                OpeningBook::new_internal()
            }
        }
    }

    /// Return all book entries for `board`'s position.
    ///
    /// Returns an empty slice when the position is unknown to the book.
    pub fn probe<'a>(&'a self, board: &Board) -> &'a [PolyEntry] {
        let hash = poly_hash(board);
        probe_entries(&self.entries, hash)
    }

    /// Pick one book move for `board` using weighted-random selection.
    ///
    /// Only moves that are legal in `board` are considered.  Returns `None`
    /// when the position is not in the book or no entry is legal.
    ///
    /// `rng` is caller-supplied so callers can seed it deterministically in
    /// tests.
    pub fn pick_move<R: Rng>(&self, board: &Board, rng: &mut R) -> Option<Move> {
        let hits = self.probe(board);
        if hits.is_empty() {
            return None;
        }

        let legal = board.legal_moves();

        // Collect only the (move, weight) pairs that are legal.
        let candidates: Vec<(Move, u16)> = hits
            .iter()
            .filter_map(|e| decode_move(e.mv, &legal).map(|mv| (mv, e.weight)))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let total: u32 = candidates.iter().map(|&(_, w)| w as u32).sum();

        // Guard against zero total (all weights were 0) — return first entry.
        if total == 0 {
            return Some(candidates[0].0);
        }

        let mut r = rng.gen_range(0..total);
        for (mv, w) in &candidates {
            if r < *w as u32 {
                return Some(*mv);
            }
            r -= *w as u32;
        }

        // Fallback — should not be reached if total > 0.
        Some(candidates[0].0)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn book() -> OpeningBook {
        OpeningBook::new_internal()
    }

    #[test]
    fn internal_book_has_entries_for_startpos() {
        let b = book();
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let hits = b.probe(&board);
        assert!(!hits.is_empty(), "internal book should have startpos entries");
    }

    #[test]
    fn pick_move_returns_legal_move() {
        let b = book();
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let mv = b.pick_move(&board, &mut rng);
        assert!(mv.is_some(), "should pick a move from startpos");
        let legal = board.legal_moves();
        assert!(legal.iter().any(|&m| Some(m) == mv), "picked move must be legal");
    }

    #[test]
    fn pick_move_returns_none_for_unknown_position() {
        let b = book();
        // Bare-kings endgame — definitely not in any book.
        let board = Board::from_fen("8/8/4k3/8/8/4K3/8/8 w - - 0 1").unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        assert!(b.pick_move(&board, &mut rng).is_none());
    }

    #[test]
    fn pick_move_is_consistent_with_seed() {
        let b = book();
        let board = Board::from_fen(STARTING_FEN).unwrap();
        // Same seed must produce the same move every time.
        let mut rng1 = StdRng::seed_from_u64(99);
        let mut rng2 = StdRng::seed_from_u64(99);
        let m1 = b.pick_move(&board, &mut rng1);
        let m2 = b.pick_move(&board, &mut rng2);
        assert_eq!(m1, m2);
    }
}
