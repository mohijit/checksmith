//! Zobrist hashing.
//!
//! A Zobrist hash is a 64-bit fingerprint of a position. We assign a random
//! 64-bit key to every (color, piece, square) combination, plus keys for the
//! side to move, castling rights, and en-passant file. The hash of a position
//! is the XOR of the keys for every feature present.
//!
//! Two properties make this ideal for a chess engine:
//!
//! * **Incremental updates.** Because XOR is its own inverse, moving a piece
//!   from `a` to `b` is `hash ^= key(a); hash ^= key(b)` — O(1), no rescan.
//!   (We do a full recompute on FEN load; incremental updates arrive with
//!   make/unmake in Milestone 2.)
//! * **Cheap, near-collision-free equality.** Two positions reached by
//!   different move orders get the same hash, which is what lets the
//!   transposition table (Milestone 6) recognize them.
//!
//! Keys are generated deterministically from a fixed seed via SplitMix64, so a
//! given position always hashes to the same value across runs — important for
//! reproducible tests and opening books.

use super::piece::{Color, PieceType};
use super::square::Square;
use std::sync::OnceLock;

/// The full table of random keys.
pub struct Zobrist {
    /// `[color][piece_type][square]`
    pieces: [[[u64; 64]; 6]; 2],
    /// XORed in when it is Black's turn to move.
    side: u64,
    /// Indexed by the 4-bit castling-rights mask (`0..16`).
    castling: [u64; 16],
    /// Indexed by the file (0..8) of the en-passant target square.
    ep_file: [u64; 8],
}

/// SplitMix64 — a tiny, high-quality PRNG used only to fill the key tables.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Zobrist {
    fn new() -> Zobrist {
        let mut rng = SplitMix64 {
            state: 0x1234_5678_9ABC_DEF0,
        };

        let mut pieces = [[[0u64; 64]; 6]; 2];
        for color in pieces.iter_mut() {
            for piece in color.iter_mut() {
                for square in piece.iter_mut() {
                    *square = rng.next();
                }
            }
        }

        let side = rng.next();

        let mut castling = [0u64; 16];
        for key in castling.iter_mut() {
            *key = rng.next();
        }

        let mut ep_file = [0u64; 8];
        for key in ep_file.iter_mut() {
            *key = rng.next();
        }

        Zobrist {
            pieces,
            side,
            castling,
            ep_file,
        }
    }

    /// Key for a specific piece standing on a specific square.
    #[inline]
    pub fn piece(&self, color: Color, piece_type: PieceType, sq: Square) -> u64 {
        self.pieces[color.index()][piece_type.index()][sq.index()]
    }

    /// Key XORed in when it is Black to move.
    #[inline]
    pub fn side(&self) -> u64 {
        self.side
    }

    /// Key for a castling-rights bitmask (`0..=15`).
    #[inline]
    pub fn castling(&self, rights: u8) -> u64 {
        self.castling[(rights & 0xF) as usize]
    }

    /// Key for the file of an en-passant target square.
    #[inline]
    pub fn ep_file(&self, file: u8) -> u64 {
        self.ep_file[(file & 7) as usize]
    }
}

/// The process-wide key table, initialized once on first use.
static ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

/// Access the shared [`Zobrist`] key table.
pub fn zobrist() -> &'static Zobrist {
    ZOBRIST.get_or_init(Zobrist::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_deterministic_and_distinct() {
        let z = zobrist();
        // Same input -> same key.
        assert_eq!(
            z.piece(Color::White, PieceType::Pawn, Square::E2),
            z.piece(Color::White, PieceType::Pawn, Square::E2)
        );
        // Different square -> different key (overwhelmingly likely).
        assert_ne!(
            z.piece(Color::White, PieceType::Pawn, Square::E2),
            z.piece(Color::White, PieceType::Pawn, Square::E4)
        );
        // No key should be zero (the seed avoids it for these slots).
        assert_ne!(z.side(), 0);
    }
}
