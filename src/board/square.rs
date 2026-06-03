//! Board squares.
//!
//! We use the **Little-Endian Rank-File (LERF)** mapping, the same convention
//! Stockfish and most modern engines use:
//!
//! ```text
//! index = rank * 8 + file
//!
//!   a1 = 0,  b1 = 1,  ..., h1 = 7
//!   a2 = 8,  ...            h2 = 15
//!   ...
//!   a8 = 56, ...            h8 = 63
//! ```
//!
//! This mapping is chosen because bit `i` of a `u64` bitboard then corresponds
//! to square `i`, and shifting a bitboard left by 8 moves every piece "up" one
//! rank — which makes pawn pushes and sliding-piece math clean.

use std::fmt;

/// A board square in `0..=63`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Square(u8);

impl Square {
    /// Create a square from a raw index. The index must be `< 64`.
    #[inline]
    pub const fn new(index: u8) -> Self {
        debug_assert!(index < 64);
        Square(index)
    }

    /// Create a square, returning `None` if the index is out of range.
    #[inline]
    pub const fn try_new(index: u8) -> Option<Self> {
        if index < 64 {
            Some(Square(index))
        } else {
            None
        }
    }

    /// Build a square from a `file` (0=a..7=h) and `rank` (0=rank 1..7=rank 8).
    #[inline]
    pub const fn from_file_rank(file: u8, rank: u8) -> Self {
        Square(rank * 8 + file)
    }

    /// The raw `0..=63` index, as a `usize` for convenient array indexing.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// File of the square: 0 = a-file .. 7 = h-file.
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// Rank of the square: 0 = rank 1 .. 7 = rank 8.
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    /// Parse algebraic coordinates like `"e4"` into a [`Square`].
    pub fn from_algebraic(s: &str) -> Option<Square> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0];
        let rank = bytes[1];
        if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
            return None;
        }
        Some(Square::from_file_rank(file - b'a', rank - b'1'))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{}{}", file, rank)
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

// Named constants for every square. Verbose, but they make tests and future
// move-generation / castling code read like the chessboard.
#[rustfmt::skip]
impl Square {
    pub const A1: Square = Square(0);  pub const B1: Square = Square(1);  pub const C1: Square = Square(2);  pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);  pub const F1: Square = Square(5);  pub const G1: Square = Square(6);  pub const H1: Square = Square(7);
    pub const A2: Square = Square(8);  pub const B2: Square = Square(9);  pub const C2: Square = Square(10); pub const D2: Square = Square(11);
    pub const E2: Square = Square(12); pub const F2: Square = Square(13); pub const G2: Square = Square(14); pub const H2: Square = Square(15);
    pub const A3: Square = Square(16); pub const B3: Square = Square(17); pub const C3: Square = Square(18); pub const D3: Square = Square(19);
    pub const E3: Square = Square(20); pub const F3: Square = Square(21); pub const G3: Square = Square(22); pub const H3: Square = Square(23);
    pub const A4: Square = Square(24); pub const B4: Square = Square(25); pub const C4: Square = Square(26); pub const D4: Square = Square(27);
    pub const E4: Square = Square(28); pub const F4: Square = Square(29); pub const G4: Square = Square(30); pub const H4: Square = Square(31);
    pub const A5: Square = Square(32); pub const B5: Square = Square(33); pub const C5: Square = Square(34); pub const D5: Square = Square(35);
    pub const E5: Square = Square(36); pub const F5: Square = Square(37); pub const G5: Square = Square(38); pub const H5: Square = Square(39);
    pub const A6: Square = Square(40); pub const B6: Square = Square(41); pub const C6: Square = Square(42); pub const D6: Square = Square(43);
    pub const E6: Square = Square(44); pub const F6: Square = Square(45); pub const G6: Square = Square(46); pub const H6: Square = Square(47);
    pub const A7: Square = Square(48); pub const B7: Square = Square(49); pub const C7: Square = Square(50); pub const D7: Square = Square(51);
    pub const E7: Square = Square(52); pub const F7: Square = Square(53); pub const G7: Square = Square(54); pub const H7: Square = Square(55);
    pub const A8: Square = Square(56); pub const B8: Square = Square(57); pub const C8: Square = Square(58); pub const D8: Square = Square(59);
    pub const E8: Square = Square(60); pub const F8: Square = Square(61); pub const G8: Square = Square(62); pub const H8: Square = Square(63);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algebraic_round_trip() {
        for i in 0..64u8 {
            let sq = Square::new(i);
            let s = sq.to_string();
            assert_eq!(Square::from_algebraic(&s), Some(sq));
        }
    }

    #[test]
    fn corners_and_indices() {
        assert_eq!(Square::from_algebraic("a1"), Some(Square::A1));
        assert_eq!(Square::from_algebraic("h8"), Some(Square::H8));
        assert_eq!(Square::A1.index(), 0);
        assert_eq!(Square::H8.index(), 63);
        assert_eq!(Square::E4.file(), 4);
        assert_eq!(Square::E4.rank(), 3);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Square::from_algebraic(""), None);
        assert_eq!(Square::from_algebraic("e9"), None);
        assert_eq!(Square::from_algebraic("i1"), None);
        assert_eq!(Square::from_algebraic("e4x"), None);
    }
}
