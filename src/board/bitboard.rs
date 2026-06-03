//! Bitboards: a 64-bit set of squares.
//!
//! A [`Bitboard`] is a `u64` where bit `i` is set iff square `i` is "in" the
//! set. Because there are exactly 64 squares on a chessboard, a single machine
//! word represents *any* subset of squares — all white pawns, all squares a
//! rook attacks, all empty squares, and so on.
//!
//! Why this matters for a chess engine:
//!
//! * **Set operations are single instructions.** Union is `|`, intersection is
//!   `&`, difference is `& !`, symmetric difference is `^`.
//! * **Move generation becomes arithmetic.** Shifting all pawns "north" one
//!   rank is `bb << 8`. Knight and king attacks are precomputed bitboards.
//! * **Counting and scanning are hardware-accelerated.** `count_ones`
//!   (POPCNT) counts pieces; `trailing_zeros` (BSF/TZCNT) finds the
//!   lowest-indexed square — the basis of iterating over set squares.

use super::square::Square;
use std::fmt;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr,
};

/// A set of squares, one bit per square (LERF mapping, see [`super::square`]).
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// The empty set.
    pub const EMPTY: Bitboard = Bitboard(0);
    /// Every square.
    pub const FULL: Bitboard = Bitboard(!0u64);

    // File masks (all 8 squares of a file).
    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);

    // Rank masks (all 8 squares of a rank). RANK_1 is White's back rank.
    pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
    pub const RANK_2: Bitboard = Bitboard(0x0000_0000_0000_FF00);
    pub const RANK_3: Bitboard = Bitboard(0x0000_0000_00FF_0000);
    pub const RANK_4: Bitboard = Bitboard(0x0000_0000_FF00_0000);
    pub const RANK_5: Bitboard = Bitboard(0x0000_00FF_0000_0000);
    pub const RANK_6: Bitboard = Bitboard(0x0000_FF00_0000_0000);
    pub const RANK_7: Bitboard = Bitboard(0x00FF_0000_0000_0000);
    pub const RANK_8: Bitboard = Bitboard(0xFF00_0000_0000_0000);

    /// A bitboard containing exactly the given square.
    #[inline]
    pub const fn from_square(sq: Square) -> Self {
        Bitboard(1u64 << sq.index())
    }

    /// True if no squares are set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True if at least one square is set.
    #[inline]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// True if `sq` is a member of the set.
    #[inline]
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 >> sq.index()) & 1 != 0
    }

    /// Number of squares in the set (population count).
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Add `sq` to the set.
    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.index();
    }

    /// Remove `sq` from the set.
    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.index());
    }

    /// The lowest-indexed set square, if any (least-significant bit).
    #[inline]
    pub const fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square::new(self.0.trailing_zeros() as u8))
        }
    }

    /// Remove and return the lowest-indexed set square.
    ///
    /// This is the workhorse for iterating over the squares in a bitboard.
    #[inline]
    pub fn pop_lsb(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let sq = Square::new(self.0.trailing_zeros() as u8);
        // Clearing the lowest set bit: x & (x - 1).
        self.0 &= self.0 - 1;
        Some(sq)
    }
}

/// Iterating a bitboard yields its squares from lowest index to highest.
///
/// `Bitboard` is `Copy`, so `for sq in bb { .. }` iterates over a *copy* and
/// leaves the original untouched.
impl Iterator for Bitboard {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        self.pop_lsb()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.count() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Bitboard {}

// --- Bitwise operators -----------------------------------------------------
// These let us write set algebra naturally: `white & pawns`, `occ | bb`, etc.

impl BitAnd for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}
impl BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}
impl BitOr for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}
impl BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}
impl BitXor for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}
impl BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}
impl Not for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}
impl Shl<u32> for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn shl(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}
impl Shr<u32> for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn shr(self, rhs: u32) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard({:#018x})", self.0)
    }
}

/// Pretty-prints the board with rank 8 on top and `1`/`.` for set/unset.
impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                write!(f, "{} ", if self.contains(sq) { '1' } else { '.' })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_clear_contains_count() {
        let mut bb = Bitboard::EMPTY;
        assert!(bb.is_empty());
        bb.set(Square::E4);
        assert!(bb.contains(Square::E4));
        assert!(!bb.contains(Square::E5));
        assert_eq!(bb.count(), 1);
        bb.clear(Square::E4);
        assert!(bb.is_empty());
    }

    #[test]
    fn iterates_in_index_order() {
        // bits 0, 1, 3 set -> a1, b1, d1
        let bb = Bitboard(0b1011);
        let squares: Vec<usize> = bb.map(|s| s.index()).collect();
        assert_eq!(squares, vec![0, 1, 3]);
    }

    #[test]
    fn set_algebra() {
        let a = Bitboard(0b1100);
        let b = Bitboard(0b0110);
        assert_eq!((a & b).0, 0b0100);
        assert_eq!((a | b).0, 0b1110);
        assert_eq!((a ^ b).0, 0b1010);
        assert_eq!(Bitboard::FULL.count(), 64);
    }
}
