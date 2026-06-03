//! Move encoding.
//!
//! A [`Move`] is packed into a single 16-bit integer — small enough that a
//! whole move list fits in cache, which matters once search is generating
//! millions of them per second:
//!
//! ```text
//! bits  0..6   from square (0..63)
//! bits  6..12  to square   (0..63)
//! bits 12..16  flag        (move type, see MoveFlag)
//! ```
//!
//! The 4-bit flag encoding (from the Chess Programming Wiki) is chosen so that
//! two bits carry meaning on their own:
//!
//! * bit 2 (value 4) set  => the move is a **capture**
//! * bit 3 (value 8) set  => the move is a **promotion**
//!
//! That lets `is_capture`/`is_promotion` be a single masked test.

use crate::board::piece::PieceType;
use crate::board::square::Square;
use std::fmt;
use std::ops::Deref;

/// The type of a move, encoded in the move's top 4 bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum MoveFlag {
    Quiet = 0,
    /// A pawn's initial two-square advance (sets the en-passant square).
    DoublePush = 1,
    KingCastle = 2,
    QueenCastle = 3,
    Capture = 4,
    EnPassant = 5,
    // 6, 7 unused (keep the capture bit meaningful).
    PromoKnight = 8,
    PromoBishop = 9,
    PromoRook = 10,
    PromoQueen = 11,
    PromoKnightCapture = 12,
    PromoBishopCapture = 13,
    PromoRookCapture = 14,
    PromoQueenCapture = 15,
}

impl MoveFlag {
    fn from_u16(value: u16) -> MoveFlag {
        match value {
            0 => MoveFlag::Quiet,
            1 => MoveFlag::DoublePush,
            2 => MoveFlag::KingCastle,
            3 => MoveFlag::QueenCastle,
            4 => MoveFlag::Capture,
            5 => MoveFlag::EnPassant,
            8 => MoveFlag::PromoKnight,
            9 => MoveFlag::PromoBishop,
            10 => MoveFlag::PromoRook,
            11 => MoveFlag::PromoQueen,
            12 => MoveFlag::PromoKnightCapture,
            13 => MoveFlag::PromoBishopCapture,
            14 => MoveFlag::PromoRookCapture,
            15 => MoveFlag::PromoQueenCapture,
            other => unreachable!("invalid move flag bits: {}", other),
        }
    }
}

/// A packed chess move.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(u16);

impl Move {
    /// A sentinel "no move", used to initialize move-list storage.
    pub const NULL: Move = Move(0);

    /// The raw 16-bit encoding, for compact storage (e.g. the transposition table).
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Reconstruct a move from its [`bits`](Move::bits). `0` is [`Move::NULL`].
    #[inline]
    pub const fn from_bits(bits: u16) -> Move {
        Move(bits)
    }

    /// Build a move from a source square, target square, and flag.
    #[inline]
    pub fn new(from: Square, to: Square, flag: MoveFlag) -> Move {
        Move((from.index() as u16) | ((to.index() as u16) << 6) | ((flag as u16) << 12))
    }

    /// Source square.
    #[inline]
    pub fn from(self) -> Square {
        Square::new((self.0 & 0x3F) as u8)
    }

    /// Target square.
    #[inline]
    pub fn to(self) -> Square {
        Square::new(((self.0 >> 6) & 0x3F) as u8)
    }

    /// The move type.
    #[inline]
    pub fn flag(self) -> MoveFlag {
        MoveFlag::from_u16(self.0 >> 12)
    }

    /// True if the move removes an enemy piece (includes en passant and
    /// promotion-captures).
    #[inline]
    pub fn is_capture(self) -> bool {
        (self.0 >> 12) & 0b0100 != 0
    }

    /// True if the move promotes a pawn.
    #[inline]
    pub fn is_promotion(self) -> bool {
        (self.0 >> 12) & 0b1000 != 0
    }

    #[inline]
    pub fn is_en_passant(self) -> bool {
        self.flag() == MoveFlag::EnPassant
    }

    #[inline]
    pub fn is_castle(self) -> bool {
        matches!(self.flag(), MoveFlag::KingCastle | MoveFlag::QueenCastle)
    }

    /// The piece a pawn promotes to, if this is a promotion.
    pub fn promotion_piece(self) -> Option<PieceType> {
        match self.flag() {
            MoveFlag::PromoKnight | MoveFlag::PromoKnightCapture => Some(PieceType::Knight),
            MoveFlag::PromoBishop | MoveFlag::PromoBishopCapture => Some(PieceType::Bishop),
            MoveFlag::PromoRook | MoveFlag::PromoRookCapture => Some(PieceType::Rook),
            MoveFlag::PromoQueen | MoveFlag::PromoQueenCapture => Some(PieceType::Queen),
            _ => None,
        }
    }

    /// Render in UCI long-algebraic notation, e.g. `"e2e4"` or `"e7e8q"`.
    pub fn to_uci(self) -> String {
        let mut s = format!("{}{}", self.from(), self.to());
        if let Some(pt) = self.promotion_piece() {
            s.push(match pt {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => unreachable!(),
            });
        }
        s
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:?})", self.to_uci(), self.flag())
    }
}

/// A fixed-capacity, stack-allocated list of moves.
///
/// No position has more than 218 legal moves, so a 256-slot array never spills
/// and we avoid a heap allocation on every node of the search.
#[derive(Clone)]
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> MoveList {
        MoveList {
            moves: [Move::NULL; 256],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < 256, "move list overflow");
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for MoveList {
    fn default() -> Self {
        MoveList::new()
    }
}

impl MoveList {
    /// The populated moves as a mutable slice, so the search can reorder them
    /// in place (selection-sort by score) without allocating.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        &mut self.moves[..self.len]
    }
}

/// Dereferencing a `MoveList` yields the populated slice, so `for mv in &list`,
/// `list.iter()`, indexing, and `.len()` all just work.
impl Deref for MoveList {
    type Target = [Move];

    #[inline]
    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack() {
        let mv = Move::new(Square::E2, Square::E4, MoveFlag::DoublePush);
        assert_eq!(mv.from(), Square::E2);
        assert_eq!(mv.to(), Square::E4);
        assert_eq!(mv.flag(), MoveFlag::DoublePush);
        assert!(!mv.is_capture());
        assert!(!mv.is_promotion());
        assert_eq!(mv.to_uci(), "e2e4");
    }

    #[test]
    fn capture_and_promotion_bits() {
        let cap = Move::new(Square::E4, Square::D5, MoveFlag::Capture);
        assert!(cap.is_capture());
        assert!(!cap.is_promotion());

        let promo = Move::new(Square::E7, Square::E8, MoveFlag::PromoQueen);
        assert!(promo.is_promotion());
        assert!(!promo.is_capture());
        assert_eq!(promo.promotion_piece(), Some(PieceType::Queen));
        assert_eq!(promo.to_uci(), "e7e8q");

        let promo_cap = Move::new(Square::E7, Square::F8, MoveFlag::PromoRookCapture);
        assert!(promo_cap.is_promotion());
        assert!(promo_cap.is_capture());
        assert_eq!(promo_cap.promotion_piece(), Some(PieceType::Rook));
    }

    #[test]
    fn list_basics() {
        let mut list = MoveList::new();
        assert!(list.is_empty());
        list.push(Move::new(Square::E2, Square::E4, MoveFlag::DoublePush));
        list.push(Move::new(Square::G1, Square::F3, MoveFlag::Quiet));
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].to_uci(), "e2e4");
        let ucis: Vec<String> = list.iter().map(|m| m.to_uci()).collect();
        assert_eq!(ucis, vec!["e2e4", "g1f3"]);
    }
}
