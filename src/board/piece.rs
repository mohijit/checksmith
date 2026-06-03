//! Colors, piece types, and pieces.
//!
//! We deliberately separate three concepts:
//!
//! * [`Color`]   — White or Black (the side).
//! * [`PieceType`] — Pawn..King, independent of color.
//! * [`Piece`]   — a `(Color, PieceType)` pair, i.e. a concrete piece.
//!
//! The board stores pieces as `[Bitboard; 6]` (one per [`PieceType`]) plus
//! `[Bitboard; 2]` (one per [`Color`]). White knights are therefore
//! `pieces[Knight] & colors[White]`. This is more compact than 12 separate
//! bitboards and keeps "all knights" / "all white" queries to one operation.

/// The side to move / piece owner.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// Both colors, useful for iteration.
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    /// 0 for White, 1 for Black — used to index `[_; 2]` arrays.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The opposing color.
    #[inline]
    pub const fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// A piece kind, independent of color.
///
/// Discriminants `0..=5` are used to index `[_; 6]` arrays, so the order here
/// is load-bearing — keep it stable.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    /// All piece types in discriminant order.
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    /// 0=Pawn .. 5=King — used to index `[_; 6]` arrays.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Inverse of [`PieceType::index`].
    #[inline]
    pub const fn from_index(i: usize) -> Option<PieceType> {
        match i {
            0 => Some(PieceType::Pawn),
            1 => Some(PieceType::Knight),
            2 => Some(PieceType::Bishop),
            3 => Some(PieceType::Rook),
            4 => Some(PieceType::Queen),
            5 => Some(PieceType::King),
            _ => None,
        }
    }
}

/// A concrete piece: a color and a piece type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    #[inline]
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { color, piece_type }
    }

    /// FEN/PGN letter: uppercase for White, lowercase for Black.
    pub fn to_char(self) -> char {
        let c = match self.piece_type {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        };
        if self.color == Color::White {
            c.to_ascii_uppercase()
        } else {
            c
        }
    }

    /// Parse a FEN/PGN piece letter, e.g. `'N'` -> white knight.
    pub fn from_char(ch: char) -> Option<Piece> {
        let color = if ch.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece_type = match ch.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        };
        Some(Piece::new(color, piece_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_round_trip() {
        for &color in &Color::ALL {
            for &pt in &PieceType::ALL {
                let p = Piece::new(color, pt);
                assert_eq!(Piece::from_char(p.to_char()), Some(p));
            }
        }
    }

    #[test]
    fn case_encodes_color() {
        assert_eq!(Piece::new(Color::White, PieceType::King).to_char(), 'K');
        assert_eq!(Piece::new(Color::Black, PieceType::King).to_char(), 'k');
        assert_eq!(Piece::from_char('x'), None);
    }

    #[test]
    fn indices_are_stable() {
        assert_eq!(PieceType::Pawn.index(), 0);
        assert_eq!(PieceType::King.index(), 5);
        assert_eq!(Color::White.index(), 0);
        assert_eq!(Color::Black.index(), 1);
    }
}
