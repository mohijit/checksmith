//! Material values and game-phase weights.
//!
//! Values are paired ([`Score`]): pawns and rooks are worth a little more in the
//! endgame, where passed pawns decide games and rooks dominate open boards. The
//! king has no material value (it's always present).

use crate::board::PieceType;
use crate::eval::score::Score;

/// Centipawn value of a piece type, as a middlegame/endgame pair.
#[inline]
pub fn material(pt: PieceType) -> Score {
    match pt {
        PieceType::Pawn => Score::new(100, 120),
        PieceType::Knight => Score::new(320, 320),
        PieceType::Bishop => Score::new(330, 330),
        PieceType::Rook => Score::new(500, 550),
        PieceType::Queen => Score::new(900, 950),
        PieceType::King => Score::ZERO,
    }
}

/// How much a piece contributes to the "game phase" used for tapering.
///
/// Summing this over all pieces gives a number from 0 (bare kings) up to
/// [`TOTAL_PHASE`](crate::eval::score::TOTAL_PHASE) (full starting material).
#[inline]
pub fn phase_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Knight | PieceType::Bishop => 1,
        PieceType::Rook => 2,
        PieceType::Queen => 4,
        _ => 0,
    }
}
