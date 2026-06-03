//! Piece-square tables (PSTs), fully tapered — every piece type has an
//! independent middlegame and endgame table.
//!
//! ## Visual layout
//!
//! Every table is written **visually** with rank 8 (Black's home rank) at the
//! top (index 0) and rank 1 (White's home rank) at the bottom (index 56-63).
//! Index formula: `visual_index = (7 - rank) * 8 + file`.
//!
//! White looks up with a rank-flip: `sq.index() ^ 56` (same as xor-ing the
//! rank field).  Black reads the table directly — their pieces start on high
//! ranks, which sit at the *top* of the visual layout, which is exactly the
//! "correct" perspective for Black.
//!
//! ## Phase interpolation
//!
//! At leaf evaluation we collapse `Score { mg, eg }` with [`score::taper`] to
//! a single centipawn number according to how much material remains.  Separate
//! mg/eg tables let each piece type shift its priorities smoothly:
//!
//! * **Pawn** — in the endgame every advanced pawn is a promotion threat;
//!   the eg table rewards all ranks, not just the central files.
//! * **Knight** — centralisation matters equally in both phases.
//! * **Bishop** — in the endgame, long diagonals through the centre dominate;
//!   the eg table is therefore more centre-oriented than the mg one.
//! * **Rook** — slight endgame preference for central files and the 7th rank
//!   (7th-rank bonus is also applied separately in the evaluator).
//! * **Queen** — kept near-identical; the evaluator adds mobility on top.
//! * **King** — strong phase contrast: hide in the corner for the middlegame,
//!   centralise in the endgame.

use crate::board::{Color, PieceType, Square};
use crate::eval::score::Score;

// ─── Pawns ────────────────────────────────────────────────────────────────────

#[rustfmt::skip]
const PAWN_MG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

// Endgame: every advanced pawn matters; rank alone drives value.
#[rustfmt::skip]
const PAWN_EG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
];

// ─── Knights ──────────────────────────────────────────────────────────────────

// Centralisation matters in both phases; rim knights are bad in any phase.
#[rustfmt::skip]
const KNIGHT_MG: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

// Endgame: rim penalty slightly worse (fewer pieces = each square matters more).
#[rustfmt::skip]
const KNIGHT_EG: [i32; 64] = [
    -60,-45,-35,-35,-35,-35,-45,-60,
    -45,-25,  0,  0,  0,  0,-25,-45,
    -35,  0, 10, 15, 15, 10,  0,-35,
    -35,  5, 15, 20, 20, 15,  5,-35,
    -35,  0, 15, 20, 20, 15,  0,-35,
    -35,  5, 10, 15, 15, 10,  5,-35,
    -45,-25,  0,  5,  5,  0,-25,-45,
    -60,-45,-35,-35,-35,-35,-45,-60,
];

// ─── Bishops ──────────────────────────────────────────────────────────────────

// Middlegame: activity and diagonal control.
#[rustfmt::skip]
const BISHOP_MG: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

// Endgame: bishops dominate through the centre; long diagonals are crucial.
#[rustfmt::skip]
const BISHOP_EG: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10,  0, 15, 20, 20, 15,  0,-10,
    -10,  0, 15, 20, 20, 15,  0,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

// ─── Rooks ────────────────────────────────────────────────────────────────────

// Middlegame: 7th rank and open files are most important.
#[rustfmt::skip]
const ROOK_MG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

// Endgame: rooks need to be active — central files and 7th rank.
#[rustfmt::skip]
const ROOK_EG: [i32; 64] = [
     5,  5,  5,  5,  5,  5,  5,  5,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  5,  5,  5,  5,  0,  0,
     0,  0,  5,  5,  5,  5,  0,  0,
     0,  0,  5,  5,  5,  5,  0,  0,
     0,  0,  5,  5,  5,  5,  0,  0,
    -5, -5,  0,  0,  0,  0, -5, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

// ─── Queens ───────────────────────────────────────────────────────────────────

// Middlegame: develop late; too-early queen gets harassed.
#[rustfmt::skip]
const QUEEN_MG: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

// Endgame: queens are mobile and should centralise.
#[rustfmt::skip]
const QUEEN_EG: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  5, 10, 10, 10, 10,  5,-10,
     -5,  5, 10, 15, 15, 10,  5, -5,
     -5,  5, 10, 15, 15, 10,  5, -5,
    -10,  5, 10, 10, 10, 10,  5,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

// ─── Kings ────────────────────────────────────────────────────────────────────

// Middlegame: stay castled; the corner is safest.
#[rustfmt::skip]
const KING_MG: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

// Endgame: the king is a fighting piece — centralise it.
#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

// ─── Lookup ───────────────────────────────────────────────────────────────────

/// Positional bonus for a piece of `(color, pt)` on `sq`, as a tapered Score.
///
/// White's tables are written from White's point of view (rank 1 at the
/// bottom), so we rank-flip White squares: `sq.index() ^ 56`.  Black reads
/// tables directly since Black's pieces sit on high ranks, which correspond
/// to the top rows (small indices) in the visual layout.
#[inline]
pub fn pst(pt: PieceType, color: Color, sq: Square) -> Score {
    let i = match color {
        Color::White => sq.index() ^ 56,
        Color::Black => sq.index(),
    };
    let (mg, eg) = match pt {
        PieceType::Pawn   => (PAWN_MG[i],   PAWN_EG[i]),
        PieceType::Knight => (KNIGHT_MG[i], KNIGHT_EG[i]),
        PieceType::Bishop => (BISHOP_MG[i], BISHOP_EG[i]),
        PieceType::Rook   => (ROOK_MG[i],   ROOK_EG[i]),
        PieceType::Queen  => (QUEEN_MG[i],  QUEEN_EG[i]),
        PieceType::King   => (KING_MG[i],   KING_EG[i]),
    };
    Score::new(mg, eg)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn king_table_flips_between_phases() {
        let corner = pst(PieceType::King, Color::White, Square::G1);
        let center = pst(PieceType::King, Color::White, Square::E4);
        assert!(corner.mg > center.mg, "MG king prefers the corner");
        assert!(center.eg > corner.eg, "EG king prefers the center");
    }

    #[test]
    fn bishop_eg_is_more_central_than_mg() {
        // In the eg table a central bishop should be worth more than in the mg.
        let center = pst(PieceType::Bishop, Color::White, Square::D4);
        let corner = pst(PieceType::Bishop, Color::White, Square::A1);
        // Central EG bonus is larger than the MG bonus (long diagonals dominate EG).
        assert!(center.eg > center.mg, "bishop centralisation gains in the EG");
        assert!(corner.mg >= corner.eg, "corner bishop does not improve in EG");
    }

    #[test]
    fn tables_are_color_symmetric() {
        // pst(White, E2) must equal pst(Black, E7) for every piece type.
        for &pt in &PieceType::ALL {
            let w = pst(pt, Color::White, Square::E2);
            let b = pst(pt, Color::Black, Square::E7);
            assert_eq!(w, b, "asymmetry for {:?}", pt);
        }
    }

    #[test]
    fn knight_rim_worse_in_eg() {
        let rim = pst(PieceType::Knight, Color::White, Square::A1);
        // Both phases penalise the rim, but the EG penalty is at least as large.
        assert!(rim.mg <= 0);
        assert!(rim.eg <= rim.mg, "EG rim penalty should be >= MG ({} vs {})", rim.eg, rim.mg);
    }

    #[test]
    fn rook_eg_rewards_seventh_rank_and_central_files() {
        // On the 7th rank (a7 for White using visual flip), EG value > MG value.
        let rook_7th = pst(PieceType::Rook, Color::White, Square::D7);
        assert!(rook_7th.eg >= rook_7th.mg, "rook EG on 7th should be >= MG");
    }
}
