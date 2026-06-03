//! Pseudo-legal move generation.
//!
//! "Pseudo-legal" means we generate every move that obeys how the pieces move,
//! *without* yet checking whether it leaves our own king in check. Filtering
//! those out happens in [`crate::movegen::legal`]. Splitting it this way keeps
//! each stage simple and easy to test.
//!
//! Pawn moves use whole-bitboard shifts: shifting all pawns "forward" one rank
//! computes every single-push destination at once. The file masks prevent a
//! capture from "wrapping" off the a/h edge of the board.

use crate::board::bitboard::Bitboard;
use crate::board::piece::{Color, PieceType};
use crate::board::square::Square;
use crate::board::{Board, CastlingRights};
use crate::movegen::attacks::{
    bishop_attacks, is_square_attacked, king_attacks, knight_attacks, pawn_attacks, rook_attacks,
};
use crate::movegen::moves::{Move, MoveFlag, MoveList};

impl Board {
    /// All pseudo-legal moves for the side to move.
    pub fn pseudo_legal_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        let us = self.side_to_move;
        generate_pawn_moves(self, us, &mut list);
        generate_knight_moves(self, us, &mut list);
        generate_king_moves(self, us, &mut list);
        generate_slider_moves(self, us, &mut list);
        generate_castling(self, us, &mut list);
        list
    }
}

/// Emit the four promotion moves (knight, bishop, rook, queen) for a pawn
/// reaching the back rank.
fn push_promotions(list: &mut MoveList, from: Square, to: Square, capture: bool) {
    let flags = if capture {
        [
            MoveFlag::PromoKnightCapture,
            MoveFlag::PromoBishopCapture,
            MoveFlag::PromoRookCapture,
            MoveFlag::PromoQueenCapture,
        ]
    } else {
        [
            MoveFlag::PromoKnight,
            MoveFlag::PromoBishop,
            MoveFlag::PromoRook,
            MoveFlag::PromoQueen,
        ]
    };
    for flag in flags {
        list.push(Move::new(from, to, flag));
    }
}

fn generate_pawn_moves(board: &Board, us: Color, list: &mut MoveList) {
    let them = us.opposite();
    let pawns = board.pieces_colored(us, PieceType::Pawn);
    let empty = !board.occupancy();
    let enemy = board.color(them);

    if us == Color::White {
        let promo_rank = Bitboard::RANK_8;

        // Single pushes (north).
        let single = (pawns << 8) & empty;
        for to in single & !promo_rank {
            list.push(Move::new(back(to, 8), to, MoveFlag::Quiet));
        }
        for to in single & promo_rank {
            push_promotions(list, back(to, 8), to, false);
        }

        // Double pushes: a single push that landed on rank 3, advanced again.
        let double = ((single & Bitboard::RANK_3) << 8) & empty;
        for to in double {
            list.push(Move::new(back(to, 16), to, MoveFlag::DoublePush));
        }

        // Captures north-east (+9) and north-west (+7); masks block edge wrap.
        let cap_ne = (pawns << 9) & !Bitboard::FILE_A & enemy;
        emit_pawn_captures(list, cap_ne, promo_rank, 9, true);
        let cap_nw = (pawns << 7) & !Bitboard::FILE_H & enemy;
        emit_pawn_captures(list, cap_nw, promo_rank, 7, true);
    } else {
        let promo_rank = Bitboard::RANK_1;

        let single = (pawns >> 8) & empty;
        for to in single & !promo_rank {
            list.push(Move::new(fwd(to, 8), to, MoveFlag::Quiet));
        }
        for to in single & promo_rank {
            push_promotions(list, fwd(to, 8), to, false);
        }

        let double = ((single & Bitboard::RANK_6) >> 8) & empty;
        for to in double {
            list.push(Move::new(fwd(to, 16), to, MoveFlag::DoublePush));
        }

        // Captures south-east (-7) and south-west (-9).
        let cap_se = (pawns >> 7) & !Bitboard::FILE_A & enemy;
        emit_pawn_captures(list, cap_se, promo_rank, 7, false);
        let cap_sw = (pawns >> 9) & !Bitboard::FILE_H & enemy;
        emit_pawn_captures(list, cap_sw, promo_rank, 9, false);
    }

    // En passant: which of our pawns can capture onto the ep square?
    if let Some(ep) = board.ep_square {
        let candidates = pawn_attacks(them, ep) & pawns;
        for from in candidates {
            list.push(Move::new(from, ep, MoveFlag::EnPassant));
        }
    }
}

/// Helper: destination minus `delta` (a "from" square one or two ranks back),
/// for White moving north.
#[inline]
fn back(to: Square, delta: u8) -> Square {
    Square::new(to.index() as u8 - delta)
}

/// Helper: destination plus `delta`, for Black moving south.
#[inline]
fn fwd(to: Square, delta: u8) -> Square {
    Square::new(to.index() as u8 + delta)
}

/// Emit captures (and promotion-captures) from a destination bitboard. `delta`
/// and `white` tell us how to recover the origin square.
fn emit_pawn_captures(
    list: &mut MoveList,
    targets: Bitboard,
    promo_rank: Bitboard,
    delta: u8,
    white: bool,
) {
    for to in targets & !promo_rank {
        let from = if white { back(to, delta) } else { fwd(to, delta) };
        list.push(Move::new(from, to, MoveFlag::Capture));
    }
    for to in targets & promo_rank {
        let from = if white { back(to, delta) } else { fwd(to, delta) };
        push_promotions(list, from, to, true);
    }
}

fn generate_knight_moves(board: &Board, us: Color, list: &mut MoveList) {
    let own = board.color(us);
    let enemy = board.color(us.opposite());
    for from in board.pieces_colored(us, PieceType::Knight) {
        for to in knight_attacks(from) & !own {
            let flag = if enemy.contains(to) {
                MoveFlag::Capture
            } else {
                MoveFlag::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }
}

fn generate_king_moves(board: &Board, us: Color, list: &mut MoveList) {
    let own = board.color(us);
    let enemy = board.color(us.opposite());
    if let Some(from) = board.king_square(us) {
        for to in king_attacks(from) & !own {
            let flag = if enemy.contains(to) {
                MoveFlag::Capture
            } else {
                MoveFlag::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }
}

fn generate_slider_moves(board: &Board, us: Color, list: &mut MoveList) {
    let own = board.color(us);
    let enemy = board.color(us.opposite());
    let occ = board.occupancy();

    let bishops = board.pieces_colored(us, PieceType::Bishop);
    let rooks = board.pieces_colored(us, PieceType::Rook);
    let queens = board.pieces_colored(us, PieceType::Queen);

    // Bishops and queens share the diagonal rays.
    for from in bishops | queens {
        for to in bishop_attacks(from, occ) & !own {
            let flag = if enemy.contains(to) {
                MoveFlag::Capture
            } else {
                MoveFlag::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }
    // Rooks and queens share the orthogonal rays.
    for from in rooks | queens {
        for to in rook_attacks(from, occ) & !own {
            let flag = if enemy.contains(to) {
                MoveFlag::Capture
            } else {
                MoveFlag::Quiet
            };
            list.push(Move::new(from, to, flag));
        }
    }
}

fn generate_castling(board: &Board, us: Color, list: &mut MoveList) {
    // Can't castle out of check. (This also guards the king's origin square,
    // so below we only need to verify the squares it passes through and lands on.)
    if board.is_in_check() {
        return;
    }
    let them = us.opposite();
    let occ = board.occupancy();
    let empty = |sq: Square| !occ.contains(sq);
    let safe = |sq: Square| !is_square_attacked(board, sq, them);

    if us == Color::White {
        if board.castling.has(CastlingRights::WHITE_KING)
            && empty(Square::F1)
            && empty(Square::G1)
            && safe(Square::F1)
            && safe(Square::G1)
        {
            list.push(Move::new(Square::E1, Square::G1, MoveFlag::KingCastle));
        }
        if board.castling.has(CastlingRights::WHITE_QUEEN)
            && empty(Square::D1)
            && empty(Square::C1)
            && empty(Square::B1)
            && safe(Square::D1)
            && safe(Square::C1)
        {
            list.push(Move::new(Square::E1, Square::C1, MoveFlag::QueenCastle));
        }
    } else {
        if board.castling.has(CastlingRights::BLACK_KING)
            && empty(Square::F8)
            && empty(Square::G8)
            && safe(Square::F8)
            && safe(Square::G8)
        {
            list.push(Move::new(Square::E8, Square::G8, MoveFlag::KingCastle));
        }
        if board.castling.has(CastlingRights::BLACK_QUEEN)
            && empty(Square::D8)
            && empty(Square::C8)
            && empty(Square::B8)
            && safe(Square::D8)
            && safe(Square::C8)
        {
            list.push(Move::new(Square::E8, Square::C8, MoveFlag::QueenCastle));
        }
    }
}
