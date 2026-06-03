//! Applying and reverting moves.
//!
//! Search explores a tree by *making* a move, recursing, then *unmaking* it to
//! return to the parent position. Doing this in place (rather than copying the
//! whole board) is the standard performance technique, but it requires us to
//! remember just enough to perfectly reverse each move.
//!
//! [`make_move`](Board::make_move) returns an [`Undo`] capturing the
//! irreversible state (the captured piece, previous castling/en-passant rights,
//! the halfmove clock, and the previous hash). [`unmake_move`](Board::unmake_move)
//! consumes it to restore the position exactly.
//!
//! The Zobrist hash is maintained incrementally: `put_piece`/`remove_piece`
//! already toggle piece keys, and here we additionally toggle the side,
//! castling, and en-passant keys. On unmake we simply restore the saved hash.

use crate::board::piece::{Color, Piece, PieceType};
use crate::board::square::Square;
use crate::board::zobrist::zobrist;
use crate::board::{Board, CastlingRights};
use crate::movegen::moves::{Move, MoveFlag};

/// Information needed to reverse a [`Move`].
#[derive(Clone, Copy)]
pub struct Undo {
    captured: Option<Piece>,
    prev_castling: CastlingRights,
    prev_ep: Option<Square>,
    prev_halfmove: u16,
    prev_hash: u64,
}

/// Information needed to reverse a null move ([`Board::make_null_move`]).
///
/// A null move passes the turn to the opponent without moving any piece. It is
/// used exclusively by null move pruning in the search — never as an actual
/// game move. Only the state that changes (EP square, halfmove clock, hash)
/// needs to be saved.
#[derive(Clone, Copy)]
pub struct NullUndo {
    prev_ep: Option<Square>,
    prev_halfmove: u16,
    prev_hash: u64,
}

/// The castling rights that are revoked when a given square is vacated or
/// captured upon (king or rook home squares). 0 for every other square.
fn castle_mask(sq: Square) -> u8 {
    match sq.index() {
        0 => CastlingRights::WHITE_QUEEN,                          // a1 rook
        4 => CastlingRights::WHITE_KING | CastlingRights::WHITE_QUEEN, // e1 king
        7 => CastlingRights::WHITE_KING,                          // h1 rook
        56 => CastlingRights::BLACK_QUEEN,                        // a8 rook
        60 => CastlingRights::BLACK_KING | CastlingRights::BLACK_QUEEN, // e8 king
        63 => CastlingRights::BLACK_KING,                        // h8 rook
        _ => 0,
    }
}

/// The rook's `(from, to)` squares for a castling king move landing on `king_to`.
fn castling_rook_squares(king_to: Square) -> (Square, Square) {
    let to = king_to.index() as u8;
    if king_to.file() == 6 {
        // King-side: rook hops from the h-file (king_to+1) to the f-file (king_to-1).
        (Square::new(to + 1), Square::new(to - 1))
    } else {
        // Queen-side: rook hops from the a-file (king_to-2) to the d-file (king_to+1).
        (Square::new(to - 2), Square::new(to + 1))
    }
}

impl Board {
    /// Apply `mv`, returning the [`Undo`] needed to revert it.
    pub fn make_move(&mut self, mv: Move) -> Undo {
        let z = zobrist();
        let us = self.side_to_move;
        let from = mv.from();
        let to = mv.to();

        let prev_castling = self.castling;
        let prev_ep = self.ep_square;
        let prev_halfmove = self.halfmove_clock;
        let prev_hash = self.hash;

        let moving = self
            .piece_at(from)
            .expect("make_move: no piece on source square");

        // Clear any existing en-passant key; we set a fresh one only on a double push.
        if let Some(ep) = self.ep_square {
            self.hash ^= z.ep_file(ep.file());
        }
        self.ep_square = None;

        // 50-move rule: reset on pawn moves and captures, otherwise increment.
        let mut reset_halfmove = moving.piece_type == PieceType::Pawn;
        let mut captured = None;

        match mv.flag() {
            MoveFlag::EnPassant => {
                // The captured pawn sits beside `to`, on the moving side's rank.
                let cap_sq = if us == Color::White {
                    Square::new(to.index() as u8 - 8)
                } else {
                    Square::new(to.index() as u8 + 8)
                };
                captured = self.remove_piece(cap_sq);
                self.remove_piece(from);
                self.put_piece(to, moving);
                reset_halfmove = true;
            }
            MoveFlag::KingCastle | MoveFlag::QueenCastle => {
                self.remove_piece(from);
                self.put_piece(to, moving);
                let (rook_from, rook_to) = castling_rook_squares(to);
                let rook = self
                    .remove_piece(rook_from)
                    .expect("make_move: no rook to castle with");
                self.put_piece(rook_to, rook);
            }
            _ => {
                if mv.is_capture() {
                    captured = self.remove_piece(to);
                    reset_halfmove = true;
                }
                self.remove_piece(from);
                let placed = match mv.promotion_piece() {
                    Some(pt) => Piece::new(us, pt),
                    None => moving,
                };
                self.put_piece(to, placed);
            }
        }

        // Set a new en-passant target square behind a double-pushed pawn.
        if mv.flag() == MoveFlag::DoublePush {
            // The skipped square is the midpoint between `from` and `to`.
            let ep = Square::new(((from.index() + to.index()) / 2) as u8);
            self.ep_square = Some(ep);
            self.hash ^= z.ep_file(ep.file());
        }

        // Update castling rights (toggle the hash key around the change).
        self.hash ^= z.castling(self.castling.0);
        self.castling.0 &= !(castle_mask(from) | castle_mask(to));
        self.hash ^= z.castling(self.castling.0);

        // Flip side to move.
        self.side_to_move = us.opposite();
        self.hash ^= z.side();

        if us == Color::Black {
            self.fullmove_number += 1;
        }
        self.halfmove_clock = if reset_halfmove { 0 } else { prev_halfmove + 1 };

        Undo {
            captured,
            prev_castling,
            prev_ep,
            prev_halfmove,
            prev_hash,
        }
    }

    /// Pass the turn to the opponent without moving any piece.
    ///
    /// Used exclusively by null move pruning. The en-passant square is cleared
    /// (there is no pawn move to set a new one), the side to move flips, and the
    /// Zobrist hash is updated to match. The hash is saved in [`NullUndo`] so
    /// [`unmake_null_move`](Board::unmake_null_move) can restore it wholesale.
    pub fn make_null_move(&mut self) -> NullUndo {
        let z = zobrist();
        let undo = NullUndo {
            prev_ep: self.ep_square,
            prev_halfmove: self.halfmove_clock,
            prev_hash: self.hash,
        };

        // Clear any EP square (no pawn double-pushed on this "move").
        if let Some(ep) = self.ep_square {
            self.hash ^= z.ep_file(ep.file());
        }
        self.ep_square = None;

        // Flip side to move.
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= z.side();

        // No capture or pawn move, so increment halfmove clock.
        self.halfmove_clock += 1;

        undo
    }

    /// Revert a null move made by [`make_null_move`](Board::make_null_move).
    pub fn unmake_null_move(&mut self, undo: NullUndo) {
        // Flip side back.
        self.side_to_move = self.side_to_move.opposite();
        // Restore saved state (hash covers both the side and EP key).
        self.ep_square = undo.prev_ep;
        self.halfmove_clock = undo.prev_halfmove;
        self.hash = undo.prev_hash;
    }

    /// Revert the most recent [`make_move`](Board::make_move).
    pub fn unmake_move(&mut self, mv: Move, undo: Undo) {
        let from = mv.from();
        let to = mv.to();
        // The side that originally made the move is the side *not* on move now.
        let us = self.side_to_move.opposite();

        match mv.flag() {
            MoveFlag::EnPassant => {
                let pawn = self
                    .remove_piece(to)
                    .expect("unmake: missing pawn on destination");
                self.put_piece(from, pawn);
                let cap_sq = if us == Color::White {
                    Square::new(to.index() as u8 - 8)
                } else {
                    Square::new(to.index() as u8 + 8)
                };
                self.put_piece(cap_sq, undo.captured.expect("unmake: missing ep capture"));
            }
            MoveFlag::KingCastle | MoveFlag::QueenCastle => {
                let king = self.remove_piece(to).expect("unmake: missing king");
                self.put_piece(from, king);
                let (rook_from, rook_to) = castling_rook_squares(to);
                let rook = self.remove_piece(rook_to).expect("unmake: missing rook");
                self.put_piece(rook_from, rook);
            }
            _ => {
                let moved = self.remove_piece(to).expect("unmake: missing piece");
                // A promoted piece reverts to a pawn of the moving side.
                let original = if mv.is_promotion() {
                    Piece::new(us, PieceType::Pawn)
                } else {
                    moved
                };
                self.put_piece(from, original);
                if let Some(captured) = undo.captured {
                    self.put_piece(to, captured);
                }
            }
        }

        // Restore the irreversible state. (Hash is restored wholesale, so the
        // intermediate piece toggles above don't need to be exact.)
        self.side_to_move = us;
        self.castling = undo.prev_castling;
        self.ep_square = undo.prev_ep;
        self.halfmove_clock = undo.prev_halfmove;
        if us == Color::Black {
            self.fullmove_number -= 1;
        }
        self.hash = undo.prev_hash;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;

    /// make then unmake must restore the position byte-for-byte (via FEN+hash).
    fn assert_make_unmake_round_trip(fen: &str, mv: Move) {
        let mut board = Board::from_fen(fen).unwrap();
        let before_fen = board.to_fen();
        let before_hash = board.hash;

        let undo = board.make_move(mv);
        // The hash maintained incrementally must match a full recompute.
        assert_eq!(board.hash, board.compute_hash(), "hash drift after make");

        board.unmake_move(mv, undo);
        assert_eq!(board.to_fen(), before_fen, "position not restored");
        assert_eq!(board.hash, before_hash, "hash not restored");
    }

    #[test]
    fn quiet_and_double_push() {
        assert_make_unmake_round_trip(
            STARTING_FEN,
            Move::new(Square::E2, Square::E4, MoveFlag::DoublePush),
        );
        assert_make_unmake_round_trip(
            STARTING_FEN,
            Move::new(Square::G1, Square::F3, MoveFlag::Quiet),
        );
    }

    #[test]
    fn capture_round_trip() {
        // White pawn on e4 captures black pawn on d5.
        let fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2";
        assert_make_unmake_round_trip(fen, Move::new(Square::E4, Square::D5, MoveFlag::Capture));
    }

    #[test]
    fn en_passant_round_trip() {
        // White pawn e5 takes black pawn d5 en passant onto d6.
        let fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        assert_make_unmake_round_trip(fen, Move::new(Square::E5, Square::D6, MoveFlag::EnPassant));
    }

    #[test]
    fn castling_round_trip() {
        let fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
        assert_make_unmake_round_trip(fen, Move::new(Square::E1, Square::G1, MoveFlag::KingCastle));
        assert_make_unmake_round_trip(fen, Move::new(Square::E1, Square::C1, MoveFlag::QueenCastle));
    }

    #[test]
    fn promotion_round_trip() {
        let fen = "8/P7/8/8/8/8/8/k6K w - - 0 1";
        assert_make_unmake_round_trip(fen, Move::new(Square::A7, Square::A8, MoveFlag::PromoQueen));
    }
}
