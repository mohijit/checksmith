//! Legal move filtering and perft.
//!
//! We turn pseudo-legal moves into legal ones the simple, bulletproof way: make
//! each move, ask whether our own king is now attacked, and unmake. This is
//! slightly slower than tracking pins and checks directly, but it is *correct by
//! construction* — including the awkward cases (en-passant discovered check,
//! pinned pieces, castling) that trip up cleverer schemes. We can optimize later
//! once perft proves the foundation is right.
//!
//! [`perft`](Board::perft) counts the leaf nodes of the move tree to a given
//! depth. Comparing those counts against published reference values is the
//! definitive test that move generation is exactly correct.

use crate::board::Board;
use crate::movegen::attacks::is_square_attacked;
use crate::movegen::moves::{Move, MoveList};

impl Board {
    /// Is the side to move currently in check?
    pub fn is_in_check(&self) -> bool {
        match self.king_square(self.side_to_move) {
            Some(ksq) => is_square_attacked(self, ksq, self.side_to_move.opposite()),
            None => false,
        }
    }

    /// The side to move is checkmated: in check with no legal reply.
    pub fn is_checkmate(&self) -> bool {
        self.is_in_check() && self.legal_moves().is_empty()
    }

    /// The side to move is stalemated: not in check but has no legal move.
    pub fn is_stalemate(&self) -> bool {
        !self.is_in_check() && self.legal_moves().is_empty()
    }

    /// All fully legal moves for the side to move.
    pub fn legal_moves(&self) -> MoveList {
        let mut legal = MoveList::new();
        let us = self.side_to_move;
        let pseudo = self.pseudo_legal_moves();

        // Work on a clone so this method can take `&self`.
        let mut board = self.clone();
        for &mv in pseudo.iter() {
            let undo = board.make_move(mv);
            // After the move it's the opponent's turn; check *our* king's safety.
            let king = board.king_square(us).expect("side to move has no king");
            if !is_square_attacked(&board, king, us.opposite()) {
                legal.push(mv);
            }
            board.unmake_move(mv, undo);
        }
        legal
    }

    /// Count leaf nodes of the legal-move tree at the given `depth`.
    ///
    /// `perft(0) == 1`. The `depth == 1` shortcut just counts the legal moves
    /// instead of recursing into each.
    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = self.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut nodes = 0;
        for &mv in moves.iter() {
            let undo = self.make_move(mv);
            nodes += self.perft(depth - 1);
            self.unmake_move(mv, undo);
        }
        nodes
    }

    /// Perft, but broken down by first move — the standard debugging aid. The
    /// per-move subtree counts can be compared against a reference engine to
    /// pinpoint exactly which move's generation is wrong.
    pub fn perft_divide(&mut self, depth: u32) -> (u64, Vec<(Move, u64)>) {
        let mut total = 0;
        let mut breakdown = Vec::new();
        for &mv in self.legal_moves().iter() {
            let undo = self.make_move(mv);
            let count = if depth <= 1 { 1 } else { self.perft(depth - 1) };
            self.unmake_move(mv, undo);
            total += count;
            breakdown.push((mv, count));
        }
        (total, breakdown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;

    /// Reference perft values from the Chess Programming Wiki. We keep the
    /// automated depths modest so `cargo test` (a debug build) stays fast; the
    /// deeper checks live in `#[ignore]`d tests you can run in release.
    fn check_perft(fen: &str, expected: &[(u32, u64)]) {
        let mut board = Board::from_fen(fen).unwrap();
        for &(depth, nodes) in expected {
            assert_eq!(
                board.perft(depth),
                nodes,
                "perft({}) wrong for {}",
                depth,
                fen
            );
        }
    }

    #[test]
    fn perft_startpos() {
        check_perft(STARTING_FEN, &[(1, 20), (2, 400), (3, 8902), (4, 197281)]);
    }

    #[test]
    fn perft_kiwipete() {
        // Dense middlegame: castling both sides, en passant, pins.
        check_perft(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[(1, 48), (2, 2039), (3, 97862)],
        );
    }

    #[test]
    fn perft_position3() {
        // Sparse position rich in en-passant and promotion edge cases.
        check_perft(
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            &[(1, 14), (2, 191), (3, 2812), (4, 43238)],
        );
    }

    #[test]
    fn perft_position4() {
        check_perft(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[(1, 6), (2, 264), (3, 9467)],
        );
    }

    #[test]
    fn perft_position5() {
        check_perft(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[(1, 44), (2, 1486), (3, 62379)],
        );
    }

    // Deeper, slower checks. Run in release with:
    //   cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn perft_deep() {
        check_perft(STARTING_FEN, &[(5, 4865609), (6, 119060324)]);
        check_perft(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[(4, 4085603), (5, 193690690)],
        );
        check_perft(
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            &[(5, 674624), (6, 11030083)],
        );
    }
}
