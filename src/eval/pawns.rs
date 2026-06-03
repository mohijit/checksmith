//! Pawn-structure evaluation, plus the file/rank bitboard helpers shared across
//! the evaluation.
//!
//! Pawns define the character of a position.  A handful of timeless rules
//! capture most of the value:
//!
//! * **Doubled** pawns (two on one file) get in each other's way — a penalty.
//! * **Isolated** pawns (no friendly pawn on an adjacent file) can't be
//!   defended by another pawn — a penalty.
//! * **Passed** pawns (no enemy pawn ahead on the same or adjacent files) are
//!   future queens — a bonus that grows sharply as they advance, and is worth
//!   even more in the endgame.
//! * **Connected passed** pawns (two passed pawns on adjacent files) support
//!   each other and are collectively much harder to stop.
//! * **Candidate passed** pawns (blocked on adjacent files only, not directly
//!   in front) have a clear breakthrough path if their supporters outnumber the
//!   opposing stoppers.

use crate::board::bitboard::Bitboard;
use crate::board::{Board, Color, PieceType};
use crate::eval::score::Score;

// ─── Penalties ────────────────────────────────────────────────────────────────

const ISOLATED: Score = Score::new(-12, -18);
const DOUBLED:  Score = Score::new(-10, -22);

// ─── Passed-pawn bonuses (indexed by relative rank 0-7) ─────────────────────
//
// Relative rank 0 = own back rank (never reached for a legal pawn position).
// Relative rank 6 = one step from promotion.
// Relative rank 7 = 0 (promotion is handled elsewhere).

const PASSED_MG: [i32; 8] = [  0,  5, 10,  20,  35,  60, 100, 0];
const PASSED_EG: [i32; 8] = [  0, 12, 22,  40,  68, 110, 160, 0];

// ─── Connected passed-pawn *extra* bonus ──────────────────────────────────────
//
// When two passed pawns are on adjacent files, each gets this bonus on top of
// its individual PASSED bonus.  The combined force of a pawn duo is far
// greater than two isolated passers: one can advance while the other covers it.

const CONNECTED_PASSED_MG: [i32; 8] = [0,  2,  5, 10, 18, 30,  50, 0];
const CONNECTED_PASSED_EG: [i32; 8] = [0,  5, 10, 20, 35, 60, 100, 0];

// ─── Candidate passed-pawn bonus ─────────────────────────────────────────────
//
// A pawn that has no enemy pawn directly in front of it but is blocked by
// enemy pawns on adjacent files.  With the right pawn structure it can
// "break through" by exchanging the adjacent stoppers.  The bonus is roughly
// half the passed-pawn value at each rank.

const CANDIDATE_MG: [i32; 8] = [0, 3,  6, 12, 20, 35, 60, 0];
const CANDIDATE_EG: [i32; 8] = [0, 6, 12, 22, 40, 68, 100, 0];

// ─── Helpers (pub so other eval sub-modules can use them) ────────────────────

/// All squares on a file (0 = a-file .. 7 = h-file).
#[inline]
pub(crate) fn file_bb(file: u8) -> Bitboard {
    Bitboard(0x0101_0101_0101_0101u64 << file)
}

/// All squares on a rank (0 = rank 1 .. 7 = rank 8).
#[inline]
pub(crate) fn rank_bb(rank: u8) -> Bitboard {
    Bitboard(0xFFu64 << (8 * rank))
}

/// The files immediately left and right of `file`.
#[inline]
pub(crate) fn adjacent_files_bb(file: u8) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    if file > 0 { bb |= file_bb(file - 1); }
    if file < 7 { bb |= file_bb(file + 1); }
    bb
}

/// Every square strictly ahead of `rank` from `color`'s point of view.
///
/// White moves towards higher rank indices; Black moves towards lower ones.
#[inline]
pub(crate) fn forward_ranks(color: Color, rank: u8) -> Bitboard {
    match color {
        Color::White if rank < 7 => Bitboard(!0u64 << (8 * (rank as u32 + 1))),
        Color::Black if rank > 0 => Bitboard(!0u64 >> (8 * (8 - rank as u32))),
        _ => Bitboard::EMPTY,
    }
}

/// Pawn relative rank: distance advanced from the side's own back rank (0..7).
#[inline]
pub(crate) fn relative_rank(color: Color, rank: u8) -> usize {
    match color {
        Color::White => rank as usize,
        Color::Black => (7 - rank) as usize,
    }
}

// ─── Main entry point ────────────────────────────────────────────────────────

/// Pawn-structure score from White's perspective (White minus Black).
pub fn evaluate_pawns(board: &Board) -> Score {
    pawns_for(board, Color::White) - pawns_for(board, Color::Black)
}

fn pawns_for(board: &Board, color: Color) -> Score {
    let mut score = Score::ZERO;
    let own   = board.pieces_colored(color,            PieceType::Pawn);
    let enemy = board.pieces_colored(color.opposite(), PieceType::Pawn);

    // ── First pass: per-pawn features ─────────────────────────────────────
    let mut passed_mask = Bitboard::EMPTY; // accumulate for the connected check

    for sq in own {
        let file  = sq.file();
        let rank  = sq.rank();
        let r     = relative_rank(color, rank);

        // Isolated: no friendly pawn on either adjacent file.
        if (own & adjacent_files_bb(file)).is_empty() {
            score += ISOLATED;
        }

        // Passed: no enemy pawn ahead on this or adjacent files.
        let forward_zone = (file_bb(file) | adjacent_files_bb(file))
                           & forward_ranks(color, rank);
        let is_passed = (enemy & forward_zone).is_empty();

        if is_passed {
            score += Score::new(PASSED_MG[r], PASSED_EG[r]);
            passed_mask.set(sq);
        } else {
            // Candidate passed pawn: no enemy pawn directly ahead on the same
            // file, but enemy pawns on adjacent files prevent it from being
            // fully passed.  Given the right lever (pawn exchange on an
            // adjacent file) it can become a passer.
            //
            // Condition: same file is clear ahead (otherwise the pawn is just
            // blocked, which is worse).  We already know !is_passed, so the
            // adjacent-file blockers are present by definition.
            let same_file_ahead = file_bb(file) & forward_ranks(color, rank);
            if (enemy & same_file_ahead).is_empty() {
                score += Score::new(CANDIDATE_MG[r], CANDIDATE_EG[r]);
            }
        }
    }

    // ── Second pass: doubled pawns ────────────────────────────────────────
    for file in 0..8u8 {
        let count = (own & file_bb(file)).count() as i32;
        if count > 1 {
            score += DOUBLED * (count - 1);
        }
    }

    // ── Third pass: connected passed pawns ───────────────────────────────
    //
    // Each passed pawn that has a neighbour passed pawn on an adjacent file
    // earns a rank-indexed bonus (on top of its individual PASSED bonus).
    for sq in passed_mask {
        if (passed_mask & adjacent_files_bb(sq.file())).any() {
            let r = relative_rank(color, sq.rank());
            score += Score::new(CONNECTED_PASSED_MG[r], CONNECTED_PASSED_EG[r]);
        }
    }

    score
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::score::taper;

    #[test]
    fn isolated_pawns_are_penalized() {
        // b2+c2 defend each other; b2+d2 are both isolated.
        let connected = Board::from_fen("4k3/8/8/8/8/8/1PP5/4K3 w - - 0 1").unwrap();
        let isolated  = Board::from_fen("4k3/8/8/8/8/8/1P1P4/4K3 w - - 0 1").unwrap();
        assert!(evaluate_pawns(&connected).mg > evaluate_pawns(&isolated).mg);
    }

    #[test]
    fn passed_pawn_is_rewarded() {
        // White pawn on e6 with no black pawns anywhere: a clear passer.
        let board = Board::from_fen("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1").unwrap();
        let s = evaluate_pawns(&board);
        assert!(taper(s, 12) > 0, "passed pawn should score positive: {:?}", s);
    }

    #[test]
    fn doubled_pawns_are_penalized() {
        let doubled = Board::from_fen("4k3/8/8/4P3/4P3/8/8/4K3 w - - 0 1").unwrap();
        let healthy = Board::from_fen("4k3/8/8/3PP3/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate_pawns(&doubled).mg < evaluate_pawns(&healthy).mg);
    }

    #[test]
    fn connected_passed_pawns_bonus() {
        // Two adjacent passed pawns score higher than one passed pawn.
        let two_passers = Board::from_fen("4k3/8/8/3PP3/8/8/8/4K3 w - - 0 1").unwrap();
        let one_passer  = Board::from_fen("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1").unwrap();
        let two_eg = taper(evaluate_pawns(&two_passers), 4);
        let one_eg = taper(evaluate_pawns(&one_passer),  4);
        // Two connected passers should score more than twice the single passer
        // (the connection bonus on top of the individual bonuses).
        assert!(two_eg > one_eg * 2,
            "connected passers should exceed 2× single passer, got {} vs {}×2",
            two_eg, one_eg);
    }

    #[test]
    fn candidate_passed_pawn_small_bonus() {
        // White pawn on e5, Black pawn on d6 (on an adjacent file, not directly
        // in front): the e-file is clear, so White's pawn is a candidate.
        // Compare with White e5 directly blocked by Black e6.
        //
        // Both pawns are isolated in both positions, so the isolated penalty
        // cancels; the difference is purely the candidate bonus.
        let candidate = Board::from_fen("4k3/8/3p4/4P3/8/8/8/4K3 w - - 0 1").unwrap();
        let blocked   = Board::from_fen("4k3/8/4p3/4P3/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate_pawns(&candidate).mg > evaluate_pawns(&blocked).mg,
            "candidate pawn should score higher than directly-blocked pawn");
    }
}
