//! Attack generation.
//!
//! For the "leaper" pieces — knights, kings, pawns — the squares they attack
//! depend only on where they stand, so we precompute a lookup table once and
//! index into it. For the "slider" pieces — bishops, rooks, queens — the
//! attacked set depends on what blocks the rays, so we compute it on the fly by
//! walking each ray until it hits an occupied square.
//!
//! The ray-walking sliders are simple and obviously correct, which is exactly
//! what we want for Milestone 2 (we validate everything with perft). Later they
//! can be swapped for *magic bitboards* — a perfect-hash lookup — purely as a
//! speed optimization, without changing the `bishop_attacks(sq, occ)` /
//! `rook_attacks(sq, occ)` interface that the rest of the engine depends on.

use crate::board::bitboard::Bitboard;
use crate::board::piece::{Color, PieceType};
use crate::board::square::Square;
use crate::board::Board;
use std::sync::OnceLock;

/// Precomputed attack sets for the leaper pieces.
struct LeaperTables {
    knight: [Bitboard; 64],
    king: [Bitboard; 64],
    /// `[color][square]` — squares a pawn of `color` on `square` attacks.
    pawn: [[Bitboard; 64]; 2],
}

// Relative (file, rank) offsets for knight and king moves.
const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING_OFFSETS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Build a leaper's attack set by trying each offset and discarding any that
/// fall off the board.
fn leaper_from(sq: Square, offsets: &[(i8, i8)]) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    let (f0, r0) = (sq.file() as i8, sq.rank() as i8);
    for &(df, dr) in offsets {
        let (f, r) = (f0 + df, r0 + dr);
        if (0..8).contains(&f) && (0..8).contains(&r) {
            bb.set(Square::from_file_rank(f as u8, r as u8));
        }
    }
    bb
}

/// Squares a pawn of `color` on `sq` attacks (the two forward diagonals).
fn pawn_from(color: Color, sq: Square) -> Bitboard {
    let dr: i8 = if color == Color::White { 1 } else { -1 };
    let mut bb = Bitboard::EMPTY;
    let (f0, r0) = (sq.file() as i8, sq.rank() as i8);
    for df in [-1i8, 1] {
        let (f, r) = (f0 + df, r0 + dr);
        if (0..8).contains(&f) && (0..8).contains(&r) {
            bb.set(Square::from_file_rank(f as u8, r as u8));
        }
    }
    bb
}

impl LeaperTables {
    fn new() -> LeaperTables {
        let mut knight = [Bitboard::EMPTY; 64];
        let mut king = [Bitboard::EMPTY; 64];
        let mut pawn = [[Bitboard::EMPTY; 64]; 2];
        for i in 0..64u8 {
            let sq = Square::new(i);
            knight[i as usize] = leaper_from(sq, &KNIGHT_OFFSETS);
            king[i as usize] = leaper_from(sq, &KING_OFFSETS);
            pawn[Color::White.index()][i as usize] = pawn_from(Color::White, sq);
            pawn[Color::Black.index()][i as usize] = pawn_from(Color::Black, sq);
        }
        LeaperTables { knight, king, pawn }
    }
}

static LEAPERS: OnceLock<LeaperTables> = OnceLock::new();

fn leapers() -> &'static LeaperTables {
    LEAPERS.get_or_init(LeaperTables::new)
}

/// Squares a knight on `sq` attacks.
#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    leapers().knight[sq.index()]
}

/// Squares a king on `sq` attacks (ignoring castling).
#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    leapers().king[sq.index()]
}

/// Squares a pawn of `color` on `sq` attacks.
#[inline]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    leapers().pawn[color.index()][sq.index()]
}

/// Walk each direction in `dirs` from `sq`, stopping at (and including) the
/// first blocker. Used for sliding-piece attacks.
fn slider_attacks(sq: Square, occ: Bitboard, dirs: &[(i8, i8)]) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    let (f0, r0) = (sq.file() as i8, sq.rank() as i8);
    for &(df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let target = Square::from_file_rank(f as u8, r as u8);
            bb.set(target);
            if occ.contains(target) {
                break; // ray stops at the first occupied square (a possible capture)
            }
            f += df;
            r += dr;
        }
    }
    bb
}

/// Squares a bishop on `sq` attacks given board occupancy `occ`.
#[inline]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    slider_attacks(sq, occ, &BISHOP_DIRS)
}

/// Squares a rook on `sq` attacks given board occupancy `occ`.
#[inline]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    slider_attacks(sq, occ, &ROOK_DIRS)
}

/// Squares a queen on `sq` attacks given board occupancy `occ`.
#[inline]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

/// Is `sq` attacked by any piece of color `by`?
///
/// This is the core of check detection and castling legality. We use a "reverse
/// attack" trick: instead of asking "what does each enemy piece attack?", we
/// place a *super-piece* on `sq` and ask "if I were a knight here, would I hit
/// an enemy knight?", and so on for each piece type. If yes, `sq` is attacked.
pub fn is_square_attacked(board: &Board, sq: Square, by: Color) -> bool {
    // Pawns: a `by` pawn attacks `sq` iff one stands where an *enemy* pawn on
    // `sq` would attack — i.e. on `pawn_attacks(by.opposite(), sq)`.
    if (pawn_attacks(by.opposite(), sq) & board.pieces_colored(by, PieceType::Pawn)).any() {
        return true;
    }
    if (knight_attacks(sq) & board.pieces_colored(by, PieceType::Knight)).any() {
        return true;
    }
    if (king_attacks(sq) & board.pieces_colored(by, PieceType::King)).any() {
        return true;
    }

    let occ = board.occupancy();
    let bishops_queens =
        board.pieces_colored(by, PieceType::Bishop) | board.pieces_colored(by, PieceType::Queen);
    if (bishop_attacks(sq, occ) & bishops_queens).any() {
        return true;
    }
    let rooks_queens =
        board.pieces_colored(by, PieceType::Rook) | board.pieces_colored(by, PieceType::Queen);
    if (rook_attacks(sq, occ) & rooks_queens).any() {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_in_corner_and_center() {
        // a1 knight reaches exactly b3 and c2.
        assert_eq!(knight_attacks(Square::A1).count(), 2);
        // A central knight reaches 8 squares.
        assert_eq!(knight_attacks(Square::E4).count(), 8);
    }

    #[test]
    fn king_edge_counts() {
        assert_eq!(king_attacks(Square::A1).count(), 3);
        assert_eq!(king_attacks(Square::E4).count(), 8);
    }

    #[test]
    fn pawn_attack_directions() {
        // White pawn on e4 attacks d5 and f5.
        let wa = pawn_attacks(Color::White, Square::E4);
        assert!(wa.contains(Square::D5) && wa.contains(Square::F5) && wa.count() == 2);
        // Black pawn on e4 attacks d3 and f3.
        let ba = pawn_attacks(Color::Black, Square::E4);
        assert!(ba.contains(Square::D3) && ba.contains(Square::F3) && ba.count() == 2);
    }

    #[test]
    fn rook_blocked_by_occupancy() {
        // Rook on a1, a blocker on a4: the ray covers a2,a3,a4 then stops.
        let occ = Bitboard::from_square(Square::A4);
        let attacks = rook_attacks(Square::A1, occ);
        assert!(attacks.contains(Square::A2));
        assert!(attacks.contains(Square::A4)); // includes the blocker (capturable)
        assert!(!attacks.contains(Square::A5)); // ray stopped
    }

    #[test]
    fn detects_check() {
        use crate::board::Board;
        // Black rook on e8 gives check to a white king on e1 down the open file.
        let board = Board::from_fen("4r3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(is_square_attacked(&board, Square::E1, Color::Black));
        assert!(!is_square_attacked(&board, Square::D1, Color::Black));
    }
}
