//! Draw detection: insufficient material, fifty-move rule, and repetition.
//!
//! These checks are applied at every internal search node (ply > 0) before
//! any other work.  Returning 0 early saves the cost of move generation and
//! the full evaluation, and prevents the engine from playing on in positions
//! that are already drawn by rule.
//!
//! ## Fifty-move rule
//! A position where neither side has made a pawn move or capture in 50 moves
//! (100 half-moves) is a draw by claim.  We return 0 at `halfmove_clock ≥ 100`.
//!
//! ## Insufficient material
//!
//! Certain material combinations cannot force checkmate regardless of play:
//!
//! | White  | Black  | Result |
//! |--------|--------|--------|
//! | K      | K      | Draw   |
//! | K + N  | K      | Draw   |
//! | K + B  | K      | Draw   |
//! | K      | K + N  | Draw   |
//! | K      | K + B  | Draw   |
//! | K + B  | K + B  | Draw when bishops are on the same colour |
//!
//! K + N + N vs K is technically drawn with best play but is computationally
//! trivial to verify in search and is left out; the engine will still reach a
//! correct evaluation.
//!
//! ## Threefold repetition
//!
//! A position that has appeared before in the game (detected via Zobrist hash
//! in the `hash_history` of the [`Searcher`]) is scored as 0.  We trigger on
//! the **second** occurrence (one match in history) at internal nodes so the
//! engine never walks into a perpetual-check trap.  The scan is bounded by
//! `halfmove_clock` because repetitions cannot span an irreversible move.

use crate::board::{Board, Color, PieceType};

// ── Insufficient material ─────────────────────────────────────────────────────

/// True when neither side can force checkmate with the remaining material.
///
/// The check is intentionally conservative: it only declares draws for the
/// most clear-cut material configurations.  Positions not listed here fall
/// through to normal search and evaluation.
pub fn is_insufficient_material(board: &Board) -> bool {
    // Any pawn can promote → always potentially decisive.
    if board.pieces(PieceType::Pawn).any() {
        return false;
    }
    // Any rook or queen can force mate.
    if board.pieces(PieceType::Rook).any() || board.pieces(PieceType::Queen).any() {
        return false;
    }

    // Only kings, bishops, and knights remain.
    let wn = board.pieces_colored(Color::White, PieceType::Knight).count();
    let bn = board.pieces_colored(Color::Black, PieceType::Knight).count();
    let wb = board.pieces_colored(Color::White, PieceType::Bishop).count();
    let bb_cnt = board.pieces_colored(Color::Black, PieceType::Bishop).count();

    let total_minors = wn + bn + wb + bb_cnt;

    // K vs K
    if total_minors == 0 {
        return true;
    }

    // K + single minor vs K (KN vs K, KB vs K, and symmetric)
    if total_minors == 1 {
        return true;
    }

    // K + B vs K + B: only drawn when bishops travel on the same colour.
    if wn == 0 && bn == 0 && wb == 1 && bb_cnt == 1 {
        let wsq = board
            .pieces_colored(Color::White, PieceType::Bishop)
            .lsb()
            .expect("just counted one");
        let bsq = board
            .pieces_colored(Color::Black, PieceType::Bishop)
            .lsb()
            .expect("just counted one");
        // Same-colour if (file + rank) has the same parity.
        let w_light = (wsq.file() + wsq.rank()) % 2 == 1;
        let b_light = (bsq.file() + bsq.rank()) % 2 == 1;
        return w_light == b_light;
    }

    false
}

// ── Repetition ────────────────────────────────────────────────────────────────

/// True if `hash` appears anywhere in `history[start..]`.
///
/// `start` is computed by the caller as `history.len() - halfmove_clock` so
/// that we only look back as far as the last irreversible move.
///
/// The hash already encodes side-to-move, so a hash match implies both the
/// board content AND the side to move are identical — correct by construction.
#[inline]
pub fn hash_in_slice(hash: u64, slice: &[u64]) -> bool {
    slice.iter().any(|&h| h == hash)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    // --- Insufficient material ---

    #[test]
    fn kk_is_draw() {
        assert!(is_insufficient_material(&board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")));
    }

    #[test]
    fn kn_vs_k_is_draw() {
        assert!(is_insufficient_material(&board("4k3/8/8/8/8/8/8/4KN2 w - - 0 1")));
    }

    #[test]
    fn kb_vs_k_is_draw() {
        assert!(is_insufficient_material(&board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")));
    }

    #[test]
    fn kb_vs_kb_same_colour_is_draw() {
        // Both bishops on light squares (c1 and f8 are both light: c1=file2+rank0=2 even=dark, wait...)
        // c1: file=2, rank=0, sum=2 even → dark square
        // f8: file=5, rank=7, sum=12 even → dark square
        // Same colour → draw
        assert!(is_insufficient_material(&board("5b2/8/8/8/8/8/8/2B1K2k w - - 0 1")));
    }

    #[test]
    fn kb_vs_kb_diff_colour_not_draw() {
        // c1 dark (sum=2), c8 light? c8: file=2, rank=7, sum=9 odd → light
        // Different colours → not drawn
        assert!(!is_insufficient_material(&board("2b5/8/8/8/8/8/8/2B1K2k w - - 0 1")));
    }

    #[test]
    fn kp_vs_k_not_draw() {
        assert!(!is_insufficient_material(&board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")));
    }

    #[test]
    fn kr_vs_k_not_draw() {
        assert!(!is_insufficient_material(&board("4k3/8/8/8/8/8/8/4KR2 w - - 0 1")));
    }
}
