//! HalfKP feature extraction for NNUE evaluation.
//!
//! ## HalfKP encoding
//!
//! The input to the NNUE is a **binary sparse vector** that encodes the board
//! from each king's perspective separately.  For a given side P (White or Black):
//!
//! * The **king square** `K` (after optional vertical flip for Black) is used as
//!   a group key — a different king square means a completely different set of
//!   weights.
//! * For each **non-king piece** on the board (both colors):
//!   * `piece_type` ∈ {Pawn, Knight, Bishop, Rook, Queen} (5 types)
//!   * `same_color`: whether the piece belongs to side P
//!   * `piece_sq`: the piece's square (vertically flipped for Black perspective)
//!
//! Feature index = `king_sq × 640 + piece_type × 128 + color_bit × 64 + piece_sq`
//!
//! where:
//! * `piece_type` ∈ 0..5 (Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4)
//! * `color_bit` = 0 (same color as P) or 1 (opposite)
//! * `piece_sq`, `king_sq` ∈ 0..64
//!
//! Total per side: 64 × 640 = **40,960** inputs; only ~30 are active in any position.
//!
//! ## Vertical flip for color-invariance
//!
//! For Black's perspective the board is flipped vertically (rank 1 ↔ rank 8):
//! ```text
//! flipped_sq = sq ^ 56   (equivalent to (7 - rank) * 8 + file)
//! ```
//!
//! This means the same weight matrix works for both sides — the network learns
//! patterns like "my pawn one square in front of my king" identically regardless
//! of which color is which.
//!
//! ## Architecture constants
//!
//! | Constant | Value | Meaning |
//! |----------|-------|---------|
//! | `INPUT_FEATURES` | 40,960 | Feature-transformer input width (per side) |
//! | `HIDDEN_SIZE` | 512 | First hidden layer width |
//! | `L1_SIZE` | 32 | Second hidden layer width |
//! | `L2_SIZE` | 32 | Third hidden layer width |
//! | `QA` | 127 | ClampedReLU ceiling for hidden activations |
//! | `QB` | 127 | Scale divisor for subsequent linear layers |

use crate::board::{Board, Color, PieceType};

// ─── Architecture constants ────────────────────────────────────────────────────

pub const PIECE_TYPES: usize = 5; // Pawn, Knight, Bishop, Rook, Queen (no King)
pub const COLOR_VARIANTS: usize = 2; // friendly / enemy
pub const SQUARES: usize = 64;

/// Features per king-square bucket: 5 piece types × 2 colors × 64 squares = 640.
pub const FEATURES_PER_KING: usize = PIECE_TYPES * COLOR_VARIANTS * SQUARES;

/// Total input features (one side): 64 king squares × 640 = 40,960.
pub const INPUT_FEATURES: usize = SQUARES * FEATURES_PER_KING;

/// First hidden layer width (standard HalfKP).
pub const HIDDEN_SIZE: usize = 512;

/// Second hidden layer width.
pub const L1_SIZE: usize = 32;

/// Third hidden layer width.
pub const L2_SIZE: usize = 32;

/// Quantization ceiling for ClampedReLU activations (feature transformer output).
pub const QA: i32 = 127;

/// Scale divisor between linear layers.
pub const QB: i32 = 127;

// ─── Feature indexing ─────────────────────────────────────────────────────────

/// Convert a `PieceType` to a 0-based index (Pawn=0 .. Queen=4).
///
/// Panics if called with `PieceType::King` (kings have no HalfKP feature).
#[inline]
pub fn piece_type_index(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn   => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook   => 3,
        PieceType::Queen  => 4,
        PieceType::King   => panic!("PieceType::King has no HalfKP feature index"),
    }
}

/// Compute a single HalfKP feature index.
///
/// All indices are 0-based; `same_color` is `true` when the piece belongs to
/// the same side as the king used as perspective.
///
/// Guaranteed to be in `0..INPUT_FEATURES`.
#[inline]
pub fn feature_index(king_sq: usize, piece_type: usize, same_color: bool, piece_sq: usize) -> usize {
    debug_assert!(king_sq < 64);
    debug_assert!(piece_type < PIECE_TYPES);
    debug_assert!(piece_sq < 64);
    let color_bit = if same_color { 0 } else { 1 };
    king_sq * FEATURES_PER_KING
        + piece_type * (COLOR_VARIANTS * SQUARES)
        + color_bit * SQUARES
        + piece_sq
}

// ─── Board → active-feature lists ────────────────────────────────────────────

/// Compute the active HalfKP features for both king perspectives simultaneously.
///
/// Returns `(white_features, black_features)` where each is a list of indices
/// into the feature-transformer weight matrix for that perspective.
///
/// If either king is absent (shouldn't happen in a legal position) the
/// corresponding feature list is empty.
pub fn active_features(board: &Board) -> (Vec<usize>, Vec<usize>) {
    let wk = match board.king_square(Color::White) {
        Some(sq) => sq.index(),
        None     => return (Vec::new(), Vec::new()),
    };
    let bk = match board.king_square(Color::Black) {
        Some(sq) => sq.index(),
        None     => return (Vec::new(), Vec::new()),
    };

    // Black's perspective: flip the board vertically.
    let bk_flipped = bk ^ 56;

    // Capacity: up to 30 non-king pieces × 1 feature each, per side.
    let mut wf = Vec::with_capacity(30);
    let mut bf = Vec::with_capacity(30);

    // Iterate over all non-king piece types.
    for &pt in &[
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        let pt_idx = piece_type_index(pt);

        // White pieces.
        for sq in board.pieces_colored(Color::White, pt) {
            let sq_idx = sq.index();
            let sq_flipped = sq_idx ^ 56;
            // White's perspective: White piece = same color (color_bit = 0).
            wf.push(feature_index(wk, pt_idx, true, sq_idx));
            // Black's perspective: White piece = enemy (color_bit = 1), flip sq.
            bf.push(feature_index(bk_flipped, pt_idx, false, sq_flipped));
        }

        // Black pieces.
        for sq in board.pieces_colored(Color::Black, pt) {
            let sq_idx = sq.index();
            let sq_flipped = sq_idx ^ 56;
            // White's perspective: Black piece = enemy (color_bit = 1).
            wf.push(feature_index(wk, pt_idx, false, sq_idx));
            // Black's perspective: Black piece = same color (color_bit = 0), flip sq.
            bf.push(feature_index(bk_flipped, pt_idx, true, sq_flipped));
        }
    }

    (wf, bf)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, STARTING_FEN};

    #[test]
    fn feature_index_max_is_in_bounds() {
        // Largest possible index: king on h8, Queen, enemy, piece on h8.
        let idx = feature_index(63, PIECE_TYPES - 1, false, 63);
        assert!(idx < INPUT_FEATURES, "max feature index {} >= INPUT_FEATURES {}", idx, INPUT_FEATURES);
    }

    #[test]
    fn feature_index_min_is_zero() {
        assert_eq!(feature_index(0, 0, true, 0), 0);
    }

    #[test]
    fn feature_index_same_color_differs_from_enemy() {
        let same = feature_index(0, 0, true,  0);
        let diff = feature_index(0, 0, false, 0);
        assert_ne!(same, diff);
        assert_eq!(diff - same, SQUARES); // color_bit shifts by 64
    }

    #[test]
    fn active_features_startpos_count() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let (wf, bf) = active_features(&board);
        // 15 White non-king pieces (8P+2N+2B+2R+1Q) + 15 Black = 30 per perspective.
        assert_eq!(wf.len(), 30, "expected 30 white-perspective features, got {}", wf.len());
        assert_eq!(bf.len(), 30, "expected 30 black-perspective features, got {}", bf.len());
    }

    #[test]
    fn active_features_are_in_bounds() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let (wf, bf) = active_features(&board);
        for &f in wf.iter().chain(bf.iter()) {
            assert!(f < INPUT_FEATURES, "feature {} out of bounds", f);
        }
    }

    #[test]
    fn active_features_no_duplicates() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let (mut wf, mut bf) = active_features(&board);
        wf.sort_unstable();
        bf.sort_unstable();
        wf.dedup();
        bf.dedup();
        let (wf2, bf2) = active_features(&board);
        assert_eq!(wf.len(), wf2.len(), "duplicate white features detected");
        assert_eq!(bf.len(), bf2.len(), "duplicate black features detected");
    }

    #[test]
    fn constants_match_evaluator_constants() {
        // Guard against drift between this module and evaluator.rs stubs.
        assert_eq!(INPUT_FEATURES, 40_960);
        assert_eq!(HIDDEN_SIZE, 512);
    }
}
