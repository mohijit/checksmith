//! Evaluation parameter vector and feature trace.
//!
//! ## Design
//!
//! The Texel tuner works by minimising the mean-squared error (MSE) between
//! the sigmoid of the static evaluation and the recorded game result, over a
//! large dataset of positions.
//!
//! For this to be fast, the evaluation must be *linear in the parameters* so
//! that changing one weight by 1 only requires an O(N) score update, not an
//! O(N × full_eval) recomputation.
//!
//! ### EvalTrace
//!
//! `compute_trace` mirrors `crate::eval::evaluate` but instead of multiplying
//! feature counts by hard-coded constants, it records the *counts* themselves
//! into an [`EvalTrace`].  All values are from White's perspective (positive
//! = good for White).
//!
//! ### EvalParams
//!
//! [`EvalParams`] holds every tunable [`Score`] in a named struct.
//! [`EvalParams::from_engine`] initialises it from the compile-time constants
//! in the handcrafted evaluator.  [`to_vec`] / [`from_vec`] flatten/restore
//! the struct to a contiguous `Vec<i32>` for the optimiser.
//!
//! ### score_from_trace
//!
//! Given a trace and a parameter set, `score_from_trace` performs the dot
//! product and tapering to reproduce the exact centipawn score that
//! `evaluate()` would return — a test verifies they match.
//!
//! ### Parameter vector layout
//!
//! ```text
//!   indices  0..10   : material  (5 pieces × mg/eg)
//!   indices 10..778  : PST       (6 pieces × 64 sq × mg/eg)
//!   indices 778..818 : 20 scalar weights × mg/eg
//!   indices 818..834 : passed pawn rank bonuses × mg/eg
//!   indices 834..850 : candidate pawn rank bonuses × mg/eg
//!   indices 850..866 : connected-passed pawn rank bonuses × mg/eg
//! ```
//!
//! Total: [`N_PARAMS`] = 866.

use crate::board::bitboard::Bitboard;
use crate::board::{Board, Color, PieceType};
use crate::eval::material;
use crate::eval::pawns::{adjacent_files_bb, file_bb, forward_ranks, rank_bb, relative_rank};
use crate::eval::pst;
use crate::eval::score::{taper, Score, TOTAL_PHASE};
use crate::movegen::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
};

// ─── Parameter count ──────────────────────────────────────────────────────────

/// Number of scalar parameters in a feature, including isolated/doubled.
const N_SCALAR_FEATURES: usize = 20;

/// Total number of tunable parameters in [`EvalParams`].
pub const N_PARAMS: usize =
    5 * 2            // material (5 pieces × mg/eg)
    + 6 * 64 * 2     // PST      (6 pieces × 64 sq × mg/eg)
    + N_SCALAR_FEATURES * 2  // scalar weights × mg/eg
    + 8 * 2          // passed pawn bonuses by rank
    + 8 * 2          // candidate pawn bonuses by rank
    + 8 * 2;         // connected-passed pawn bonuses by rank
// = 10 + 768 + 40 + 48 = 866

// ─── EvalParams ───────────────────────────────────────────────────────────────

/// All tunable evaluation weights as [`Score`] pairs (mg / eg).
///
/// Initialise with [`EvalParams::from_engine`] to get the current hand-crafted
/// values.  After tuning, call [`to_vec`] to extract the updated values and
/// paste them back into the source.
#[derive(Clone, Debug)]
pub struct EvalParams {
    // Material base values (King is always 0, not included).
    pub material: [Score; 5],      // Pawn, Knight, Bishop, Rook, Queen

    // Piece-square tables. Visual layout (rank 8 at top = index 0).
    // White looks up with `sq.index() ^ 56`; Black uses `sq.index()` directly.
    pub pst: [[Score; 64]; 6],     // indexed by PieceType::index()

    // Mobility (per reachable square, by piece type).
    pub knight_mobility: Score,
    pub bishop_mobility: Score,
    pub rook_mobility:   Score,
    pub queen_mobility:  Score,

    // Structural.
    pub bishop_pair:     Score,
    pub bad_bishop_pawn: Score,    // per own pawn on same-colour sq as the bishop
    pub rook_open_file:  Score,
    pub rook_semi_open:  Score,
    pub rook_on_seventh: Score,

    // King safety.
    pub king_shield_pawn:     Score,
    pub king_open_file:       Score,
    pub king_semi_open:       Score,
    pub king_attacker_knight: Score, // per enemy knight with ≥1 attack in king zone
    pub king_attacker_bishop: Score,
    pub king_attacker_rook:   Score,
    pub king_attacker_queen:  Score,

    // Outposts.
    pub outpost_knight: Score,
    pub outpost_bishop: Score,

    // Pawn structure.
    pub isolated: Score,
    pub doubled:  Score,

    // Rank-indexed arrays (relative rank 0..7; index 0 and 7 are always 0).
    pub passed:           [Score; 8],
    pub candidate:        [Score; 8],
    pub connected_passed: [Score; 8],
}

impl EvalParams {
    /// Initialise from the hand-crafted constants baked into the engine.
    /// This is the baseline before any tuning runs.
    pub fn from_engine() -> Self {
        use crate::eval::pst::pst as pst_fn;
        use crate::board::Square;

        // Build PST tables: for each (piece, visual_index) pair, store the
        // Score that `pst_fn(pt, White, sq)` returns for the White piece on the
        // square corresponding to visual_index.
        let mut pst_tables = [[Score::ZERO; 64]; 6];
        for (pt_idx, &pt) in PieceType::ALL.iter().enumerate() {
            for sq_raw in 0u8..64 {
                let sq = Square::new(sq_raw);
                // White flip: visual_index = sq_raw ^ 56.
                // pst_fn(White, sq) uses visual_index internally.
                pst_tables[pt_idx][sq_raw as usize ^ 56] = pst_fn(pt, Color::White, sq);
            }
        }

        EvalParams {
            material: [
                material::material(PieceType::Pawn),
                material::material(PieceType::Knight),
                material::material(PieceType::Bishop),
                material::material(PieceType::Rook),
                material::material(PieceType::Queen),
            ],
            pst: pst_tables,

            knight_mobility: Score::new(4, 4),
            bishop_mobility: Score::new(4, 4),
            rook_mobility:   Score::new(2, 4),
            queen_mobility:  Score::new(1, 2),

            bishop_pair:     Score::new(30, 50),
            bad_bishop_pawn: Score::new(-3, -6),
            rook_open_file:  Score::new(25, 12),
            rook_semi_open:  Score::new(12,  6),
            rook_on_seventh: Score::new(10, 20),

            king_shield_pawn:     Score::new( 9,  0),
            king_open_file:       Score::new(-15, 0),
            king_semi_open:       Score::new( -7, 0),
            king_attacker_knight: Score::new(-2,  0),
            king_attacker_bishop: Score::new(-2,  0),
            king_attacker_rook:   Score::new(-3,  0),
            king_attacker_queen:  Score::new(-5,  0),

            outpost_knight: Score::new(25, 12),
            outpost_bishop: Score::new(12,  6),

            isolated: Score::new(-12, -18),
            doubled:  Score::new(-10, -22),

            passed: [
                Score::ZERO,
                Score::new(  5,  12),
                Score::new( 10,  22),
                Score::new( 20,  40),
                Score::new( 35,  68),
                Score::new( 60, 110),
                Score::new(100, 160),
                Score::ZERO,
            ],
            candidate: [
                Score::ZERO,
                Score::new( 3,  6),
                Score::new( 6, 12),
                Score::new(12, 22),
                Score::new(20, 40),
                Score::new(35, 68),
                Score::new(60,100),
                Score::ZERO,
            ],
            connected_passed: [
                Score::ZERO,
                Score::new( 2,  5),
                Score::new( 5, 10),
                Score::new(10, 20),
                Score::new(18, 35),
                Score::new(30, 60),
                Score::new(50,100),
                Score::ZERO,
            ],
        }
    }

    // ── Serialisation ─────────────────────────────────────────────────────────

    /// Flatten to a contiguous `Vec<i32>` following the documented layout.
    pub fn to_vec(&self) -> Vec<i32> {
        let mut v = Vec::with_capacity(N_PARAMS);
        // Material
        for s in &self.material { v.push(s.mg); v.push(s.eg); }
        // PST
        for pt in &self.pst { for s in pt { v.push(s.mg); v.push(s.eg); } }
        // Scalars (order must match `scalar_count` in param_delta)
        for s in self.scalar_slice() { v.push(s.mg); v.push(s.eg); }
        // Rank arrays
        for s in &self.passed           { v.push(s.mg); v.push(s.eg); }
        for s in &self.candidate        { v.push(s.mg); v.push(s.eg); }
        for s in &self.connected_passed { v.push(s.mg); v.push(s.eg); }
        debug_assert_eq!(v.len(), N_PARAMS);
        v
    }

    /// Reconstruct from a flat vector produced by [`to_vec`].
    pub fn from_vec(v: &[i32]) -> Self {
        assert_eq!(v.len(), N_PARAMS, "parameter vector length mismatch");
        let mut i = 0;
        let mut take = |v: &[i32], i: &mut usize| -> Score {
            let s = Score::new(v[*i], v[*i + 1]);
            *i += 2;
            s
        };

        let mut material = [Score::ZERO; 5];
        for m in &mut material { *m = take(v, &mut i); }

        let mut pst = [[Score::ZERO; 64]; 6];
        for pt in &mut pst { for sq in pt { *sq = take(v, &mut i); } }

        let knight_mobility = take(v, &mut i);
        let bishop_mobility = take(v, &mut i);
        let rook_mobility   = take(v, &mut i);
        let queen_mobility  = take(v, &mut i);
        let bishop_pair     = take(v, &mut i);
        let bad_bishop_pawn = take(v, &mut i);
        let rook_open_file  = take(v, &mut i);
        let rook_semi_open  = take(v, &mut i);
        let rook_on_seventh = take(v, &mut i);
        let king_shield_pawn     = take(v, &mut i);
        let king_open_file       = take(v, &mut i);
        let king_semi_open       = take(v, &mut i);
        let king_attacker_knight = take(v, &mut i);
        let king_attacker_bishop = take(v, &mut i);
        let king_attacker_rook   = take(v, &mut i);
        let king_attacker_queen  = take(v, &mut i);
        let outpost_knight = take(v, &mut i);
        let outpost_bishop = take(v, &mut i);
        let isolated = take(v, &mut i);
        let doubled  = take(v, &mut i);

        let mut passed           = [Score::ZERO; 8];
        let mut candidate        = [Score::ZERO; 8];
        let mut connected_passed = [Score::ZERO; 8];
        for r in &mut passed           { *r = take(v, &mut i); }
        for r in &mut candidate        { *r = take(v, &mut i); }
        for r in &mut connected_passed { *r = take(v, &mut i); }

        EvalParams {
            material, pst,
            knight_mobility, bishop_mobility, rook_mobility, queen_mobility,
            bishop_pair, bad_bishop_pawn, rook_open_file, rook_semi_open, rook_on_seventh,
            king_shield_pawn, king_open_file, king_semi_open,
            king_attacker_knight, king_attacker_bishop, king_attacker_rook, king_attacker_queen,
            outpost_knight, outpost_bishop,
            isolated, doubled,
            passed, candidate, connected_passed,
        }
    }

    /// Print any parameter that differs from `other` (used to show tuning progress).
    pub fn print_diff(&self, baseline: &EvalParams) {
        let mine = self.to_vec();
        let base = baseline.to_vec();
        let mut changed = 0usize;
        for (i, (a, b)) in mine.iter().zip(base.iter()).enumerate() {
            if a != b {
                println!("  param[{}]: {} -> {}", i, b, a);
                changed += 1;
            }
        }
        if changed == 0 { println!("  (no changes)"); }
    }

    // Return scalar slice in the fixed order expected by to_vec/from_vec/param_delta.
    fn scalar_slice(&self) -> [Score; N_SCALAR_FEATURES] {
        [
            self.knight_mobility, self.bishop_mobility,
            self.rook_mobility,   self.queen_mobility,
            self.bishop_pair,     self.bad_bishop_pawn,
            self.rook_open_file,  self.rook_semi_open, self.rook_on_seventh,
            self.king_shield_pawn, self.king_open_file, self.king_semi_open,
            self.king_attacker_knight, self.king_attacker_bishop,
            self.king_attacker_rook,   self.king_attacker_queen,
            self.outpost_knight,  self.outpost_bishop,
            self.isolated,        self.doubled,
        ]
    }
}

// ─── EvalTrace ────────────────────────────────────────────────────────────────

/// Feature counts for one position, from White's perspective.
///
/// Each field stores the **net White − Black** count for that feature.
/// Positive values mean White benefits from this feature more than Black.
///
/// The trace is computed by [`compute_trace`] and consumed by
/// [`score_from_trace`] and the Texel coordinate-descent loop.
#[derive(Clone)]
pub struct EvalTrace {
    // Net White-Black piece counts (material).
    pub material: [i32; 5],   // [Pawn..Queen]

    // Net PST occupancy: pst[pt][visual_index] = +1 White, -1 Black.
    // Visual index: same as the index used inside pst.rs tables.
    pub pst: [[i32; 64]; 6],

    // Mobility (net square counts).
    pub knight_mobility: i32,
    pub bishop_mobility: i32,
    pub rook_mobility:   i32,
    pub queen_mobility:  i32,

    // Structural (net White advantage count for each feature).
    pub bishop_pair:     i32,
    pub bad_bishop_pawn: i32,
    pub rook_open_file:  i32,
    pub rook_semi_open:  i32,
    pub rook_on_seventh: i32,

    // King safety.
    pub king_shield_pawn:     i32,
    pub king_open_file:       i32,
    pub king_semi_open:       i32,
    // king_attacker_X:
    //   positive = white pieces attacking black's king zone
    //   negative = black pieces attacking white's king zone
    pub king_attacker_knight: i32,
    pub king_attacker_bishop: i32,
    pub king_attacker_rook:   i32,
    pub king_attacker_queen:  i32,

    // Outposts.
    pub outpost_knight: i32,
    pub outpost_bishop: i32,

    // Pawn structure (net).
    pub isolated:  i32,
    pub doubled:   i32,

    // Rank-indexed (net).
    pub passed:           [i32; 8],
    pub candidate:        [i32; 8],
    pub connected_passed: [i32; 8],

    // Game phase (for tapering, not a parameter index).
    pub phase: i32,
}

impl Default for EvalTrace {
    fn default() -> Self {
        EvalTrace {
            material: [0; 5],
            pst: [[0i32; 64]; 6],
            knight_mobility: 0, bishop_mobility: 0,
            rook_mobility: 0,   queen_mobility: 0,
            bishop_pair: 0,     bad_bishop_pawn: 0,
            rook_open_file: 0,  rook_semi_open: 0, rook_on_seventh: 0,
            king_shield_pawn: 0, king_open_file: 0, king_semi_open: 0,
            king_attacker_knight: 0, king_attacker_bishop: 0,
            king_attacker_rook: 0,   king_attacker_queen: 0,
            outpost_knight: 0, outpost_bishop: 0,
            isolated: 0, doubled: 0,
            passed:           [0; 8],
            candidate:        [0; 8],
            connected_passed: [0; 8],
            phase: 0,
        }
    }
}

// ─── compute_trace ────────────────────────────────────────────────────────────

/// Compute the feature trace for `board`.
///
/// The returned trace contains exactly the feature counts that `evaluate`
/// multiplies by its constants.  `score_from_trace(trace, EvalParams::from_engine())`
/// should reproduce `evaluate(board)` exactly.
pub fn compute_trace(board: &Board) -> EvalTrace {
    let mut t = EvalTrace::default();

    // ── Material + PST ────────────────────────────────────────────────────────
    for (pt_idx, &pt) in PieceType::ALL.iter().enumerate() {
        for sq in board.pieces_colored(Color::White, pt) {
            if pt != PieceType::King {
                t.material[pt_idx] += 1;
            }
            t.pst[pt_idx][sq.index() ^ 56] += 1; // White: visual_index = sq ^ 56
            t.phase += material::phase_value(pt);
        }
        for sq in board.pieces_colored(Color::Black, pt) {
            if pt != PieceType::King {
                t.material[pt_idx] -= 1;
            }
            t.pst[pt_idx][sq.index()] -= 1; // Black: visual_index = sq.index()
            t.phase += material::phase_value(pt);
        }
    }

    // ── Mobility ──────────────────────────────────────────────────────────────
    trace_mobility(board, &mut t);

    // ── King safety ───────────────────────────────────────────────────────────
    trace_king_safety(board, &mut t);

    // ── Bishop pair ───────────────────────────────────────────────────────────
    let wb = board.pieces_colored(Color::White, PieceType::Bishop).count();
    let bb = board.pieces_colored(Color::Black, PieceType::Bishop).count();
    t.bishop_pair += (wb >= 2) as i32 - (bb >= 2) as i32;

    // ── Bad bishop ────────────────────────────────────────────────────────────
    trace_bad_bishop(board, &mut t);

    // ── Outposts ──────────────────────────────────────────────────────────────
    trace_outposts(board, &mut t);

    // ── Rook activity ─────────────────────────────────────────────────────────
    trace_rook_activity(board, &mut t);

    // ── Pawn structure ────────────────────────────────────────────────────────
    trace_pawns(board, &mut t);

    t
}

fn trace_mobility(board: &Board, t: &mut EvalTrace) {
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let own = board.color(color);
        let occ = board.occupancy();
        for sq in board.pieces_colored(color, PieceType::Knight) {
            t.knight_mobility += sign * (knight_attacks(sq) & !own).count() as i32;
        }
        for sq in board.pieces_colored(color, PieceType::Bishop) {
            t.bishop_mobility += sign * (bishop_attacks(sq, occ) & !own).count() as i32;
        }
        for sq in board.pieces_colored(color, PieceType::Rook) {
            t.rook_mobility += sign * (rook_attacks(sq, occ) & !own).count() as i32;
        }
        for sq in board.pieces_colored(color, PieceType::Queen) {
            t.queen_mobility += sign * (queen_attacks(sq, occ) & !own).count() as i32;
        }
    }
}

fn trace_king_safety(board: &Board, t: &mut EvalTrace) {
    let occ = board.occupancy();

    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let Some(ksq) = board.king_square(color) else { continue; };
        let enemy     = color.opposite();
        let own_pawns = board.pieces_colored(color, PieceType::Pawn);
        let enemy_pawns = board.pieces_colored(enemy, PieceType::Pawn);

        // Pawn shield.
        let shield_files = file_bb(ksq.file()) | adjacent_files_bb(ksq.file());
        let kr = ksq.rank() as i32;
        let mut shield_zone = Bitboard::EMPTY;
        for step in 1..=2i32 {
            let r = match color { Color::White => kr + step, Color::Black => kr - step };
            if (0..8i32).contains(&r) { shield_zone |= rank_bb(r as u8); }
        }
        t.king_shield_pawn += sign * (own_pawns & shield_files & shield_zone).count() as i32;

        // Open / semi-open files near king.
        for df in -1i32..=1 {
            let f = ksq.file() as i32 + df;
            if !(0..8i32).contains(&f) { continue; }
            let fbb = file_bb(f as u8);
            if (own_pawns & fbb).is_empty() {
                if (enemy_pawns & fbb).is_empty() {
                    t.king_open_file  += sign;
                } else {
                    t.king_semi_open  += sign;
                }
            }
        }

        // Attacker-type counts near the king.
        //
        // Convention: trace.king_attacker_X = B_on_W - W_on_B (positive = Black
        // pieces attacking White's king MORE than White pieces attack Black's).
        // This matches the per-side penalty in evaluate() where KING_ATTACKER_*
        // is negative: -2 * (B_on_W - W_on_B) = 2*(W_on_B - B_on_W).
        //
        // For White loop (sign=+1): enemy=Black → B_on_W term → add +sign = +1.
        // For Black loop (sign=-1): enemy=White → W_on_B term → add +sign = -1.
        let atk_sign = sign;
        let king_zone = king_attacks(ksq) | Bitboard::from_square(ksq);
        for sq in board.pieces_colored(enemy, PieceType::Knight) {
            if (knight_attacks(sq) & king_zone).any() { t.king_attacker_knight += atk_sign; }
        }
        for sq in board.pieces_colored(enemy, PieceType::Bishop) {
            if (bishop_attacks(sq, occ) & king_zone).any() { t.king_attacker_bishop += atk_sign; }
        }
        for sq in board.pieces_colored(enemy, PieceType::Rook) {
            if (rook_attacks(sq, occ) & king_zone).any() { t.king_attacker_rook += atk_sign; }
        }
        for sq in board.pieces_colored(enemy, PieceType::Queen) {
            if (queen_attacks(sq, occ) & king_zone).any() { t.king_attacker_queen += atk_sign; }
        }
    }
}

fn trace_bad_bishop(board: &Board, t: &mut EvalTrace) {
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let own_pawns = board.pieces_colored(color, PieceType::Pawn);
        for bsq in board.pieces_colored(color, PieceType::Bishop) {
            let light = (bsq.file() + bsq.rank()) % 2 == 1;
            let count = own_pawns
                .filter(|&psq| ((psq.file() + psq.rank()) % 2 == 1) == light)
                .count() as i32;
            t.bad_bishop_pawn += sign * count;
        }
    }
}

fn trace_outposts(board: &Board, t: &mut EvalTrace) {
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let own_pawns   = board.pieces_colored(color,            PieceType::Pawn);
        let enemy_pawns = board.pieces_colored(color.opposite(), PieceType::Pawn);

        let check = |sq: crate::board::Square| -> bool {
            let rank = sq.rank();
            match color {
                Color::White if rank < 4 => return false,
                Color::Black if rank > 3 => return false,
                _ => {}
            }
            if (own_pawns & pawn_attacks(color.opposite(), sq)).is_empty() { return false; }
            let adj = adjacent_files_bb(sq.file()) & forward_ranks(color, rank);
            (enemy_pawns & adj).is_empty()
        };

        for sq in board.pieces_colored(color, PieceType::Knight) {
            if check(sq) { t.outpost_knight += sign; }
        }
        for sq in board.pieces_colored(color, PieceType::Bishop) {
            if check(sq) { t.outpost_bishop += sign; }
        }
    }
}

fn trace_rook_activity(board: &Board, t: &mut EvalTrace) {
    for color in [Color::White, Color::Black] {
        let sign    = if color == Color::White { 1 } else { -1 };
        let own_p   = board.pieces_colored(color,            PieceType::Pawn);
        let enemy_p = board.pieces_colored(color.opposite(), PieceType::Pawn);
        let seventh = match color { Color::White => 6u8, Color::Black => 1 };

        for sq in board.pieces_colored(color, PieceType::Rook) {
            let fbb = file_bb(sq.file());
            if (own_p & fbb).is_empty() {
                if (enemy_p & fbb).is_empty() {
                    t.rook_open_file  += sign;
                } else {
                    t.rook_semi_open  += sign;
                }
            }
            if sq.rank() == seventh { t.rook_on_seventh += sign; }
        }
    }
}

fn trace_pawns(board: &Board, t: &mut EvalTrace) {
    // Mirrors pawns::evaluate_pawns / pawns_for.
    for color in [Color::White, Color::Black] {
        let sign  = if color == Color::White { 1 } else { -1 };
        let own   = board.pieces_colored(color,            PieceType::Pawn);
        let enemy = board.pieces_colored(color.opposite(), PieceType::Pawn);

        let mut passed_mask = Bitboard::EMPTY;

        for sq in own {
            let file = sq.file();
            let rank = sq.rank();
            let r    = relative_rank(color, rank);

            // Isolated.
            if (own & adjacent_files_bb(file)).is_empty() { t.isolated += sign; }

            // Passed / candidate.
            let forward_zone = (file_bb(file) | adjacent_files_bb(file))
                               & forward_ranks(color, rank);
            let is_passed = (enemy & forward_zone).is_empty();
            if is_passed {
                t.passed[r] += sign;
                passed_mask.set(sq);
            } else {
                let ahead = file_bb(file) & forward_ranks(color, rank);
                if (enemy & ahead).is_empty() {
                    t.candidate[r] += sign;
                }
            }
        }

        // Doubled.
        for file in 0..8u8 {
            let count = (own & file_bb(file)).count() as i32;
            if count > 1 { t.doubled += sign * (count - 1); }
        }

        // Connected passed.
        for sq in passed_mask {
            if (passed_mask & adjacent_files_bb(sq.file())).any() {
                let r = relative_rank(color, sq.rank());
                t.connected_passed[r] += sign;
            }
        }
    }
}

// ─── score_from_trace ────────────────────────────────────────────────────────

/// Compute a centipawn score from a trace and parameter set.
///
/// This is the linear model that the tuner optimises.  When called with
/// [`EvalParams::from_engine`], the result should match `evaluate(board)`
/// exactly (verified by the `trace_matches_evaluate` test).
pub fn score_from_trace(trace: &EvalTrace, params: &EvalParams) -> i32 {
    let mut mg = 0i32;
    let mut eg = 0i32;

    macro_rules! acc {
        ($s:expr, $c:expr) => {
            mg += $s.mg * $c;
            eg += $s.eg * $c;
        };
    }

    // Material.
    for i in 0..5 { acc!(params.material[i], trace.material[i]); }

    // PST.
    for pt in 0..6 {
        for sq in 0..64 {
            acc!(params.pst[pt][sq], trace.pst[pt][sq]);
        }
    }

    // Scalars.
    acc!(params.knight_mobility,       trace.knight_mobility);
    acc!(params.bishop_mobility,       trace.bishop_mobility);
    acc!(params.rook_mobility,         trace.rook_mobility);
    acc!(params.queen_mobility,        trace.queen_mobility);
    acc!(params.bishop_pair,           trace.bishop_pair);
    acc!(params.bad_bishop_pawn,       trace.bad_bishop_pawn);
    acc!(params.rook_open_file,        trace.rook_open_file);
    acc!(params.rook_semi_open,        trace.rook_semi_open);
    acc!(params.rook_on_seventh,       trace.rook_on_seventh);
    acc!(params.king_shield_pawn,      trace.king_shield_pawn);
    acc!(params.king_open_file,        trace.king_open_file);
    acc!(params.king_semi_open,        trace.king_semi_open);
    // King attacker: trace uses "White-attacking-Black" sign convention;
    // params store the contribution per such attacker (should be positive for
    // the tuner to converge correctly).
    acc!(params.king_attacker_knight,  trace.king_attacker_knight);
    acc!(params.king_attacker_bishop,  trace.king_attacker_bishop);
    acc!(params.king_attacker_rook,    trace.king_attacker_rook);
    acc!(params.king_attacker_queen,   trace.king_attacker_queen);
    acc!(params.outpost_knight,        trace.outpost_knight);
    acc!(params.outpost_bishop,        trace.outpost_bishop);
    acc!(params.isolated,              trace.isolated);
    acc!(params.doubled,               trace.doubled);

    // Rank arrays.
    for r in 0..8 {
        acc!(params.passed[r],           trace.passed[r]);
        acc!(params.candidate[r],        trace.candidate[r]);
        acc!(params.connected_passed[r], trace.connected_passed[r]);
    }

    taper(Score::new(mg, eg), trace.phase)
}

// ─── param_delta ─────────────────────────────────────────────────────────────

/// For parameter at flat index `param_idx` and a given trace, compute the
/// change in score when that parameter increases by 1.
///
/// This is used by the coordinate-descent loop to update scores incrementally
/// without recomputing the full dot product.
pub fn param_delta(trace: &EvalTrace, param_idx: usize) -> f64 {
    let phase = trace.phase.clamp(0, TOTAL_PHASE) as f64;
    let tot   = TOTAL_PHASE as f64;
    let mg_f  = phase / tot;
    let eg_f  = 1.0 - mg_f;

    let mut idx = param_idx;

    // Material (0..10)
    if idx < 10 {
        let count  = trace.material[idx / 2] as f64;
        let factor = if idx % 2 == 0 { mg_f } else { eg_f };
        return count * factor;
    }
    idx -= 10;

    // PST (0..768)
    if idx < 768 {
        let pt    = idx / 128;
        let rest  = idx % 128;
        let sq    = rest / 2;
        let count  = trace.pst[pt][sq] as f64;
        let factor = if rest % 2 == 0 { mg_f } else { eg_f };
        return count * factor;
    }
    idx -= 768;

    // Scalars (0..40)
    if idx < 40 {
        let scalar_idx = idx / 2;
        let count      = scalar_trace_count(trace, scalar_idx) as f64;
        let factor     = if idx % 2 == 0 { mg_f } else { eg_f };
        return count * factor;
    }
    idx -= 40;

    // Passed (0..16)
    if idx < 16 {
        let r      = idx / 2;
        let count  = trace.passed[r] as f64;
        let factor = if idx % 2 == 0 { mg_f } else { eg_f };
        return count * factor;
    }
    idx -= 16;

    // Candidate (0..16)
    if idx < 16 {
        let r      = idx / 2;
        let count  = trace.candidate[r] as f64;
        let factor = if idx % 2 == 0 { mg_f } else { eg_f };
        return count * factor;
    }
    idx -= 16;

    // Connected passed (0..16)
    if idx < 16 {
        let r      = idx / 2;
        let count  = trace.connected_passed[r] as f64;
        let factor = if idx % 2 == 0 { mg_f } else { eg_f };
        return count * factor;
    }

    0.0
}

/// Extract the scalar feature count at position `scalar_idx` (0..N_SCALAR_FEATURES).
/// Order must match [`EvalParams::scalar_slice`] and [`EvalParams::from_vec`].
fn scalar_trace_count(t: &EvalTrace, scalar_idx: usize) -> i32 {
    match scalar_idx {
        0  => t.knight_mobility,
        1  => t.bishop_mobility,
        2  => t.rook_mobility,
        3  => t.queen_mobility,
        4  => t.bishop_pair,
        5  => t.bad_bishop_pawn,
        6  => t.rook_open_file,
        7  => t.rook_semi_open,
        8  => t.rook_on_seventh,
        9  => t.king_shield_pawn,
        10 => t.king_open_file,
        11 => t.king_semi_open,
        12 => t.king_attacker_knight,
        13 => t.king_attacker_bishop,
        14 => t.king_attacker_rook,
        15 => t.king_attacker_queen,
        16 => t.outpost_knight,
        17 => t.outpost_bishop,
        18 => t.isolated,
        19 => t.doubled,
        _  => 0,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, STARTING_FEN};

    fn check_trace_matches(fen: &str) {
        let board  = Board::from_fen(fen).unwrap();
        let params = EvalParams::from_engine();
        let trace  = compute_trace(&board);
        let trace_score = score_from_trace(&trace, &params);
        let eval_score  = crate::eval::evaluate(&board);

        // Adjust for side-to-move perspective: evaluate() returns from STM pov;
        // score_from_trace returns White-relative.
        let expected = if board.side_to_move == Color::White {
            eval_score
        } else {
            -eval_score
        };

        assert_eq!(
            trace_score, expected,
            "trace score {} ≠ evaluate {} for {}", trace_score, expected, fen
        );
    }

    #[test]
    fn trace_matches_evaluate_starting() {
        check_trace_matches(STARTING_FEN);
    }

    #[test]
    fn trace_matches_evaluate_midgame() {
        check_trace_matches(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"
        );
    }

    #[test]
    fn trace_matches_evaluate_endgame() {
        check_trace_matches("8/5pk1/8/2p5/2P5/8/5PK1/8 w - - 0 1");
    }

    #[test]
    fn trace_matches_evaluate_black_to_move() {
        check_trace_matches(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"
        );
    }

    #[test]
    fn round_trip_to_from_vec() {
        let params = EvalParams::from_engine();
        let v = params.to_vec();
        assert_eq!(v.len(), N_PARAMS);
        let restored = EvalParams::from_vec(&v);
        let v2 = restored.to_vec();
        assert_eq!(v, v2, "round-trip through to_vec/from_vec changed values");
    }

    #[test]
    fn param_delta_is_nonzero_for_occupied_square() {
        // Use an asymmetric position so the pawn PST contribution doesn't cancel
        // with a mirrored Black pawn.  Include a queen so phase > 0 (otherwise
        // mg_factor = 0 and the mg delta is trivially zero).
        //
        // White pawn e4: sq.index()=28, visual_index = 28^56 = 36.
        // PST param for pawn (pt=0) at visual_index 36, mg side:
        //   index = 10 + 0*128 + 36*2 = 82.
        let board = Board::from_fen("8/7k/8/4Q3/4P3/8/8/4K3 w - - 0 1").unwrap();
        let trace = compute_trace(&board);
        // White queen contributes phase_value=4 → mg_factor = 4/24 ≠ 0.
        assert!(trace.phase > 0, "need non-zero phase for mg delta test");
        assert_eq!(trace.pst[0][36], 1, "pst[pawn][36] should be +1 for White pawn on e4");
        let mg_delta = param_delta(&trace, 82);
        assert!(mg_delta.abs() > 0.0, "expected non-zero mg delta for white pawn PST at e4");
    }
}
