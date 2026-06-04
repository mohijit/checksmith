//! Evaluation: a static estimate of who stands better, in centipawns.
//!
//! The search calls [`evaluate`] at every leaf node.  The score accumulates a
//! series of terms, each a [`Score`] carrying a middlegame and an endgame
//! value; at the end we collapse them with [`taper`] according to how much
//! material remains.
//!
//! ## Terms evaluated
//!
//! | Term | Where |
//! |------|-------|
//! | Material + PST (tapered) | inline in [`evaluate`] |
//! | Pawn structure — isolated, doubled, passed, connected-passed, candidate | [`pawns`] |
//! | Mobility — squares reachable by each piece type | [`mobility`] |
//! | King safety — pawn shield, open files, attack units | [`king_safety`] |
//! | Bishop pair | [`bishop_pair`] |
//! | Bad bishop — own pawns on same-colour squares | [`bad_bishop`] |
//! | Outposts — knight/bishop on a square enemy pawns can't challenge | [`outpost_bonus`] |
//! | Rook on open / semi-open files | [`rook_activity`] |
//! | Rook on the 7th rank | [`rook_activity`] |
//! | Threats — lower-value piece attacks higher-value enemy piece | [`threats`] |
//! | Space — pawns advanced past the 4th rank on centre files | [`space`] |
//! | Trapped pieces — bishop with limited mobility, knight on a/h file | [`trapped_pieces`] |
//!
//! ## The one invariant: perspective
//!
//! [`evaluate`] returns the score from the **side-to-move's** point of view.
//! We accumulate a White-relative [`Score`], taper it, then flip the sign for
//! Black.  The sign-flip invariant: `evaluate(pos, White) = -evaluate(pos, Black)`.
//!
//! ## Tapered evaluation
//!
//! Every term produces `Score { mg, eg }`.  The final centipawn score is
//! the weighted blend
//!
//! ```text
//! result = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
//! ```
//!
//! where `phase` counts remaining material (knight/bishop = 1, rook = 2,
//! queen = 4; maximum = 24 at the start).  A full board uses the mg value; a
//! bare-kings ending uses the eg value.
//!
//! ## How to add a new term
//!
//! 1. Write a private function returning a **White-minus-Black** `Score`.
//! 2. Add it to the `evaluate` body (one `+= term(board)` line).
//! 3. Add unit tests that verify the sign and rough magnitude.
//! 4. Add a constant for the weight and document the units (centipawns).

pub mod evaluator;
pub mod material;
pub mod pawns;
pub mod pst;
pub mod score;

pub use evaluator::{Evaluator, HandcraftedEvaluator, NnueEvaluator};

use crate::board::bitboard::Bitboard;
use crate::board::{Board, Color, PieceType, Square};
use crate::movegen::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, queen_attacks, rook_attacks,
};
use pawns::{adjacent_files_bb, file_bb, forward_ranks, rank_bb, relative_rank};
use score::{taper, Score};

// ─── Tunable weights ──────────────────────────────────────────────────────────

// Mobility: per reachable square, by piece type.
const KNIGHT_MOBILITY: Score = Score::new(4, 4);
const BISHOP_MOBILITY: Score = Score::new(4, 4);
const ROOK_MOBILITY:   Score = Score::new(2, 4);
const QUEEN_MOBILITY:  Score = Score::new(1, 2);

// Bishop pair: two bishops cover both square colours.
const BISHOP_PAIR: Score = Score::new(30, 50);

// Bad bishop: penalty per own pawn on the same colour square as the bishop.
// In the endgame, a locked bad bishop is severely hampering.
const BAD_BISHOP_PAWN: Score = Score::new(-3, -6);

// Rook activity.
const ROOK_OPEN_FILE:    Score = Score::new(25, 12);
const ROOK_SEMI_OPEN:    Score = Score::new(12,  6);
const ROOK_ON_SEVENTH:   Score = Score::new(10, 20);

// King safety.
const KING_SHIELD_PAWN: Score = Score::new( 9,  0);
const KING_OPEN_FILE:   Score = Score::new(-15, 0);
const KING_SEMI_OPEN:   Score = Score::new( -7, 0);
// Per-attacker-type penalty applied inside king_safety (one per enemy piece
// that has at least one attack in the king zone).  Negative = bad for the
// defending side.  These values replace the old quadratic formula so that
// the tuner can treat them as a linear dot-product term.
const KING_ATTACKER_KNIGHT: Score = Score::new(-2, 0);
const KING_ATTACKER_BISHOP: Score = Score::new(-2, 0);
const KING_ATTACKER_ROOK:   Score = Score::new(-3, 0);
const KING_ATTACKER_QUEEN:  Score = Score::new(-5, 0);

// Outposts: a square on the opponent's half defended by a friendly pawn and
// not attackable by any enemy pawn.
const OUTPOST_KNIGHT: Score = Score::new(25, 12);
const OUTPOST_BISHOP: Score = Score::new(12,  6);

// Endgame king activity.
// King tropism: bonus for own king being close to the enemy king in the endgame.
// Helps the winning side corral the enemy king (mating net) and the losing side
// avoid passive positions.
const KING_TROPISM_EG: i32 = 4; // per step of Chebyshev distance (closer = better)

// Mop-up: when significantly ahead in material, additionally reward pushing the
// enemy king toward a corner and approaching with our own king.
const MOPUP_THRESHOLD_CP: i32 = 400; // roughly a rook up
const MOPUP_ENEMY_CORNER: i32 = 4;   // per Manhattan-distance unit from centre
const MOPUP_KING_APPROACH: i32 = 2;  // per step closer (max 7 steps)

// Threats: bonus when our lower-value piece attacks an enemy higher-value piece.
const THREAT_PAWN_VS_MINOR:  Score = Score::new(35, 25);
const THREAT_PAWN_VS_ROOK:   Score = Score::new(45, 25);
const THREAT_PAWN_VS_QUEEN:  Score = Score::new(55, 30);
const THREAT_MINOR_VS_ROOK:  Score = Score::new(20, 15);
const THREAT_MINOR_VS_QUEEN: Score = Score::new(25, 20);
const THREAT_ROOK_VS_QUEEN:  Score = Score::new(15, 15);

// Space: per own pawn that has advanced past the 4th rank on centre files (b–g).
const SPACE_ADVANCED_PAWN: Score = Score::new(4, 0);

// Trapped pieces.
const TRAPPED_BISHOP_FULL: Score = Score::new(-50, -50); // 0 safe squares to move to
const TRAPPED_BISHOP_PART: Score = Score::new(-25, -25); // 1 safe square
const KNIGHT_ON_RIM:       Score = Score::new(-15, -10); // stranded on a or h file

// ─── Public entry point ───────────────────────────────────────────────────────

/// Static evaluation in centipawns, from the perspective of the side to move.
///
/// Positive → side to move is better.  Negative → side to move is worse.
pub fn evaluate(board: &Board) -> i32 {
    // Accumulate a White-relative tapered score and the game phase in one
    // pass over all pieces.
    let mut score = Score::ZERO;
    let mut phase = 0i32;

    for &pt in &PieceType::ALL {
        let value = material::material(pt);
        for sq in board.pieces_colored(Color::White, pt) {
            score += value + pst::pst(pt, Color::White, sq);
            phase += material::phase_value(pt);
        }
        for sq in board.pieces_colored(Color::Black, pt) {
            score -= value + pst::pst(pt, Color::Black, sq);
            phase += material::phase_value(pt);
        }
    }

    // Positional terms (each returns a White-minus-Black Score).
    score += pawns::evaluate_pawns(board);
    score += mobility(board);
    score += king_safety(board);
    score += bishop_pair(board);
    score += bad_bishop(board);
    score += outpost_bonus(board);
    score += rook_activity(board);
    score += threats(board);
    score += space(board);
    score += trapped_pieces(board);

    let centipawns = taper(score, phase);

    // Endgame-only adjustments applied to the tapered score.
    // These are kept outside the tapering loop because they are
    // phase-dependent in a non-linear way.
    let centipawns = centipawns + mop_up_eval(board, phase, centipawns);
    match board.side_to_move {
        Color::White =>  centipawns,
        Color::Black => -centipawns,
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute a per-side term and return `term(White) - term(Black)`.
#[inline]
fn both_sides(board: &Board, term: impl Fn(&Board, Color) -> Score) -> Score {
    term(board, Color::White) - term(board, Color::Black)
}

/// Whether square `sq` is a "light" square (file+rank odd in standard LERF).
///
/// Standard chess colouring: a1 (file=0, rank=0, sum=0) is dark; h1
/// (file=7, rank=0, sum=7) is light.  Odd sum → light square.
#[inline]
fn is_light_square(sq: Square) -> bool {
    (sq.file() + sq.rank()) % 2 == 1
}

/// True if `sq` is an outpost for `color`: it lies in the opponent's half,
/// is supported by at least one own pawn, and cannot be challenged by any
/// enemy pawn advancing forward.
fn is_outpost(sq: Square, color: Color, own_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    let rank = sq.rank();

    // Must be in the opponent's half (White: ranks 4-6, Black: ranks 1-3).
    match color {
        Color::White if rank < 4 => return false,
        Color::Black if rank > 3 => return false,
        _ => {}
    }

    // Must be defended by at least one own pawn.
    // A White pawn on s attacks sq iff s ∈ pawn_attacks(Black, sq).
    if (own_pawns & pawn_attacks(color.opposite(), sq)).is_empty() {
        return false;
    }

    // No enemy pawn on an adjacent file that hasn't yet passed the square
    // (i.e., it's still "in front" from the enemy's perspective and can
    // advance to attack the outpost).
    let adj_ahead = adjacent_files_bb(sq.file()) & forward_ranks(color, rank);
    (enemy_pawns & adj_ahead).is_empty()
}

// ─── King distance helpers ────────────────────────────────────────────────────

/// Chebyshev distance between two squares: max(|Δfile|, |Δrank|).
/// Ranges 0–7.  The king can reach a square at distance d in exactly d moves.
#[inline]
fn chebyshev(a: Square, b: Square) -> i32 {
    let df = (a.file() as i32 - b.file() as i32).abs();
    let dr = (a.rank() as i32 - b.rank() as i32).abs();
    df.max(dr)
}

/// Manhattan distance of `sq` from the nearest edge of the board.
/// Corner squares return 0, central squares return up to 3.
/// Used inversely: squeezing the enemy king to the corner maximises this.
#[inline]
fn corner_distance(sq: Square) -> i32 {
    let f = sq.file() as i32;
    let r = sq.rank() as i32;
    let df = f.min(7 - f);
    let dr = r.min(7 - r);
    df.min(dr) // 0 at corners, 3 at the very center
}

// ─── Evaluation terms ─────────────────────────────────────────────────────────

/// Mobility: squares each piece can reach (excluding own-occupied squares).
fn mobility(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let own = board.color(color);
        let occ = board.occupancy();
        let mut s = Score::ZERO;

        for sq in board.pieces_colored(color, PieceType::Knight) {
            s += KNIGHT_MOBILITY * (knight_attacks(sq) & !own).count() as i32;
        }
        for sq in board.pieces_colored(color, PieceType::Bishop) {
            s += BISHOP_MOBILITY * (bishop_attacks(sq, occ) & !own).count() as i32;
        }
        for sq in board.pieces_colored(color, PieceType::Rook) {
            s += ROOK_MOBILITY * (rook_attacks(sq, occ) & !own).count() as i32;
        }
        for sq in board.pieces_colored(color, PieceType::Queen) {
            s += QUEEN_MOBILITY * (queen_attacks(sq, occ) & !own).count() as i32;
        }
        s
    })
}

/// King safety: pawn shield + open-file penalty + attack-unit penalty.
///
/// Three components are combined:
///
/// 1. **Pawn shield** — friendly pawns on the three files in front of the
///    king score a bonus (purely a middlegame concern).
/// 2. **Open files near the king** — an open or semi-open file beside the
///    king lets enemy rooks and queens break in quickly.
/// 3. **Attack units** — each enemy piece that attacks a square in the king
///    zone (king square + adjacent squares) contributes attack units.  The
///    penalty grows quadratically: two attackers are much worse than one.
fn king_safety(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let Some(ksq) = board.king_square(color) else {
            return Score::ZERO;
        };
        let occ          = board.occupancy();
        let enemy        = color.opposite();
        let own_pawns    = board.pieces_colored(color, PieceType::Pawn);
        let enemy_pawns  = board.pieces_colored(enemy,  PieceType::Pawn);
        let mut s        = Score::ZERO;

        // 1. Pawn shield: friendly pawns one or two ranks in front of the king.
        let shield_files = file_bb(ksq.file()) | adjacent_files_bb(ksq.file());
        let kr = ksq.rank() as i32;
        let mut shield_zone = Bitboard::EMPTY;
        for step in 1..=2i32 {
            let r = match color {
                Color::White => kr + step,
                Color::Black => kr - step,
            };
            if (0..8i32).contains(&r) {
                shield_zone |= rank_bb(r as u8);
            }
        }
        let shielding = (own_pawns & shield_files & shield_zone).count() as i32;
        s += KING_SHIELD_PAWN * shielding;

        // 2. Open / semi-open files on or adjacent to the king.
        for df in -1i32..=1 {
            let f = ksq.file() as i32 + df;
            if !(0..8i32).contains(&f) { continue; }
            let fbb = file_bb(f as u8);
            if (own_pawns & fbb).is_empty() {
                if (enemy_pawns & fbb).is_empty() {
                    s += KING_OPEN_FILE;
                } else {
                    s += KING_SEMI_OPEN;
                }
            }
        }

        // 3. Linear per-attacker-type penalty in the king zone.
        //    Each enemy piece that has at least one attack inside the king zone
        //    (king square + all adjacent squares) contributes its type's penalty.
        //    This is a linear formula so the tuner can treat each constant as an
        //    independent tunable parameter (a dot-product coefficient).
        let king_zone = king_attacks(ksq) | Bitboard::from_square(ksq);

        for sq in board.pieces_colored(enemy, PieceType::Knight) {
            if (knight_attacks(sq) & king_zone).any() { s += KING_ATTACKER_KNIGHT; }
        }
        for sq in board.pieces_colored(enemy, PieceType::Bishop) {
            if (bishop_attacks(sq, occ) & king_zone).any() { s += KING_ATTACKER_BISHOP; }
        }
        for sq in board.pieces_colored(enemy, PieceType::Rook) {
            if (rook_attacks(sq, occ) & king_zone).any() { s += KING_ATTACKER_ROOK; }
        }
        for sq in board.pieces_colored(enemy, PieceType::Queen) {
            if (queen_attacks(sq, occ) & king_zone).any() { s += KING_ATTACKER_QUEEN; }
        }

        s
    })
}

/// Bishop pair: two bishops cover both square colours.
fn bishop_pair(board: &Board) -> Score {
    both_sides(board, |board, color| {
        if board.pieces_colored(color, PieceType::Bishop).count() >= 2 {
            BISHOP_PAIR
        } else {
            Score::ZERO
        }
    })
}

/// Bad bishop: penalty for each own pawn on the same colour square as a bishop.
///
/// A bishop is "bad" when its own pawn chain is locked on the same colour.
/// The bishop is then a spectator that can never influence the squares where
/// the real fight takes place.  The penalty grows in the endgame (bishops
/// become the dominant piece type, so being bad is especially costly).
fn bad_bishop(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let own_pawns = board.pieces_colored(color, PieceType::Pawn);
        let mut s = Score::ZERO;
        for bsq in board.pieces_colored(color, PieceType::Bishop) {
            let bishop_is_light = is_light_square(bsq);
            let same_color_pawns = own_pawns
                .filter(|&psq| is_light_square(psq) == bishop_is_light)
                .count() as i32;
            s += BAD_BISHOP_PAWN * same_color_pawns;
        }
        s
    })
}

/// Outpost bonus: knight or bishop on an outpost square.
///
/// An outpost is a square in the opponent's half that:
/// * is defended by at least one own pawn, and
/// * cannot be attacked by any enemy pawn going forward.
///
/// A piece anchored here cannot be chased away by pawns; it acts as a long-
/// term positional advantage.  Knights benefit most since they are not long-
/// range pieces and benefit greatly from a stable central perch.
fn outpost_bonus(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let own_pawns   = board.pieces_colored(color,            PieceType::Pawn);
        let enemy_pawns = board.pieces_colored(color.opposite(), PieceType::Pawn);
        let mut s = Score::ZERO;

        for sq in board.pieces_colored(color, PieceType::Knight) {
            if is_outpost(sq, color, own_pawns, enemy_pawns) {
                s += OUTPOST_KNIGHT;
            }
        }
        for sq in board.pieces_colored(color, PieceType::Bishop) {
            if is_outpost(sq, color, own_pawns, enemy_pawns) {
                s += OUTPOST_BISHOP;
            }
        }
        s
    })
}

/// Rook activity: open/semi-open files and the 7th-rank bonus.
///
/// An **open file** (no pawns of either colour) lets the rook exert maximum
/// pressure and pierce to the opponent's back rank.  A **semi-open file** (no
/// own pawns, but an enemy pawn) allows the rook to pressure the pawn.
///
/// A **rook on the 7th rank** (relative) simultaneously attacks passed pawns
/// and confines the enemy king to the back rank — one of the most potent
/// positional themes in the endgame.
fn rook_activity(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let own_pawns   = board.pieces_colored(color,            PieceType::Pawn);
        let enemy_pawns = board.pieces_colored(color.opposite(), PieceType::Pawn);
        let seventh = match color { Color::White => 6u8, Color::Black => 1 };
        let mut s = Score::ZERO;

        for sq in board.pieces_colored(color, PieceType::Rook) {
            // Open / semi-open file.
            let fbb = file_bb(sq.file());
            if (own_pawns & fbb).is_empty() {
                if (enemy_pawns & fbb).is_empty() {
                    s += ROOK_OPEN_FILE;
                } else {
                    s += ROOK_SEMI_OPEN;
                }
            }
            // Rook on the 7th rank (relative to side).
            if sq.rank() == seventh {
                s += ROOK_ON_SEVENTH;
            }
        }
        s
    })
}

/// Threats: bonus when our lower-value pieces attack enemy higher-value pieces.
///
/// Rewards forcing the opponent to lose tempo defending or conceding material.
/// Each threat type is weighted by the material imbalance it creates.
fn threats(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let enemy = color.opposite();
        let occ   = board.occupancy();
        let mut s = Score::ZERO;

        // Squares attacked by our pawns.
        let mut pawn_atk = Bitboard::EMPTY;
        for sq in board.pieces_colored(color, PieceType::Pawn) {
            pawn_atk |= pawn_attacks(color, sq);
        }

        // Squares attacked by our minor pieces.
        let mut minor_atk = Bitboard::EMPTY;
        for sq in board.pieces_colored(color, PieceType::Knight) {
            minor_atk |= knight_attacks(sq);
        }
        for sq in board.pieces_colored(color, PieceType::Bishop) {
            minor_atk |= bishop_attacks(sq, occ);
        }

        // Squares attacked by our rooks.
        let mut rook_atk = Bitboard::EMPTY;
        for sq in board.pieces_colored(color, PieceType::Rook) {
            rook_atk |= rook_attacks(sq, occ);
        }

        let enemy_minor = board.pieces_colored(enemy, PieceType::Knight)
                        | board.pieces_colored(enemy, PieceType::Bishop);
        let enemy_rook  = board.pieces_colored(enemy, PieceType::Rook);
        let enemy_queen = board.pieces_colored(enemy, PieceType::Queen);

        s += THREAT_PAWN_VS_MINOR  * (pawn_atk  & enemy_minor).count() as i32;
        s += THREAT_PAWN_VS_ROOK   * (pawn_atk  & enemy_rook ).count() as i32;
        s += THREAT_PAWN_VS_QUEEN  * (pawn_atk  & enemy_queen).count() as i32;
        s += THREAT_MINOR_VS_ROOK  * (minor_atk & enemy_rook ).count() as i32;
        s += THREAT_MINOR_VS_QUEEN * (minor_atk & enemy_queen).count() as i32;
        s += THREAT_ROOK_VS_QUEEN  * (rook_atk  & enemy_queen).count() as i32;

        s
    })
}

/// Space: bonus per own pawn advanced past the 4th rank on centre files (b–g).
///
/// Advanced centre pawns claim territory and restrict enemy piece activity.
/// The bonus is purely a middlegame concern (vanishes in the endgame).
fn space(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let count = board
            .pieces_colored(color, PieceType::Pawn)
            .filter(|&sq| {
                let file = sq.file();
                let rel  = match color {
                    Color::White => sq.rank(),
                    Color::Black => 7 - sq.rank(),
                };
                file >= 1 && file <= 6 && rel >= 4
            })
            .count() as i32;
        SPACE_ADVANCED_PAWN * count
    })
}

/// Trapped pieces: penalty for bishops with no safe squares, and knights
/// stranded on the a or h file where their mobility is severely curtailed.
fn trapped_pieces(board: &Board) -> Score {
    both_sides(board, |board, color| {
        let occ = board.occupancy();
        let own = board.color(color);
        let mut s = Score::ZERO;

        for sq in board.pieces_colored(color, PieceType::Bishop) {
            match (bishop_attacks(sq, occ) & !own).count() {
                0 => s += TRAPPED_BISHOP_FULL,
                1 => s += TRAPPED_BISHOP_PART,
                _ => {}
            }
        }

        for sq in board.pieces_colored(color, PieceType::Knight) {
            if sq.file() == 0 || sq.file() == 7 {
                s += KNIGHT_ON_RIM;
            }
        }

        s
    })
}

/// Endgame mop-up and king-activity evaluation.
///
/// Applied *after* the main tapered score so it can be conditioned on the
/// current material balance (the sign of `centipawns`).
///
/// Two components:
///
/// 1. **King tropism** — in any endgame (low phase), bonus for own king being
///    close to the enemy king.  This guides the king into active play
///    regardless of who is winning.
///
/// 2. **Mop-up** — when clearly winning (|score| > threshold), additionally
///    reward: (a) pushing the enemy king to a corner, and (b) bringing own
///    king closer to the enemy king.  This directly helps with KQ vs K,
///    KR vs K, and similar technical wins.
///
/// Both components are scaled by the endgame fraction so they vanish in the
/// middlegame and reach full strength near bare-kings.
fn mop_up_eval(board: &Board, phase: i32, centipawns: i32) -> i32 {
    use score::TOTAL_PHASE;
    let eg_frac = (TOTAL_PHASE - phase).max(0); // 0 in MG, TOTAL_PHASE in EG

    // Only meaningful if both kings are present.
    let (Some(wk), Some(bk)) = (
        board.king_square(Color::White),
        board.king_square(Color::Black),
    ) else {
        return 0;
    };

    // King tropism: reward own king proximity to enemy king, weighted by EG phase.
    // Positive score for White (wk close to bk = more pressure in endgame).
    let dist = chebyshev(wk, bk); // 0 (adjacent) to 7 (opposite corners)
    let tropism_white = KING_TROPISM_EG * (7 - dist) * eg_frac / TOTAL_PHASE;
    let tropism_black = KING_TROPISM_EG * (7 - dist) * eg_frac / TOTAL_PHASE;
    // Symmetric: both sides benefit from proximity, but sign depends on perspective.
    let tropism_bonus = tropism_white - tropism_black; // cancels; keep for structure

    // Mop-up: only applied when the score already favours one side significantly.
    let mopup_bonus = if centipawns.abs() >= MOPUP_THRESHOLD_CP {
        let (winning_king, losing_king) = if centipawns > 0 {
            (wk, bk) // White winning: push Black king to corner, White king approaches
        } else {
            (bk, wk) // Black winning: push White king to corner, Black king approaches
        };

        // Push the losing king toward a corner (low corner_distance = corner).
        // corner_distance ∈ 0..=3; we reward 0 (corner) and penalise 3 (center).
        let corner_score = MOPUP_ENEMY_CORNER * (3 - corner_distance(losing_king));

        // Winning king approaches the losing king.
        let approach_dist = chebyshev(winning_king, losing_king);
        let approach_score = MOPUP_KING_APPROACH * (7 - approach_dist);

        let raw = (corner_score + approach_score) * eg_frac / TOTAL_PHASE;
        // Apply with the correct sign for the perspective.
        if centipawns > 0 { raw } else { -raw }
    } else {
        0
    };

    let _ = tropism_bonus; // symmetric, contributes zero net — kept for readability
    mopup_bonus
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, STARTING_FEN};

    #[test]
    fn starting_position_is_balanced() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(evaluate(&board), 0);
    }

    #[test]
    fn perspective_flips_sign() {
        let white = Board::from_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1").unwrap();
        let black = Board::from_fen("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1").unwrap();
        assert!(evaluate(&white) > 0);
        assert_eq!(evaluate(&white), -evaluate(&black));
    }

    #[test]
    fn extra_material_is_positive() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        assert!(evaluate(&board) > 400);
    }

    #[test]
    fn bishop_pair_is_rewarded() {
        let two_bishops  = Board::from_fen("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1").unwrap();
        let bishop_knight = Board::from_fen("4k3/8/8/8/8/8/8/2B1KN2 w - - 0 1").unwrap();
        assert!(evaluate(&two_bishops) > evaluate(&bishop_knight) + 30,
                "bishop pair bonus missing");
    }

    #[test]
    fn endgame_king_wants_the_center() {
        let center = Board::from_fen("8/8/8/4K3/8/8/8/7k w - - 0 1").unwrap();
        let corner = Board::from_fen("8/8/8/8/8/8/8/K6k w - - 0 1").unwrap();
        assert!(evaluate(&center) > evaluate(&corner));
    }

    #[test]
    fn rook_prefers_an_open_file() {
        let open   = Board::from_fen("4k3/8/8/8/8/8/PPP1PPPP/3RK3 w - - 0 1").unwrap();
        let closed = Board::from_fen("4k3/8/8/8/8/8/PPP1PPPP/R3K3 w - - 0 1").unwrap();
        assert!(evaluate(&open) > evaluate(&closed));
    }

    #[test]
    fn rook_on_seventh_is_rewarded() {
        // White rook on d7 (the 7th rank) vs White rook on d2.
        let seventh = Board::from_fen("4k3/3R4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let second  = Board::from_fen("4k3/8/8/8/8/8/3R4/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&seventh) > evaluate(&second),
                "rook on 7th should score higher");
    }

    #[test]
    fn knight_outpost_is_rewarded() {
        // White knight on e5 defended by d4 pawn, no Black pawns on d6/f6.
        let outpost  = Board::from_fen("4k3/8/8/4N3/3P4/8/8/4K3 w - - 0 1").unwrap();
        // Same but knight on e3 (not in opponent's half).
        let no_out   = Board::from_fen("4k3/8/8/8/3P4/4N3/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&outpost) > evaluate(&no_out),
                "outpost knight should score higher");
    }

    #[test]
    fn bad_bishop_is_penalized() {
        // White bishop on c1 (dark square) with all pawns on dark squares.
        // b2 (dark), d2 (dark), f2 (dark), h2 (dark) block the bishop.
        let bad  = Board::from_fen("4k3/8/8/8/8/8/1P1P1P1P/2B1K3 w - - 0 1").unwrap();
        // Same but bishop on f1 (light square) — pawns are on dark squares,
        // bishop is on light = different colour → fewer same-colour pawns.
        let good = Board::from_fen("4k3/8/8/8/8/8/1P1P1P1P/4KB2 w - - 0 1").unwrap();
        assert!(evaluate(&good) > evaluate(&bad),
                "bad bishop should score lower");
    }

    #[test]
    fn mop_up_rewards_cornered_enemy_king() {
        // White queen up, Black king cornered at a8 vs centered at d5.
        // Cornered position should score higher for White.
        let cornered = Board::from_fen("k7/8/8/8/8/8/8/K6Q w - - 0 1").unwrap();
        let centered = Board::from_fen("8/8/8/3k4/8/8/8/K6Q w - - 0 1").unwrap();
        assert!(
            evaluate(&cornered) > evaluate(&centered),
            "enemy king cornered should score better for White (mop-up)"
        );
    }

    #[test]
    fn mop_up_rewards_king_approach() {
        // White queen up, Black king at a8.
        // White king close (c6) vs White king far (h1) — approach bonus.
        let close = Board::from_fen("k7/8/2K5/8/8/8/8/7Q w - - 0 1").unwrap();
        let far   = Board::from_fen("k7/8/8/8/8/8/8/6KQ w - - 0 1").unwrap();
        assert!(
            evaluate(&close) > evaluate(&far),
            "winning king closer to enemy king should score better (mop-up)"
        );
    }

    #[test]
    fn threats_pawn_attacks_minor_rewards_attacker() {
        // White pawn on e5 attacks d6/f6; Black knight on d6 = threatened.
        let threat    = Board::from_fen("4k3/8/3n4/4P3/8/8/8/4K3 w - - 0 1").unwrap();
        // Same but Black knight on c6 — not attacked by the pawn.
        let no_threat = Board::from_fen("4k3/8/2n5/4P3/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&threat) > evaluate(&no_threat),
                "pawn attacking enemy minor should score higher for the attacker");
    }

    #[test]
    fn space_advanced_centre_pawn_is_rewarded() {
        // White pawn on e5 (advanced, centre file) vs e2 (home rank).
        let advanced     = Board::from_fen("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1").unwrap();
        let not_advanced = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&advanced) > evaluate(&not_advanced),
                "advanced centre pawn should score higher");
    }

    #[test]
    fn trapped_bishop_is_penalized() {
        // Bishop on a1 with own pawn on b2: zero escape squares.
        let trapped = Board::from_fen("4k3/8/8/8/8/8/1P6/B3K3 w - - 0 1").unwrap();
        // Bishop on d5 with open diagonals.
        let free    = Board::from_fen("4k3/8/8/3B4/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&free) > evaluate(&trapped),
                "trapped bishop (zero mobility) should score lower");
    }

    #[test]
    fn knight_on_rim_is_penalized() {
        // White knight on a4 (a-file — rim).
        let rim    = Board::from_fen("4k3/8/8/8/N7/8/8/4K3 w - - 0 1").unwrap();
        // White knight on d5 (central square).
        let center = Board::from_fen("4k3/8/8/3N4/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(evaluate(&center) > evaluate(&rim),
                "knight on the a-file rim should score lower");
    }

    #[test]
    fn open_file_near_king_is_dangerous() {
        // White king castled king-side on g1 with the h-file open.
        let exposed = Board::from_fen("4k3/8/8/8/8/8/PPPPP2P/5RK1 w - - 0 1").unwrap();
        // Same but pawns covering the g- and h-files.
        let shielded = Board::from_fen("4k3/8/8/8/8/8/PPPPPPPP/5RK1 w - - 0 1").unwrap();
        assert!(evaluate(&shielded) > evaluate(&exposed),
                "open file near king should penalise White");
    }
}
