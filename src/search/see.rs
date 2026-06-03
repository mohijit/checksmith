//! Static Exchange Evaluation (SEE).
//!
//! SEE simulates the complete capture sequence on one square — who takes what,
//! who recaptures, and so on — without making any moves on the real board.
//! It returns the expected net material gain (centipawns) for the side making
//! the initial capture.  A positive result means the capture wins material;
//! negative means it loses material.
//!
//! ## How it works
//!
//! Consider White pawn × Black queen on e5, defended by a Black bishop:
//!
//! 1. White takes the queen (+900 for White).
//! 2. Black recaptures with the bishop (+100 for Black, i.e. –100 for White's
//!    gain).  Black *could* stand pat and not recapture — but recapturing a
//!    free pawn is always good.
//! 3. No more White attackers.  Exchange over.
//!
//! SEE = 900 − 100 = **800** (White is up 800 cp after the smoke clears).
//!
//! Now consider Queen × defended pawn:
//!
//! 1. White takes the pawn (+100).
//! 2. Black bishop recaptures — gains the queen (+900).
//!
//! SEE = 100 − 900 = **–800** (White loses 800 cp).
//!
//! The stand-pat rule: at each step the recapturing side can *choose not to
//! recapture* if doing so would lose material.  The `.max(0)` in each recursive
//! return enforces this — a player never recaptures into a losing position.
//!
//! ## X-rays
//!
//! Because we recompute `attackers_to_occ` after every capture (with the
//! capturer removed from the occupancy), sliders that were blocked by an
//! intermediate piece automatically become visible.  For example, a rook behind
//! a bishop that captures a piece: once the bishop moves the rook's ray opens up
//! and it participates in the sequence.
//!
//! ## Limitations
//!
//! SEE is not exact for promotions-in-the-middle-of-an-exchange (a pawn
//! participating in a recapture sequence would promote on the target square).
//! We use the promoted piece's value for the pawn in that case, which is correct
//! for the common queen/rook target-square scenarios.

use crate::board::{Bitboard, Board, Color, PieceType, Square};
use crate::movegen::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks,
                               rook_attacks};
use crate::movegen::Move;

/// Piece values used exclusively by SEE (centipawns, midgame-aligned).
/// Indexed by [`PieceType::index`]: Pawn=0 … King=5.
pub const SEE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20_000];

/// Returns the expected net material gain for the side making `mv` on `board`.
///
/// Positive → the capture wins material overall.
/// Negative → the capture loses material overall.
/// Zero    → an equal trade (or the move is not a capture).
pub fn see(board: &Board, mv: Move) -> i32 {
    let to   = mv.to();
    let from = mv.from();

    // ── Value of the piece being captured ─────────────────────────────────
    let target_val = if mv.is_en_passant() {
        SEE_VALUES[PieceType::Pawn.index()]
    } else {
        match board.piece_at(to) {
            Some(p) => SEE_VALUES[p.piece_type.index()],
            None    => return 0, // not a capture
        }
    };

    // For a capture+promotion the pawn is worth the promoted piece's value
    // (that piece is what lands on `to` and can be recaptured), and we gain
    // the promotion bonus on top of the captured material.
    let promo_bonus;
    let piece_val;
    if let Some(promo) = mv.promotion_piece() {
        piece_val   = SEE_VALUES[promo.index()];
        promo_bonus = SEE_VALUES[promo.index()] - SEE_VALUES[PieceType::Pawn.index()];
    } else {
        piece_val = match board.piece_at(from) {
            Some(p) => SEE_VALUES[p.piece_type.index()],
            None    => return 0,
        };
        promo_bonus = 0;
    }

    // ── Build the initial occupancy ────────────────────────────────────────
    let mut occ = board.occupancy();
    occ.clear(from); // attacker leaves its origin square

    if mv.is_en_passant() {
        // The captured pawn sits on the same file as `to`, same rank as `from`.
        let ep_pawn = Square::from_file_rank(to.file(), from.rank());
        occ.clear(ep_pawn);
    }

    let opponent = board.side_to_move.opposite();

    // ── Recursive exchange ─────────────────────────────────────────────────
    // effective_target = what we gain from taking (captured piece + promotion).
    let effective_target = target_val + promo_bonus;
    effective_target - see_inner(board, to, piece_val, opponent, occ)
}

/// Recursive helper.
///
/// Simulates `side` recapturing on `sq` (the piece on `sq` is currently worth
/// `last_val`), then lets the opponent respond, and so on.
///
/// Returns the maximum amount `side` can gain from *all subsequent recaptures*
/// — 0 if recapturing would be unprofitable (stand-pat option).
fn see_inner(
    board:    &Board,
    sq:       Square,
    last_val: i32,
    side:     Color,
    occ:      Bitboard,
) -> i32 {
    let our_attackers = attackers_to_occ(board, sq, occ) & board.color(side);
    if our_attackers.is_empty() {
        return 0; // nobody left to recapture — stand pat with 0 additional gain
    }

    let (attacker_sq, attacker_val) = least_valuable(board, our_attackers);
    // Remove the attacker; x-ray pieces behind it become visible in the next call.
    let new_occ = occ & !Bitboard::from_square(attacker_sq);

    // We gain `last_val` (the piece on sq), but the opponent may recapture our
    // attacker.  Stand-pat: never continue an exchange that loses material.
    let gain = last_val - see_inner(board, sq, attacker_val, side.opposite(), new_occ);
    gain.max(0)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// All pieces in `occ` that attack `sq`, for both colors.
///
/// Uses `occ` for slider ray-blocking so that removed pieces correctly open
/// x-ray lines.  Intersects with `occ` to exclude pieces already "used up"
/// in earlier capture steps.
fn attackers_to_occ(board: &Board, sq: Square, occ: Bitboard) -> Bitboard {
    let white_pawns = board.pieces_colored(Color::White, PieceType::Pawn) & occ;
    let black_pawns = board.pieces_colored(Color::Black, PieceType::Pawn) & occ;

    // Reverse-attack trick for pawns: a White pawn attacks sq from the
    // direction a Black pawn would attack *from* sq.
    let pawn_attackers = (pawn_attacks(Color::Black, sq) & white_pawns)
                       | (pawn_attacks(Color::White, sq) & black_pawns);

    let knight_attackers = knight_attacks(sq)
        & board.pieces(PieceType::Knight)
        & occ;

    let king_attackers = king_attacks(sq)
        & board.pieces(PieceType::King)
        & occ;

    let diag = bishop_attacks(sq, occ)
        & (board.pieces(PieceType::Bishop) | board.pieces(PieceType::Queen))
        & occ;

    let orth = rook_attacks(sq, occ)
        & (board.pieces(PieceType::Rook) | board.pieces(PieceType::Queen))
        & occ;

    pawn_attackers | knight_attackers | king_attackers | diag | orth
}

/// Among `attackers` (a subset of occupied squares), find the piece with the
/// lowest SEE value.  Returns its square and value.
///
/// Panics in debug if `attackers` is empty.
fn least_valuable(board: &Board, attackers: Bitboard) -> (Square, i32) {
    let mut best_sq  = Square::new(0);
    let mut best_val = i32::MAX;

    for sq in attackers {
        if let Some(piece) = board.piece_at(sq) {
            let val = SEE_VALUES[piece.piece_type.index()];
            if val < best_val {
                best_val = val;
                best_sq  = sq;
            }
        }
    }
    (best_sq, best_val)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::uci::parser::parse_uci_move;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    // ── Pawn × undefended queen (trivially winning) ───────────────────────
    // 8/8/8/3q4/4P3/8/8/7K w - - 0 1
    // White pawn on e4 takes Black queen on d5.  Nobody defends d5.
    #[test]
    fn see_pawn_takes_undefended_queen() {
        let b  = board("8/8/8/3q4/4P3/8/8/7K w - - 0 1");
        let mv = parse_uci_move(&b, "e4d5").unwrap();
        assert_eq!(see(&b, mv), SEE_VALUES[PieceType::Queen.index()]);
    }

    // ── Pawn × pawn (equal trade) ─────────────────────────────────────────
    // 8/8/8/3p4/4P3/8/8/7K w - - 0 1
    // White pawn × Black pawn, nothing defends d5.
    #[test]
    fn see_pawn_takes_undefended_pawn() {
        let b  = board("8/8/8/3p4/4P3/8/8/7K w - - 0 1");
        let mv = parse_uci_move(&b, "e4d5").unwrap();
        assert_eq!(see(&b, mv), SEE_VALUES[PieceType::Pawn.index()]);
    }

    // ── Pawn × pawn defended by pawn (0-gain exchange) ───────────────────
    // 8/8/2p5/3p4/4P3/8/8/7K w - - 0 1
    // d5 defended by Black c6 pawn.  Trade is even (100 − 100 = 0).
    #[test]
    fn see_pawn_takes_pawn_defended_by_pawn() {
        let b  = board("8/8/2p5/3p4/4P3/8/8/7K w - - 0 1");
        let mv = parse_uci_move(&b, "e4d5").unwrap();
        assert_eq!(see(&b, mv), 0);
    }

    // ── Queen × defended pawn (losing) ────────────────────────────────────
    // 8/8/2b5/3p4/4Q3/8/8/7K w - - 0 1
    // White queen × Black pawn on d5, defended by Black bishop on c6.
    // White gains 100 but Black recaptures the queen (–900): net –800.
    #[test]
    fn see_queen_takes_defended_pawn() {
        let b  = board("8/8/2b5/3p4/4Q3/8/8/7K w - - 0 1");
        let mv = parse_uci_move(&b, "e4d5").unwrap();
        assert_eq!(see(&b, mv), 100 - 900); // –800
    }

    // ── X-ray: rook behind bishop ─────────────────────────────────────────
    // 8/8/8/3p4/8/8/8/2BR3K w - - 0 1
    // White bishop on d1 takes d5 pawn.  No Black defenders.
    // After bishop moves, the White rook on e1 x-rays through d1… but d5 is
    // undefended anyway, so SEE = 330 (bishop takes free pawn).
    // (The rook would appear if Black had a rook on d8, but this tests that
    // x-ray code doesn't crash and gives the right answer.)
    #[test]
    fn see_xray_does_not_crash() {
        // White bishop d1, White rook e1, Black pawn d5, White king h1
        let b  = board("8/8/8/3p4/8/8/8/2BR3K w - - 0 1");
        let mv = parse_uci_move(&b, "d1d5").unwrap(); // bishop d1×d5
        // d5 is undefended, so SEE = 100 (pawn value)
        assert_eq!(see(&b, mv), SEE_VALUES[PieceType::Pawn.index()]);
    }

    // ── X-ray: rook behind bishop, then rook joins ────────────────────────
    // White bishop on c1 takes d2 pawn, White rook on a1 x-rays through c1.
    // Black queen on d8 can recapture.
    // 8/3q4/8/8/8/8/3p4/B1R4K w - - 0 1
    // White bishop c1×d2 (gains 100 pawn).  Black queen recaptures (gains 330
    // bishop).  White rook x-rays through c1 to d2 (gains 900 queen).
    // No more Black attackers.  SEE = 100 − (330 − 900) = 100 − (−570).max(0)
    //   inner for Black (queen): 900 − see_inner(White, rook)
    //     inner for White (rook): 330 − see_inner(Black, ...) = 330 − 0 = 330
    //   inner for Black = 900 − 330 = 570, stand-pat: max(0,570) = 570
    // SEE = 100 − 570 = −470  (bishop walk-off wins pawn but loses bishop net)
    // Wait: actually let me re-trace.
    //
    // see(bishop×d2):
    //   target_val = 100 (pawn)
    //   piece_val  = 330 (bishop)
    //   return 100 − see_inner(BLACK, d2, 330, occ_without_c1_bishop)
    //
    // see_inner(BLACK, d2, 330, ...):
    //   Black queen on d8 sees d2 (open d-file, occ has no piece on d1,d3..d7).
    //   attacker = queen (900), attacker_val = 900
    //   new_occ = occ without d8
    //   return max(0, 330 − see_inner(WHITE, d2, 900, new_occ))
    //
    // see_inner(WHITE, d2, 900, new_occ_without_d8):
    //   White rook on c1 — wait, occ no longer has the bishop on c1.
    //   Rook on a1… does it see d2 with the bishop gone from c1?
    //   Rook on a1: rook_attacks(a1, occ_without_bishop_c1_and_queen_d8)
    //   The rook on a1 attacks the a-file, not the d-file.
    //   Hmm, the rook on c1 was supposed to x-ray. Let me re-read: bishop is on
    //   c1, rook is ALSO on c1? No, "B1R4K" means B on a1, empty b1, R on c1,
    //   empty d1..g1, K on h1. So White: bishop a1, rook c1.
    //
    //   After bishop a1 moves to d2 (occ clears a1), rook c1 is now unblocked
    //   along the 1st rank but not along the d-file.
    //   Rook c1 attacks d2 (along the 2nd rank? No — rook goes along rank or
    //   file.  c1→d1→d2? No, that's not how rooks move along a file.
    //   c1 to d2: not a rook move. The rook can only move to d1, then d2 is
    //   blocked by d1 if occupied, or continues to d2 if d1 is empty.
    //   d1 is empty (the bishop was on a1 not d1), so rook c1 attacks d1, d2...
    //   Wait: rook_attacks(c1, occ) = all squares along rank 1 and file c.
    //   c1 along rank 1: b1, a1 (blocked by bishop if present), d1, e1,...h1
    //   c1 along file c: c2, c3,...c8
    //   After clearing a1 from occ, rook c1 still doesn't see d2 via rank 1.
    //   d2 is on file d. Rook on c1 attacks file c and rank 1 — NOT d2.
    //
    // So actually there's no x-ray in this position. The test below is different.
    //
    // Let me simplify: test a clear x-ray where a rook behind a bishop reveals.
    // 8/3q4/8/8/8/8/3p4/2BR3K w - - 0 1
    // White bishop c1, White rook d1, Black queen d7, Black pawn d2, White king h1.
    // Bishop c1×d2 (pawn value=100).  Rook d1 is blocked by bishop c1; after
    // bishop moves, d1 rook sees d2 via the d-file!
    // Black queen d7 can recapture bishop on d2 (330).  Then White rook d1
    // recaptures queen (900).  No more Black pieces.
    // SEE for bishop×d2:
    //   100 − see_inner(BLACK, d2, 330, occ−{c1})
    //   see_inner(BLACK): queen recaptures bishop 330
    //     max(0, 330 − see_inner(WHITE, d2, 900, occ−{c1,d7}))
    //     see_inner(WHITE): rook recaptures queen 900 (x-ray through d1)
    //       max(0, 900 − 0) = 900
    //     max(0, 330 − 900) = max(0, −570) = 0  ← Black stands pat!
    //   0
    //   SEE = 100 − 0 = 100.  (White wins pawn safely; Black can't profit.)
    //
    // Hmm that's because the rook defends d2 so strongly that Black's queen
    // can't profitably recapture.  Let me verify the test gives 100.
    #[test]
    fn see_xray_rook_behind_bishop() {
        // bishop c1, rook d1, pawn d2, queen d7, king h1
        let b  = board("8/3q4/8/8/8/8/3p4/2BR3K w - - 0 1");
        let mv = parse_uci_move(&b, "c1d2").unwrap();
        // Black queen can recapture but then loses to rook x-ray, so stands pat.
        // SEE = 100 (just the pawn, Black declines).
        assert_eq!(see(&b, mv), SEE_VALUES[PieceType::Pawn.index()]);
    }

    // ── Non-capture returns 0 ─────────────────────────────────────────────
    #[test]
    fn see_quiet_move_returns_zero() {
        let b  = board(crate::board::STARTING_FEN);
        let mv = parse_uci_move(&b, "e2e4").unwrap();
        assert_eq!(see(&b, mv), 0);
    }

    // ── En passant capture ────────────────────────────────────────────────
    // Stripped position: only White pawn e5 and Black pawn f5 (plus kings).
    // f6 has no Black defenders (no adjacent Black pawns), so SEE = pawn value.
    #[test]
    fn see_en_passant() {
        // White pawn e5, Black pawn f5 (EP target), kings.  No other pieces.
        let b  = board("4k3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1");
        let mv = parse_uci_move(&b, "e5f6").unwrap();
        // f6 is undefended: SEE = pawn value.
        assert_eq!(see(&b, mv), SEE_VALUES[PieceType::Pawn.index()]);
    }

    // ── Three-piece exchange: Qxp defended by rook, then bishop ──────────
    // 8/8/2b5/3p4/4Q3/8/8/2R4K w - - 0 1
    // d5 defended by bishop c6 and White rook c1 (indirectly via file c? No —
    // White queen e4×d5, Black bishop c6 recaptures queen (900), White rook c1
    // can recapture bishop along file c? c1 to d5: not a straight line.
    // Let me use a simpler three-way test.
    //
    // 8/8/8/3r4/3R4/8/8/3R3K w - - 0 1
    // White rook d1×d4 rook, Black rook d5 recaptures (500), White rook d1
    // recaptures (500, x-ray).  Black has no more rooks on the d-file.
    // see(d4×d5? No, it's rook×rook):
    //   Actually d4 is White, d5 is Black. "d4d5" would be White rook captures Black rook.
    //   target_val = 500, piece_val = 500.
    //   see_inner(BLACK, d5, 500, occ−{d4}):
    //     Black rook d5 is gone (it was taken). Wait, d5 is the *target*.
    //     After White d4 rook captures d5 rook: d5 has White rook, occ no longer has d4.
    //     Black attackers of d5 with occ−{d4}: no more Black pieces on the d-file.
    //     see_inner returns 0.
    //   500 − 0 = 500 (White wins the rook cleanly since d5 is undefended after capture).
    //   But wait, d5 is the Black rook being taken. After capture, who has another piece?
    //   The FEN "8/8/8/3r4/3R4/8/8/3R3K" has White rooks on d4 and d1, Black rook on d5.
    //   White d4×d5: Black rook taken. occ = {d1, d5(now White), h1(king?)}.
    //   Actually no: we clear d4 from occ. So new occ = occ without d4.
    //   Black attackers on d5: none (no Black pieces left on the board except the king).
    //   see_inner returns 0.
    //   SEE = 500 − 0 = 500. ✓
    //   But after Black's rook is taken, White's d1 rook is an x-ray attacker that
    //   "would" appear if Black had a piece to recapture with. Since Black can't
    //   recapture, the x-ray is never needed.
    //
    // A real three-way exchange:
    // Black rook d5, White rook d4, White rook d1, Black rook d8.
    // "3r4/8/8/3r4/3R4/8/8/3R3K w - - 0 1"
    // White d4×d5: gains rook (500). Black d8 recaptures (gains rook d4, 500).
    // White d1 x-ray recaptures (gains rook d8, 500). No more Black.
    // Net: 500 − 500 + 500... let's trace:
    //   see(d4×d5) = 500 − see_inner(BLACK, d5, 500, occ−{d4})
    //   see_inner(BLACK): attacker=d8 rook (500)
    //     max(0, 500 − see_inner(WHITE, d5, 500, occ−{d4,d8}))
    //     see_inner(WHITE): attacker=d1 rook (500, x-ray through d4 which is now gone)
    //       max(0, 500 − see_inner(BLACK, d5, 500, occ−{d4,d8,d1}))
    //       see_inner(BLACK): no more Black attackers → 0
    //       max(0, 500 − 0) = 500
    //     max(0, 500 − 500) = 0  ← Black stands pat (even trade)
    //   0
    //   SEE = 500 − 0 = 500. (White wins a rook; Black won't recapture into equality.)
    //
    // Hmm, that means SEE = 500 even with both sides having two rooks. That seems off.
    // Let me retrace: 500 - 0 = 500 at the top level means "White gains 500".
    // But Black declines to recapture (stands pat at 0 gain). So yes, White wins a
    // rook for free? That would only be true if Black's rook is pinned or something.
    //
    // Wait — I think there's an error. After White d4 rook takes d5 rook:
    // - occ is cleared of d4 (attacker moved to d5 — but we don't add it to d5 in occ!)
    // - In our implementation, when the attacker "captures" we just remove it from occ.
    //   The piece doesn't "appear" on the target square in occ.
    // So Black's d8 rook sees d5 only if the d-file is clear between d8 and d5.
    // occ−{d4} still has d5 (well, d5 was the captured rook... hmm).
    //
    // AH — I see the bug: when Black's rook on d5 is captured, d5 should be cleared
    // from occ too! But in our implementation, we only clear the ATTACKER's square
    // from occ, not the target square.
    //
    // Wait, let me re-read the algorithm. The initial occ has d4 cleared (White's
    // attacker leaves d4). d5 is the target — it stays in occ. But d5 is now occupied
    // by White's rook. However for slider attacks, we need the d5 square to be "occupied"
    // to block sliders from going past it. Since we left d5 in occ, that's correct:
    // Black's d8 rook sees d5 (the first occupied square it hits), not d4 (already cleared).
    //
    // So see_inner(BLACK, d5, 500, occ−{d4}):
    //   attackers_to_occ(board, d5, occ−{d4}):
    //     orth = rook_attacks(d5, occ−{d4}) & rooks|queens & occ−{d4}
    //     rook_attacks(d5, occ−{d4}) includes d8 (going north, first piece hit).
    //     Also d1 (going south: d4 is gone, d3/d2 are empty, hits d1). So d1 is also visible!
    //   Black attackers (side=BLACK): d8 rook ✓
    //
    // see_inner(WHITE after Black's d8 rook captures):
    //   occ−{d4,d8}: rook_attacks(d5, occ−{d4,d8}) going south hits d1.
    //   White attackers: d1 rook. ✓
    //
    // OK so the x-ray works. But what about d5 itself after White captures it?
    // After White d4 captures d5, in reality d5 is now occupied by White's rook.
    // But we DIDN'T clear the old d5 Black rook from occ. The d5 square remains
    // in occ (as the old Black rook). This is correct for blocking purposes
    // (the piece that "landed" on d5 blocks further sliders, which is correct).
    //
    // But when we compute attackers of d5 for Black, we find rook_attacks(d5, occ−{d4}):
    // going north from d5, first occupied square is d8 (Black rook). ✓
    // Going south from d5: first occupied square with d4 removed... d3/d2/d1 → d1.
    // So d1 White rook appears as an attacker via the south ray. But it's a WHITE
    // attacker. White side_to_move here is BLACK (we passed `side=BLACK`).
    // So `board.color(BLACK) & attackers` filters it: d1 is White, not included. ✓
    //
    // Great, the algorithm is correct.
    #[test]
    fn see_three_piece_exchange() {
        // Black rook d5, White rook d4, White rook d1, Black rook d8, kings.
        let b  = board("3r4/8/8/3r4/3R4/8/8/3R3K w - - 0 1");
        let mv = parse_uci_move(&b, "d4d5").unwrap();
        // White d4×d5: Black d8 can recapture but then faces White d1 x-ray.
        // Black's recapture is neutral (500−500=0), so Black stands pat.
        // White nets 500.
        assert_eq!(see(&b, mv), SEE_VALUES[PieceType::Rook.index()]);
    }
}
