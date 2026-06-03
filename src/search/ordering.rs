//! Move ordering.
//!
//! Alpha-beta prunes far more when the *best* move is tried first: a quick beta
//! cutoff means the rest of the moves are never searched. We can't know the best
//! move for sure, but we can make excellent guesses and search them first. This
//! is the single biggest "free" speedup in a chess engine — same result, often
//! an order of magnitude fewer nodes.
//!
//! We score each move and search them highest-first, in this priority order:
//!
//! 1. **Principal-variation / hash move** — the best move from the previous
//!    iteration (or, later, the transposition table). Far and away the most
//!    likely to be best again.
//! 2. **Captures**, ranked by **MVV-LVA** (Most Valuable Victim, Least Valuable
//!    Attacker): grabbing a queen with a pawn is tried before a queen takes a
//!    pawn, because the former is more likely to win material.
//! 3. **Killer moves** — quiet moves that caused a beta cutoff at the *same ply*
//!    in a sibling line. They tend to refute many positions at that depth.
//! 4. **History heuristic** — quiet moves that have historically caused cutoffs
//!    anywhere, weighted by the depth at which they did. A soft, learned ordering
//!    of the remaining quiet moves.

use crate::board::{Board, Color, PieceType};
use crate::movegen::Move;
use crate::search::see::{see, SEE_VALUES};

/// Maximum ply we track killer moves for (also bounds quiescence recursion).
pub const MAX_PLY: usize = 128;

// Score bands (searched highest-first).  Gaps prevent category overlap.
const SCORE_TT: i32 = 2_000_000;
/// Winning / neutral captures (SEE >= 0): searched before quiet moves.
pub const SCORE_WIN_CAP: i32 = 1_000_000;
const SCORE_KILLER_1: i32 = 900_000;
const SCORE_KILLER_2: i32 = 800_000;
/// Counter-move: the quiet move that most recently refuted the predecessor move.
pub(super) const SCORE_COUNTER: i32 = 700_000;
/// History + continuation scores are clamped below the counter-move band.
pub const HISTORY_MAX: i32 = SCORE_COUNTER - 1;
/// Losing captures (SEE < 0): searched after quiet moves.
const SCORE_LOSE_CAP: i32 = -1_000_000;

// Keep the old name as an alias so external tests don't break.
#[allow(dead_code)]
pub const SCORE_CAPTURE: i32 = SCORE_WIN_CAP;

/// Shared gravity-based history update.  Works for both positive (bonus) and
/// negative (malus) values: the entry is pulled toward `+max` on bonus and
/// `–max` on malus, saturating at ±max.
#[inline]
pub(super) fn history_update(entry: &mut i32, bonus: i32) {
    *entry += bonus - *entry * bonus.abs() / HISTORY_MAX;
    *entry = (*entry).clamp(-HISTORY_MAX, HISTORY_MAX);
}

/// Quiescence ordering.
///
/// Captures with non-negative SEE score get a positive score (searched first,
/// ranked by SEE value).  SEE-negative captures get a negative score so the
/// quiescence loop can skip them cheaply once it hits the first negative score.
/// Non-capture, non-promotion moves sort last (score –1).
pub fn capture_score(board: &Board, mv: Move) -> i32 {
    if mv.is_capture() {
        let s = see(board, mv);
        if let Some(pt) = mv.promotion_piece() {
            // capture + promotion: add the promotion bonus to ensure it sorts high
            s + SEE_VALUES[pt.index()]
        } else {
            s
        }
    } else if let Some(pt) = mv.promotion_piece() {
        // Non-capturing promotion: always good — use promoted piece value as score.
        SEE_VALUES[pt.index()]
    } else {
        -1
    }
}

/// Full move ordering for the main search.
///
/// Priority order (highest score searched first):
/// 1. TT / hash move
/// 2. Winning or neutral captures (SEE ≥ 0), ranked by SEE value
/// 3. Non-capturing promotions (always gain material)
/// 4. Killer moves (quiet moves with prior beta-cutoffs at this ply)
/// 5. Counter-move (quiet move that refuted the predecessor — supplied by caller)
/// 6. History heuristic, including continuation history bonus added by caller
/// 7. Losing captures (SEE < 0), ranked by SEE value (least-bad first)
///
/// `counter_move` is `Some(mv)` when `mv` is the stored counter to the
/// predecessor move; pass `None` if unavailable (root, null-move subtrees).
pub fn score_move(
    board: &Board,
    mv: Move,
    ply: usize,
    tt_move: Option<Move>,
    killers: &Killers,
    history: &History,
    counter_move: Option<Move>,
) -> i32 {
    if Some(mv) == tt_move {
        return SCORE_TT;
    }
    if mv.is_capture() {
        let see_val = see(board, mv);
        return if see_val >= 0 {
            SCORE_WIN_CAP + see_val
        } else {
            SCORE_LOSE_CAP + see_val
        };
    }
    if let Some(pt) = mv.promotion_piece() {
        // Non-capturing promotion: treat as a winning tactical move.
        return SCORE_WIN_CAP + SEE_VALUES[pt.index()];
    }
    // Quiet move: killers → counter-move → history.
    if killers.first(ply) == Some(mv) {
        return SCORE_KILLER_1;
    }
    if killers.second(ply) == Some(mv) {
        return SCORE_KILLER_2;
    }
    if Some(mv) == counter_move {
        return SCORE_COUNTER;
    }
    // Base history score; continuation bonus is added by the caller.
    history.get(board.side_to_move, mv).clamp(-HISTORY_MAX, HISTORY_MAX)
}

/// Two "killer" quiet moves per ply.
pub struct Killers {
    table: [[Option<Move>; 2]; MAX_PLY],
}

impl Killers {
    pub fn new() -> Killers {
        Killers {
            table: [[None; 2]; MAX_PLY],
        }
    }

    /// Record `mv` as the most recent killer at `ply`, shifting the previous one
    /// into the second slot.
    pub fn add(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_PLY || self.table[ply][0] == Some(mv) {
            return;
        }
        self.table[ply][1] = self.table[ply][0];
        self.table[ply][0] = Some(mv);
    }

    #[inline]
    pub fn first(&self, ply: usize) -> Option<Move> {
        if ply < MAX_PLY {
            self.table[ply][0]
        } else {
            None
        }
    }

    #[inline]
    pub fn second(&self, ply: usize) -> Option<Move> {
        if ply < MAX_PLY {
            self.table[ply][1]
        } else {
            None
        }
    }
}

/// History heuristic: a `[side][from][to]` table of how often a quiet move has
/// produced a beta cutoff, weighted by the depth at which it did.
pub struct History {
    // Boxed so the 32 KB table lives on the heap, not in every `Searcher` move.
    table: Box<[[[i32; 64]; 64]; 2]>,
}

impl History {
    pub fn new() -> History {
        History {
            table: Box::new([[[0; 64]; 64]; 2]),
        }
    }

    /// Reward a quiet move that caused a cutoff. Uses gravity so the table
    /// never saturates: entries approach `HISTORY_MAX` asymptotically.
    pub fn add(&mut self, side: Color, mv: Move, depth: u32) {
        let bonus = (depth as i32).min(13) * (depth as i32).min(13);
        let e = &mut self.table[side.index()][mv.from().index()][mv.to().index()];
        history_update(e, bonus);
    }

    /// Penalise a quiet move that was searched but did NOT cause a cutoff.
    /// Pulls the entry toward `–HISTORY_MAX` using the same gravity formula.
    pub fn add_malus(&mut self, side: Color, mv: Move, depth: u32) {
        let malus = (depth as i32).min(13) * (depth as i32).min(13);
        let e = &mut self.table[side.index()][mv.from().index()][mv.to().index()];
        history_update(e, -malus);
    }

    /// Halve all history entries.  Call once per iterative-deepening iteration
    /// to prevent scores from prior depths from permanently dominating.
    pub fn decay(&mut self) {
        for side_table in self.table.iter_mut() {
            for row in side_table.iter_mut() {
                for e in row.iter_mut() {
                    *e /= 2;
                }
            }
        }
    }

    #[inline]
    pub fn get(&self, side: Color, mv: Move) -> i32 {
        self.table[side.index()][mv.from().index()][mv.to().index()]
    }
}

// ── Continuation history ───────────────────────────────────────────────────

/// Per-ply continuation history: how well does `(curr_piece → curr_to)` work
/// when the predecessor move was `(prev_piece → prev_to)`?
///
/// Indexed as `[prev_piece_type][prev_to][curr_piece_type][curr_to]`.
/// Stored in a flat `Vec` to keep the 576 KB off the stack.
/// Two instances in `Searcher` cover the 1-ply and 2-ply lookbacks.
pub struct ContinuationHistory {
    table: Vec<i32>, // length = 6 × 64 × 6 × 64 = 147,456
}

impl ContinuationHistory {
    pub fn new() -> Self {
        ContinuationHistory {
            table: vec![0; 6 * 64 * 6 * 64],
        }
    }

    #[inline]
    fn idx(prev_pt: usize, prev_to: usize, curr_pt: usize, curr_to: usize) -> usize {
        ((prev_pt * 64 + prev_to) * 6 + curr_pt) * 64 + curr_to
    }

    #[inline]
    pub fn get(&self, prev_pt: usize, prev_to: usize, curr_pt: usize, curr_to: usize) -> i32 {
        self.table[Self::idx(prev_pt, prev_to, curr_pt, curr_to)]
    }

    /// Gravity-based update.  `bonus` is positive for a reward (cutoff) and
    /// negative for a malus (searched but failed).
    pub fn update(&mut self, prev_pt: usize, prev_to: usize, curr_pt: usize, curr_to: usize, bonus: i32) {
        let idx = Self::idx(prev_pt, prev_to, curr_pt, curr_to);
        history_update(&mut self.table[idx], bonus);
    }

    /// Halve all entries — called once per iterative-deepening depth.
    pub fn decay(&mut self) {
        for e in self.table.iter_mut() {
            *e /= 2;
        }
    }
}

// ── Counter-move table ─────────────────────────────────────────────────────

/// For each (from, to) pair of the predecessor move, stores the quiet move
/// that most recently caused a beta cutoff at that position.
pub struct CounterMoves {
    // [from][to] → most recent quiet counter-move.
    table: Box<[[Option<Move>; 64]; 64]>,
}

impl CounterMoves {
    pub fn new() -> Self {
        CounterMoves {
            table: Box::new([[None; 64]; 64]),
        }
    }

    /// Returns the stored counter-move for `prev_mv`, or `None`.
    #[inline]
    pub fn get(&self, prev_mv: Option<Move>) -> Option<Move> {
        prev_mv.and_then(|m| self.table[m.from().index()][m.to().index()])
    }

    /// Record `response` as the counter to `prev_mv`.
    pub fn set(&mut self, prev_mv: Move, response: Move) {
        self.table[prev_mv.from().index()][prev_mv.to().index()] = Some(response);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    // White pawn c4×d5 queen (undefended): SEE = 900 (winning capture).
    // White queen d1×d5 queen (undefended): SEE = 900 (same value, different attacker).
    // Both should land in the win-cap band with identical scores.
    #[test]
    fn winning_capture_ranked_by_see() {
        // White pawn c4, White queen d1, Black queen d5 (undefended), kings.
        let board = Board::from_fen("8/8/8/3q4/2P5/8/8/3Q3K w - - 0 1").unwrap();
        let killers = Killers::new();
        let history = History::new();
        let pxq = crate::uci::parser::parse_uci_move(&board, "c4d5").unwrap();
        let qxq = crate::uci::parser::parse_uci_move(&board, "d1d5").unwrap();
        let pxq_s = score_move(&board, pxq, 0, None, &killers, &history, None);
        let qxq_s = score_move(&board, qxq, 0, None, &killers, &history, None);
        // d5 is undefended: SEE = 900 for both.
        assert_eq!(pxq_s, qxq_s, "equal SEE → equal score");
        assert!(pxq_s >= SCORE_WIN_CAP, "winning capture must be in the win-cap band");
    }

    #[test]
    fn losing_capture_goes_below_quiets() {
        // 8/8/2b5/3p4/4Q3/8/8/7K w - - 0 1
        // White queen e4×d5 pawn, defended by Black bishop c6.  SEE = –800.
        let board = Board::from_fen("8/8/2b5/3p4/4Q3/8/8/7K w - - 0 1").unwrap();
        let killers = Killers::new();
        let history = History::new();
        let losing_cap = crate::uci::parser::parse_uci_move(&board, "e4d5").unwrap();
        let cap_score = score_move(&board, losing_cap, 0, None, &killers, &history, None);
        // Losing capture must score below killer band (below 0).
        assert!(cap_score < 0, "losing capture must sort after quiet moves, got {cap_score}");
    }

    #[test]
    fn tt_move_ranks_above_captures() {
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
            .unwrap();
        let killers = Killers::new();
        let history = History::new();
        let capture = crate::uci::parser::parse_uci_move(&board, "e4d5").unwrap();
        let quiet = crate::uci::parser::parse_uci_move(&board, "g1f3").unwrap();

        let cap_score = score_move(&board, capture, 0, None, &killers, &history, None);
        let tt_score = score_move(&board, quiet, 0, Some(quiet), &killers, &history, None);
        assert!(tt_score > cap_score, "the hash move must be tried first");
        // e4xd5 is defended by the Black queen → SEE = 0 → SCORE_WIN_CAP + 0
        assert!(cap_score >= SCORE_WIN_CAP, "neutral capture sits in the win-cap band");
    }

    #[test]
    fn killers_then_history_for_quiets() {
        let board = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let mut killers = Killers::new();
        let mut history = History::new();
        let killer = crate::uci::parser::parse_uci_move(&board, "e2e4").unwrap();
        let other = crate::uci::parser::parse_uci_move(&board, "d2d4").unwrap();

        killers.add(0, killer);
        history.add(Color::White, other, 5);

        let ks = score_move(&board, killer, 0, None, &killers, &history, None);
        let hs = score_move(&board, other, 0, None, &killers, &history, None);
        assert!(ks > hs, "a killer outranks a history move");
        assert_eq!(ks, SCORE_KILLER_1);
    }

    #[test]
    fn counter_move_ranks_between_killers_and_history() {
        let board = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let killers = Killers::new();
        let history = History::new();
        let e4 = crate::uci::parser::parse_uci_move(&board, "e2e4").unwrap();
        let d4 = crate::uci::parser::parse_uci_move(&board, "d2d4").unwrap();

        let counter = score_move(&board, e4, 0, None, &killers, &history, Some(e4));
        let plain   = score_move(&board, d4, 0, None, &killers, &history, None);

        assert_eq!(counter, SCORE_COUNTER, "counter-move must land in the counter band");
        assert!(counter > plain, "counter-move ranks above plain history");
        assert!(counter < SCORE_KILLER_2, "counter-move ranks below killer-2");
    }

    #[test]
    fn history_add_uses_gravity() {
        let mut history = History::new();
        // After many large-depth rewards the entry should saturate near HISTORY_MAX.
        let board = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let e4 = crate::uci::parser::parse_uci_move(&board, "e2e4").unwrap();
        for _ in 0..1000 {
            history.add(crate::board::Color::White, e4, 10);
        }
        let val = history.get(crate::board::Color::White, e4);
        assert!(val <= HISTORY_MAX, "history must stay <= HISTORY_MAX, got {val}");
        assert!(val > 0, "should have positive history after rewards");
    }

    #[test]
    fn history_malus_reduces_score() {
        let mut history = History::new();
        let board = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let e4 = crate::uci::parser::parse_uci_move(&board, "e2e4").unwrap();
        // Build up a moderate score first.
        for _ in 0..20 {
            history.add(crate::board::Color::White, e4, 5);
        }
        let before = history.get(crate::board::Color::White, e4);
        history.add_malus(crate::board::Color::White, e4, 5);
        let after = history.get(crate::board::Color::White, e4);
        assert!(after < before, "malus must reduce the history score");
    }

    #[test]
    fn history_decay_halves_values() {
        let mut history = History::new();
        let board = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let e4 = crate::uci::parser::parse_uci_move(&board, "e2e4").unwrap();
        history.add(crate::board::Color::White, e4, 10);
        let before = history.get(crate::board::Color::White, e4);
        history.decay();
        let after = history.get(crate::board::Color::White, e4);
        assert_eq!(after, before / 2, "decay must halve the history score");
    }

    #[test]
    fn continuation_history_stores_and_retrieves() {
        let mut ch = ContinuationHistory::new();
        ch.update(0, 10, 2, 30, 100);
        assert_eq!(ch.get(0, 10, 2, 30), 100);
        assert_eq!(ch.get(0, 10, 2, 31), 0, "unset entries should be zero");
    }

    #[test]
    fn continuation_history_gravity_bounds() {
        let mut ch = ContinuationHistory::new();
        for _ in 0..1000 {
            ch.update(0, 0, 0, 0, 169); // depth=13 → bonus=169
        }
        assert!(ch.get(0, 0, 0, 0) <= HISTORY_MAX, "must not exceed HISTORY_MAX");
        for _ in 0..1000 {
            ch.update(0, 0, 0, 0, -169);
        }
        assert!(ch.get(0, 0, 0, 0) >= -HISTORY_MAX, "must not go below -HISTORY_MAX");
    }

    #[test]
    fn counter_moves_get_returns_none_when_empty() {
        let cm = CounterMoves::new();
        assert_eq!(cm.get(None), None);
    }

    #[test]
    fn counter_moves_set_and_get_round_trip() {
        let mut cm = CounterMoves::new();
        let board = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let e4 = crate::uci::parser::parse_uci_move(&board, "e2e4").unwrap();
        let d4 = crate::uci::parser::parse_uci_move(&board, "d2d4").unwrap();
        cm.set(e4, d4);
        assert_eq!(cm.get(Some(e4)), Some(d4));
        assert_eq!(cm.get(Some(d4)), None, "unset key should return None");
    }
}
