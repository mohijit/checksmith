//! Negamax with alpha-beta, PVS, NMP, LMR, and a transposition table.
//!
//! ## Negamax
//!
//! Minimax collapses into `max over moves of -value(child)` because
//! [`evaluate`](crate::eval::evaluate) already returns a score from the
//! *side-to-move's* perspective.
//!
//! ## Alpha-beta + PVS
//!
//! A `[alpha, beta]` window prunes branches the opponent would never allow.
//! PVS (Principal Variation Search) verifies non-first moves cheaply with a
//! null window, re-searching only the rare move that beats `alpha`.
//!
//! ## Null Move Pruning (NMP)
//!
//! The side to move "passes" (plays no move). If the resulting position still
//! exceeds beta after a reduced-depth search, the position is so good that
//! even doing nothing wins — so we prune the branch immediately.
//!
//! **Why it works:** Chess is not a zugzwang game in the middlegame; having the
//! move is almost always an advantage. If you're already far ahead and giving
//! up your move still beats beta, any real move will too.
//!
//! **When it is unsafe:**
//! * When the side to move is in check (must make a legal move).
//! * When only pawns and kings remain (zugzwang is possible — the move can be
//!   a liability). We guard this with [`has_non_pawn_material`].
//! * When the parent call was also a null move (consecutive passes give trivially
//!   optimistic scores). We guard this with the `skip_null` flag.
//!
//! **Parameters:**
//! ```text
//! R = 3 + depth/3   (adaptive reduction — larger at higher depths)
//! null_depth = depth.saturating_sub(R + 1)
//! ```
//! A null move at depth 3 uses quiescence for verification (depth 0); at
//! depth 7+ the verification is a real reduced-depth search.
//!
//! ## Late Move Reductions (LMR)
//!
//! Moves ordered late in the list are almost certainly bad. Instead of
//! searching them at full depth, we try them at a reduced depth first. If the
//! reduced search beats `alpha` (the move is surprisingly good), we re-search
//! at full depth to get an accurate score.
//!
//! **Why it works:** Move ordering (TT move first, then captures by MVV-LVA,
//! then killers, then history) puts the best move first with high probability.
//! The 3rd or 10th or 20th move failing to beat alpha at reduced depth is almost
//! always correct; we waste very little accuracy.
//!
//! **Reduction formula:**
//! ```text
//! R = floor(ln(depth) × ln(move_index + 1) / 2)
//!   clamped to [1, depth-1]
//! ```
//! This gives R=1 for small depth/index values, growing logarithmically as
//! searches get deeper and move lists get longer.
//!
//! **Conditions that disable LMR for a move:**
//! * The position is in check.
//! * The move is a capture or promotion (tactical).
//! * The move is a killer (proved good at the same ply in another line).
//! * `depth < 3` or `move_index < 2` (too shallow to be meaningful).
//!
//! ## Razoring
//!
//! At depth 1, if the static evaluation plus a margin is still below alpha,
//! the position is unlikely to raise alpha — fall straight through to
//! quiescence rather than generating and searching all quiet moves.
//!
//! ## Late Move Pruning (LMP)
//!
//! At low depths, after searching the first N quiet moves, the remaining ones
//! are very unlikely to raise alpha and can be skipped entirely.  Unlike LMR
//! (which reduces depth), LMP discards moves outright.  Only applied at
//! non-PV nodes and when not in check.
//!
//! ## ProbCut
//!
//! Before the main move loop, try each capture whose SEE exceeds a raised
//! threshold (beta + PROBCUT_MARGIN) at a sharply reduced depth.  If any
//! scores above that threshold, the full-depth search would too — prune
//! the node immediately.  This is especially powerful at depths ≥ 5 where
//! it replaces many expensive sub-trees with a cheap scout.
//!
//! ## Singular Extensions
//!
//! Before searching the TT move at a node, verify whether it is the *only*
//! good move: search all other moves at half depth with a narrow window
//! centred below `tt_score − margin`.  If they all fail low, the TT move
//! is "singular" and earns one extra ply of search depth.  This prevents
//! the horizon from cutting off forced sequences prematurely.
//!
//! **Conditions:** `depth ≥ 6`, TT has a lower/exact bound at `depth − 3` or
//! more, the stored score is not a mate score, and we are not already inside
//! an SE verification call.
//!
//! ## Transposition table
//!
//! Every node probes the [`TranspositionTable`]: a deep-enough stored result
//! returns immediately (TT cutoff), and the stored best move is ordered first.
//! Mate scores are converted to be node-relative on store/load.

use super::draw;
use super::ordering::{
    score_move, ContinuationHistory, CounterMoves, History, Killers,
    HISTORY_MAX, MAX_PLY, SCORE_COUNTER,
};
use super::params;
use super::see::see;
use super::tt::{Bound, TranspositionTable};
use super::{INFINITY, MATE, MATE_IN_MAX};
use crate::board::{Board, Color, PieceType};
use crate::eval::evaluator::{Evaluator, HandcraftedEvaluator};
use crate::movegen::Move;
use crate::tablebase;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// --- LMR tuning constants -----------------------------------------------

/// Minimum depth at which LMR is applied.
const LMR_MIN_DEPTH: u32 = 3;

/// First move index eligible for reduction (0-indexed; index 0 = first move).
const LMR_MOVE_THRESHOLD: usize = 2;

// --- NMP tuning constants -----------------------------------------------

/// Minimum depth at which NMP is attempted.
const NMP_MIN_DEPTH: u32 = 3;

// --- Futility pruning constants -----------------------------------------

// --- Futility pruning constants (runtime-configurable via params module) ---
// FUTILITY_MARGIN_1, FUTILITY_MARGIN_2 → params::futility_margin_{1,2}()
// RAZOR_MARGIN                          → params::razor_margin()
// LMP_BASE                              → params::lmp_base()
// PROBCUT_MARGIN                        → params::probcut_margin()
// SE_DEPTH_FACTOR                       → params::se_depth_factor()

// --- Depth thresholds (compile-time; too coarse for gradient-based tuning) ---

/// Maximum depth at which LMP is applied (inclusive).
const LMP_MAX_DEPTH: u32 = 5;

/// Minimum depth to attempt ProbCut.
const PROBCUT_MIN_DEPTH: u32 = 5;

/// Depth reduction for the ProbCut verification search.
const PROBCUT_REDUCTION: u32 = 4;

/// Minimum depth at which a singular extension is considered.
const SE_MIN_DEPTH: u32 = 6;

// -------------------------------------------------------------------------

/// The outcome of a search.
pub struct SearchResult {
    /// Best move found, or `None` if the position is terminal (mate/stalemate).
    pub best_move: Option<Move>,
    /// Score in centipawns from the side-to-move's perspective.
    pub score: i32,
    /// Depth actually completed.
    pub depth: u32,
    /// Nodes visited.
    pub nodes: u64,
}

/// Drives one search: node count, killer/history tables, and stop conditions.
///
/// The generic parameter `E` is the static evaluator used at leaf nodes.
/// The default is [`HandcraftedEvaluator`]; pass a different type implementing
/// [`Evaluator`] to swap the evaluation function without touching the search.
pub struct Searcher<E: Evaluator = HandcraftedEvaluator> {
    pub nodes: u64,
    /// Set once a stop condition trips; the current search unwinds.
    pub stopped: bool,
    /// Maximum ply reached in the current search (including quiescence).
    /// Reset to 0 before each iterative-deepening iteration in `think()`.
    pub max_ply: i32,

    // ── Diagnostic counters ───────────────────────────────────────────────
    // These are zero-cost in release builds (simple increments) and invaluable
    // for debugging and tuning. Read them after `search_root` completes.

    /// Beta cutoffs (fail-high nodes): the move was so good we pruned the rest.
    /// Healthy engines see ~55–70% of interior nodes fail high.
    pub fail_highs: u64,
    /// LMR + PVS re-searches: extra negamax calls triggered when a reduced or
    /// null-window search beat alpha. High rates indicate active move ordering.
    pub researches: u64,
    /// Successful `tt.probe()` calls (move/score retrieved from table).
    pub tt_hits: u64,
    /// TT hits that caused an immediate return (score matched the window).
    pub tt_cutoffs: u64,
    /// Nodes where the position was in check and depth was extended by 1.
    pub check_extensions: u64,
    /// TT moves verified singular and extended by 1 ply.
    pub singular_extensions: u64,
    /// Successful WDL tablebase probes that replaced the static evaluator.
    pub tb_hits: u64,

    pub(crate) evaluator: E,
    killers: Killers,
    pub history: History,
    /// Move played at each ply; `None` for null moves.  Used to look up the
    /// continuation history for the 1-ply and 2-ply ancestors of any node.
    move_stack: [Option<Move>; MAX_PLY],
    /// Continuation history conditioned on the move at `ply - 1`.
    pub cont_hist_1: ContinuationHistory,
    /// Continuation history conditioned on the move at `ply - 2`.
    pub cont_hist_2: ContinuationHistory,
    /// Stores the quiet move that most recently refuted each predecessor move.
    counter_moves: CounterMoves,
    /// Per-ply move excluded from the move loop during an SE verification search.
    /// Set to `Some(mv)` just before the recursive SE call; cleared afterward.
    /// Also used as a re-entry guard: SE is not attempted when this is `Some`.
    se_excluded: [Option<Move>; MAX_PLY],
    deadline: Option<Instant>,
    stop: Arc<AtomicBool>,
    node_limit: Option<u64>,
    /// Zobrist hashes of all positions visited so far (game history + search path).
    ///
    /// Pre-populated with the game history before the search root.  As the
    /// search makes moves the current node's hash is pushed before recursing
    /// and popped after returning — giving each recursive call a view of the
    /// full path from the game start to the current position.
    ///
    /// Used for threefold-repetition detection: if the current position hash
    /// appears anywhere in this slice, the position is (at least) a 2-fold
    /// repetition and we return 0.
    hash_history: Vec<u64>,
}

impl<E: Evaluator> Searcher<E> {
    pub fn new(
        deadline: Option<Instant>,
        stop: Arc<AtomicBool>,
        node_limit: Option<u64>,
        evaluator: E,
    ) -> Self {
        Searcher::new_with_history(&[], deadline, stop, node_limit, evaluator)
    }

    /// Construct a `Searcher` pre-loaded with the position history from the
    /// game so far.  The history must contain the hashes of all positions
    /// *before* the root — not including the root position itself.
    pub fn new_with_history(
        game_history: &[u64],
        deadline: Option<Instant>,
        stop: Arc<AtomicBool>,
        node_limit: Option<u64>,
        evaluator: E,
    ) -> Self {
        Searcher {
            nodes: 0,
            stopped: false,
            max_ply: 0,
            fail_highs: 0,
            researches: 0,
            tt_hits: 0,
            tt_cutoffs: 0,
            check_extensions: 0,
            singular_extensions: 0,
            tb_hits: 0,
            evaluator,
            killers: Killers::new(),
            history: History::new(),
            move_stack: [None; MAX_PLY],
            cont_hist_1: ContinuationHistory::new(),
            cont_hist_2: ContinuationHistory::new(),
            counter_moves: CounterMoves::new(),
            se_excluded: [None; MAX_PLY],
            deadline,
            stop,
            node_limit,
            hash_history: game_history.to_vec(),
        }
    }

    // ── Repetition detection ─────────────────────────────────────────────────

    /// True if `hash` has appeared before in the search path or game history.
    ///
    /// We only scan back `halfmove_clock` entries because an irreversible move
    /// (pawn push or capture) resets the clock and makes repetition impossible
    /// beyond that point.
    #[inline]
    fn is_repetition(&self, hash: u64, halfmove_clock: u16) -> bool {
        let n = self.hash_history.len();
        let max_back = halfmove_clock as usize;
        let start = n.saturating_sub(max_back);
        draw::hash_in_slice(hash, &self.hash_history[start..])
    }

    /// True if the search should stop now. Latches [`stopped`](Searcher::stopped).
    pub(crate) fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.stop.load(Ordering::Relaxed) {
            self.stopped = true;
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            self.stopped = true;
        } else if self.node_limit.is_some_and(|n| self.nodes >= n) {
            self.stopped = true;
        }
        self.stopped
    }

    /// Search the root and return the best move and its score, abortable mid-way.
    ///
    /// `pv_move` is the best move from the previous iterative-deepening pass,
    /// used to order the search if the TT doesn't already supply one.
    ///
    /// `excluded` lists moves to skip at the root — used for MultiPV (each
    /// successive PV line runs with the previously found best moves excluded).
    /// `alpha` / `beta` define the search window.  Pass `(-INFINITY, INFINITY)`
    /// for a full-window search; the aspiration loop in `think()` narrows this.
    pub fn search_root(
        &mut self,
        board: &mut Board,
        depth: u32,
        pv_move: Option<Move>,
        tt: &TranspositionTable,
        excluded: &[Move],
        alpha: i32,
        beta: i32,
    ) -> (Option<Move>, i32) {
        let mut moves = board.legal_moves();
        if moves.is_empty() {
            let score = if board.is_in_check() { -MATE } else { 0 };
            return (None, score);
        }

        // Order the previously-best move (TT or PV) first.
        let tt_move = tt.probe(board.hash).and_then(|d| d.best_move()).or(pv_move);

        let n = moves.len();
        let mut scores = [0i32; 256];
        {
            let slice = moves.as_mut_slice();
            for i in 0..n {
                // Root has no predecessor move, so counter-move band doesn't apply.
                scores[i] = score_move(board, slice[i], 0, tt_move, &self.killers, &self.history, None);
            }
        }

        let alpha_orig = alpha;
        let mut best_move = None;
        let mut best = -INFINITY;
        let mut alpha = alpha; // mutable running lower bound; `beta` stays fixed

        let slice = moves.as_mut_slice();
        for i in 0..n {
            select_next(slice, &mut scores, i);
            let mv = slice[i];

            // MultiPV: skip root moves that were the best in a previous PV line.
            if excluded.contains(&mv) { continue; }

            // Record for continuation history: negamax at ply=1 reads move_stack[0].
            self.move_stack[0] = Some(mv);
            self.hash_history.push(board.hash);
            let undo = board.make_move(mv);
            let score = if i == 0 {
                -self.negamax(board, depth - 1, -beta, -alpha, 1, false, tt)
            } else {
                // Null-window probe first; only re-search if it beats alpha.
                let probe = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1, false, tt);
                if probe > alpha && probe < beta && !self.stopped {
                    self.researches += 1;
                    -self.negamax(board, depth - 1, -beta, -alpha, 1, false, tt)
                } else {
                    probe
                }
            };
            board.unmake_move(mv, undo);
            self.hash_history.pop();

            if self.stopped {
                break; // partial iteration: caller discards it
            }
            if score > best {
                best = score;
                best_move = Some(mv);
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        // Beta cutoff at root — only reachable with a narrow aspiration window.
                        self.fail_highs += 1;
                        break;
                    }
                }
            }
        }

        // Store with the correct bound so the next iteration / re-search can use it.
        if !self.stopped {
            if let Some(bm) = best_move {
                let bound = if best <= alpha_orig {
                    Bound::Upper
                } else if best >= beta {
                    Bound::Lower
                } else {
                    Bound::Exact
                };
                tt.store(board.hash, depth, score_to_tt(best, 0), bound, Some(bm));
            }
        }

        (best_move, best)
    }

    fn negamax(
        &mut self,
        board: &mut Board,
        depth: u32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        // skip_null: if true, skip NMP at this node (parent was a null move —
        //            back-to-back passes would cause false cutoffs).
        skip_null: bool,
        tt: &TranspositionTable,
    ) -> i32 {
        // Warm the TT cache line for this node before any other work.
        // The prefetch is issued early so the cluster is in L1 by the time
        // we probe the TT a few instructions later.
        tt.prefetch(board.hash);

        if self.nodes & 4095 == 0 && self.should_stop() {
            return 0;
        }
        self.nodes += 1;

        // ── Draw detection (ply > 0 only; root always needs a move) ────────
        //
        // Order matters: repetition/50-move are rule-based draws and take
        // precedence over the evaluation.  Insufficient material is an
        // engine-side draw detection (no corresponding forced-draw rule, but
        // these positions can never be won).
        if ply > 0 {
            // Fifty-move rule: 100 half-moves since last pawn move or capture.
            if board.halfmove_clock >= 100 {
                return 0;
            }
            // Repetition: current position appeared at least once before.
            if self.is_repetition(board.hash, board.halfmove_clock) {
                return 0;
            }
            // Insufficient material: cannot force checkmate.
            if draw::is_insufficient_material(board) {
                return 0;
            }
        }
        // Track the deepest ply visited for the seldepth info field.
        if ply > self.max_ply { self.max_ply = ply; }
        // Hard ceiling: beyond MAX_PLY evaluate statically to prevent stack overflow.
        if ply as usize >= MAX_PLY {
            return self.evaluator.evaluate(board);
        }

        // Check extension: if the current side is in check, search one ply
        // deeper so forced sequences are never cut off at the horizon.
        // Safe because evasions always leave the *opponent* out of check,
        // so depth decreases at every other ply in a perpetual-check line.
        let in_check = board.is_in_check();
        let depth = if in_check {
            self.check_extensions += 1;
            depth + 1
        } else {
            depth
        };

        let alpha_orig = alpha;

        // True when this call is the verification search for a singular extension.
        // In that case we must not take TT cutoffs (the stored result was computed
        // without the move exclusion and would be wrong here).
        let is_se_search = self.se_excluded[ply as usize].is_some();

        // --- Tablebase probe ---
        //
        // When the position has few enough pieces and the 50-move clock is at
        // zero (just after an irreversible move), probe_wdl_after_zeroing gives
        // the exact game-theoretic result without any search.
        //
        // We skip the probe inside SE verification searches to avoid polluting
        // the narrow-window verification score.
        if !is_se_search {
            let limit = tablebase::piece_limit();
            if limit > 0 && board.halfmove_clock == 0 {
                let all = board.color(Color::White) | board.color(Color::Black);
                if all.0.count_ones() <= limit {
                    if let Some(wdl) = tablebase::probe_wdl(board) {
                        self.tb_hits += 1;
                        let tb_score = wdl.to_search_score();
                        // Win/Loss: return immediately (exact, perfect play).
                        // Effective draw: tighten alpha/beta to 0.
                        if !wdl.is_effective_draw() {
                            let bound = if tb_score > 0 { Bound::Lower } else { Bound::Upper };
                            tt.store(board.hash, 200, score_to_tt(tb_score, ply), bound, None);
                            return tb_score;
                        }
                        // Draw: score is 0; use it to tighten the window.
                        if 0 >= beta { return 0; }
                        if alpha < 0 { alpha = 0; }
                        tt.store(board.hash, 200, 0, Bound::Exact, None);
                    }
                }
            }
        }

        // --- Transposition-table probe ---
        let mut tt_move = None;
        // (score, depth, bound) saved for the SE check later in the move loop.
        let mut tt_data_se: Option<(i32, u8, Bound)> = None;
        if let Some(data) = tt.probe(board.hash) {
            self.tt_hits += 1;
            tt_move = data.best_move();
            let s = score_from_tt(data.score, ply);
            tt_data_se = Some((s, data.depth, data.bound));
            // Skip TT cutoffs inside an SE search; they were stored without the
            // current move exclusion and would corrupt the verification score.
            if data.depth as u32 >= depth && !is_se_search {
                match data.bound {
                    Bound::Exact => { self.tt_cutoffs += 1; return s; }
                    Bound::Lower if s >= beta  => { self.tt_cutoffs += 1; return s; }
                    Bound::Upper if s <= alpha => { self.tt_cutoffs += 1; return s; }
                    _ => {}
                }
            }
        }

        if depth == 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        // in_check is computed above; NMP, terminal detection, and LMR use it.

        // --- Null Move Pruning ---
        //
        // We "pass" and let the opponent move. If the resulting position (searched
        // at reduced depth) still exceeds beta, the real position is so good that
        // any real move will too — prune the branch.
        //
        // Conditions: not PV node, not in check, sufficient depth, and the side
        // to move has non-pawn material (no zugzwang risk).
        let is_pv = beta > alpha + 1;
        if !skip_null
            && !is_pv
            && !in_check
            && depth >= NMP_MIN_DEPTH
            && has_non_pawn_material(board)
        {
            // Only try NMP if the static evaluation suggests we're already "winning"
            // (i.e., above beta). If we're behind, passing would only make it worse.
            let static_eval = self.evaluator.evaluate(board);
            if static_eval >= beta {
                // Adaptive reduction: larger at higher depths.
                let r = 3 + depth / 3;
                let null_depth = depth.saturating_sub(r + 1);

                // Null move has no piece move, so no continuation-history context.
                self.move_stack[ply as usize] = None;
                let null_undo = board.make_null_move();
                // Search with skip_null=true to prevent back-to-back passes.
                let null_score =
                    -self.negamax(board, null_depth, -beta, -beta + 1, ply + 1, true, tt);
                board.unmake_null_move(null_undo);

                if self.stopped {
                    return 0;
                }
                if null_score >= beta {
                    // Null move cutoff. Return beta (fail-hard) rather than
                    // null_score because null_score can be inflated (mate scores
                    // from a potentially illegal pass position).
                    return beta;
                }
            }
        }

        // --- Razoring ---
        //
        // At depth 1, if the static eval plus a safety margin is still below
        // alpha, a quiet search is unlikely to help — fall straight through to
        // quiescence.  Avoids generating and trying all quiet moves at the leaf.
        if depth == 1 && !in_check && !is_pv {
            let static_eval = self.evaluator.evaluate(board);
            if static_eval + params::razor_margin() < alpha {
                return self.quiescence(board, alpha, beta, ply);
            }
        }

        // --- ProbCut ---
        //
        // Before the main search, try captures that SEE clearly above beta.
        // A shallow null-window search at (depth − PROBCUT_REDUCTION) confirms
        // the cutoff cheaply.  If any capture scores above (beta + margin) at
        // that reduced depth, the full search would too.
        if !is_pv
            && depth >= PROBCUT_MIN_DEPTH
            && !in_check
            && beta.abs() < MATE_IN_MAX
        {
            let pc_beta  = beta + params::probcut_margin();
            let pc_depth = depth.saturating_sub(PROBCUT_REDUCTION);
            let pc_moves = board.legal_moves();

            for &mv in pc_moves.iter() {
                if !mv.is_capture() { continue; }
                // Only try captures that are likely to be worth at least pc_beta.
                if see(board, mv) + params::probcut_margin() < 0 { continue; }

                self.move_stack[ply as usize] = Some(mv);
                self.hash_history.push(board.hash);
                let undo = board.make_move(mv);
                let score =
                    -self.negamax(board, pc_depth, -pc_beta, -pc_beta + 1, ply + 1, false, tt);
                board.unmake_move(mv, undo);
                self.hash_history.pop();

                if self.stopped { return 0; }

                if score >= pc_beta {
                    // Record as a lower bound at the reduced depth so the main
                    // search can skip re-exploring this node.
                    tt.store(board.hash, pc_depth,
                             score_to_tt(score, ply), Bound::Lower, Some(mv));
                    return score;
                }
            }
        }

        // --- Move generation ---
        let mut moves = board.legal_moves();
        if moves.is_empty() {
            return if in_check { -MATE + ply } else { 0 };
        }

        // --- Continuation history context ----------------------------------------
        // Predecessor moves from the move stack, needed for counter-move lookup
        // and continuation history bonuses.
        let prev_mv  = if ply > 0 { self.move_stack[(ply as usize) - 1] } else { None };
        let prev2_mv = if ply > 1 { self.move_stack[(ply as usize) - 2] } else { None };

        // (piece_type_index, to_square_index) of each predecessor — used to key
        // into the continuation history tables.
        let prev_key: Option<(usize, usize)> = prev_mv.and_then(|m| {
            board.piece_at(m.to()).map(|p| (p.piece_type.index(), m.to().index()))
        });
        let prev2_key: Option<(usize, usize)> = prev2_mv.and_then(|m| {
            board.piece_at(m.to()).map(|p| (p.piece_type.index(), m.to().index()))
        });
        let counter_mv = self.counter_moves.get(prev_mv);

        let n = moves.len();
        let mut scores = [0i32; 256];
        {
            let slice = moves.as_mut_slice();
            for i in 0..n {
                let base = score_move(
                    board, slice[i], ply as usize, tt_move,
                    &self.killers, &self.history, counter_mv,
                );
                // For quiet moves below the counter-move band, blend in continuation
                // history so moves that fit the current position's "context" rise.
                scores[i] = if !slice[i].is_capture()
                    && !slice[i].is_promotion()
                    && base < SCORE_COUNTER
                {
                    let cont = self.cont_bonus(board, slice[i], &prev_key, &prev2_key);
                    (base + cont).clamp(-HISTORY_MAX, HISTORY_MAX)
                } else {
                    base
                };
            }
        }

        // --- Futility pruning setup ---
        //
        // At shallow depths (1-2) quiet moves are unlikely to raise alpha if the
        // static evaluation is already too far below it.  We compute the margin
        // once here and check per-move in the loop.
        //
        // Conditions: not PV node, not in check (forced moves can never be skipped).
        let futility_base = if !is_pv && !in_check && depth <= 2 {
            let eval = self.evaluator.evaluate(board);
            let margin = if depth == 1 { params::futility_margin_1() } else { params::futility_margin_2() };
            Some(eval + margin)
        } else {
            None
        };

        let side = board.side_to_move;
        let mut best = -INFINITY;
        let mut best_move = None;
        let slice = moves.as_mut_slice();

        for i in 0..n {
            select_next(slice, &mut scores, i);
            let mv = slice[i];

            // Skip the move excluded during a singular extension verification.
            if self.se_excluded[ply as usize] == Some(mv) { continue; }

            // Futility pruning: at depth 1-2, skip quiet moves that can't
            // possibly raise alpha even with an optimistic static eval bump.
            // Always search at least one move (i > 0 guard) and skip only if
            // we already have a real score to fall back on.
            if let Some(base) = futility_base {
                if i > 0
                    && !mv.is_capture()
                    && !mv.is_promotion()
                    && base < alpha
                    && best > -INFINITY
                {
                    continue;
                }
            }

            // Late Move Pruning: at low depth, once we have searched enough
            // quiet moves, the remaining ones are very unlikely to raise alpha.
            // Captures and promotions are always searched.
            if !is_pv
                && !in_check
                && depth <= LMP_MAX_DEPTH
                && !mv.is_capture()
                && !mv.is_promotion()
                && i >= params::lmp_base() + (depth * depth) as usize
                && best > -INFINITY
            {
                continue;
            }

            // --- Singular Extension ---
            //
            // For the TT move at sufficient depth: verify it is the *only* move
            // that scores above a threshold by searching all other moves at
            // half depth in a narrow window.  If they all fail low, this move
            // is "singular" — extend it one extra ply.
            //
            // Guard: not already inside an SE search (`is_se_search`) to
            // prevent recursive singularity checks.
            let mut extension = 0u32;
            if tt_move == Some(mv)
                && depth >= SE_MIN_DEPTH
                && !is_se_search
            {
                if let Some((tt_sc, tt_d, tt_b)) = tt_data_se {
                    if (tt_b == Bound::Lower || tt_b == Bound::Exact)
                        && tt_d as u32 >= depth.saturating_sub(3)
                        && tt_sc.abs() < MATE_IN_MAX
                    {
                        let s_beta = (tt_sc - params::se_depth_factor() * depth as i32)
                            .max(-MATE_IN_MAX + 1);
                        let se_depth = depth / 2;
                        // Exclude this move and recurse on the same board/ply.
                        self.se_excluded[ply as usize] = Some(mv);
                        let se_score = self.negamax(
                            board, se_depth, s_beta - 1, s_beta, ply, true, tt,
                        );
                        self.se_excluded[ply as usize] = None;
                        if self.stopped { return 0; }
                        if se_score < s_beta {
                            extension = 1;
                            self.singular_extensions += 1;
                        }
                    }
                }
            }

            // Record the move at this ply so the child can look up continuation
            // history for its own ordering and history updates.
            self.move_stack[ply as usize] = Some(mv);
            self.hash_history.push(board.hash);
            let undo = board.make_move(mv);

            let score = if i == 0 {
                // First move: full window, no reduction.  Apply singular extension
                // if this move was verified as the only good option at this node.
                -self.negamax(board, depth - 1 + extension, -beta, -alpha, ply + 1, false, tt)
            } else {
                // --- Late Move Reductions ---
                //
                // Moves sorted late are probably bad. Search them at reduced depth
                // first. If they beat alpha (surprising), re-search at full depth.
                //
                // Disabled for: tactical moves (captures/promos), killers, check
                // evasions, and shallow depths where reduction is meaningless.
                let use_lmr = !in_check
                    && depth >= LMR_MIN_DEPTH
                    && i >= LMR_MOVE_THRESHOLD
                    && !mv.is_capture()
                    && !mv.is_promotion()
                    && self.killers.first(ply as usize) != Some(mv)
                    && self.killers.second(ply as usize) != Some(mv);

                let reduction = if use_lmr { lmr_reduction(depth, i).max(1) } else { 0 };
                // Ensure search_depth >= 0 (u32 saturating_sub prevents underflow).
                let search_depth = (depth - 1).saturating_sub(reduction);

                // Initial probe: null window at (possibly reduced) depth.
                let probe =
                    -self.negamax(board, search_depth, -alpha - 1, -alpha, ply + 1, false, tt);

                // LMR failed high: the move was good at reduced depth.
                // Re-search at full depth to confirm and get an accurate score.
                let probe = if use_lmr && reduction > 0 && probe > alpha && !self.stopped {
                    self.researches += 1;
                    -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, false, tt)
                } else {
                    probe
                };

                // PVS: null window beat alpha at full depth → full-window re-search.
                // This only fires when we're at a PV node (beta > alpha + 1).
                if probe > alpha && probe < beta && !self.stopped {
                    self.researches += 1;
                    -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, false, tt)
                } else {
                    probe
                }
            };

            board.unmake_move(mv, undo);
            self.hash_history.pop();

            if self.stopped {
                return 0; // unwind without using this bogus score
            }
            if score > best {
                best = score;
                best_move = Some(mv);
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        self.fail_highs += 1;
                        if !mv.is_capture() && !mv.is_promotion() {
                            self.killers.add(ply as usize, mv);
                            self.history.add(side, mv, depth);

                            // Counter-move: record this quiet move as the refutation.
                            if let Some(pm) = prev_mv {
                                self.counter_moves.set(pm, mv);
                            }

                            // Continuation history bonus for the cutoff move.
                            let h_bonus = (depth as i32).min(13) * (depth as i32).min(13);
                            if let Some((pp, pt)) = prev_key {
                                if let Some(piece) = board.piece_at(mv.from()) {
                                    self.cont_hist_1.update(pp, pt, piece.piece_type.index(), mv.to().index(), h_bonus);
                                }
                            }
                            if let Some((pp, pt)) = prev2_key {
                                if let Some(piece) = board.piece_at(mv.from()) {
                                    self.cont_hist_2.update(pp, pt, piece.piece_type.index(), mv.to().index(), h_bonus);
                                }
                            }

                            // History malus for quiet moves tried before this cutoff.
                            // They were ordered too high; penalise so future searches rank
                            // them lower in similar positions.
                            let h_malus = h_bonus;
                            for j in 0..i {
                                let prior = slice[j];
                                if !prior.is_capture() && !prior.is_promotion() {
                                    self.history.add_malus(side, prior, depth);
                                    if let Some((pp, pt)) = prev_key {
                                        if let Some(piece) = board.piece_at(prior.from()) {
                                            self.cont_hist_1.update(pp, pt, piece.piece_type.index(), prior.to().index(), -h_malus);
                                        }
                                    }
                                    if let Some((pp, pt)) = prev2_key {
                                        if let Some(piece) = board.piece_at(prior.from()) {
                                            self.cont_hist_2.update(pp, pt, piece.piece_type.index(), prior.to().index(), -h_malus);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        // --- Store the result with the right bound ---
        let bound = if best <= alpha_orig {
            Bound::Upper // failed low: score is an upper bound
        } else if best >= beta {
            Bound::Lower // failed high: score is a lower bound
        } else {
            Bound::Exact
        };
        tt.store(board.hash, depth, score_to_tt(best, ply), bound, best_move);

        best
    }

    /// Sum of 1-ply and 2-ply continuation history bonuses for `mv`.
    #[inline]
    fn cont_bonus(
        &self,
        board: &Board,
        mv: Move,
        prev_key: &Option<(usize, usize)>,
        prev2_key: &Option<(usize, usize)>,
    ) -> i32 {
        let Some(piece) = board.piece_at(mv.from()) else { return 0; };
        let curr_pt = piece.piece_type.index();
        let curr_to = mv.to().index();
        let mut bonus = 0i32;
        if let Some(&(pp, pt)) = prev_key.as_ref() {
            bonus += self.cont_hist_1.get(pp, pt, curr_pt, curr_to);
        }
        if let Some(&(pp, pt)) = prev2_key.as_ref() {
            bonus += self.cont_hist_2.get(pp, pt, curr_pt, curr_to);
        }
        bonus
    }

    /// Fraction of interior nodes that ended in a beta cutoff.
    ///
    /// Healthy values: 55–70 %. Lower suggests poor move ordering; higher is
    /// possible at very narrow aspiration windows.
    pub fn fail_high_rate(&self) -> f64 {
        if self.nodes == 0 { return 0.0; }
        self.fail_highs as f64 / self.nodes as f64
    }

    /// Fraction of nodes that triggered an additional re-search (LMR or PVS).
    pub fn research_rate(&self) -> f64 {
        if self.nodes == 0 { return 0.0; }
        self.researches as f64 / self.nodes as f64
    }

    /// Fraction of TT probes that caused an immediate early-return cutoff.
    pub fn tt_cutoff_rate(&self) -> f64 {
        if self.tt_hits == 0 { return 0.0; }
        self.tt_cutoffs as f64 / self.tt_hits as f64
    }

    /// Fraction of internal nodes where a check extension was applied.
    pub fn check_extension_rate(&self) -> f64 {
        if self.nodes == 0 { return 0.0; }
        self.check_extensions as f64 / self.nodes as f64
    }

    /// Fraction of interior nodes where a singular extension fired.
    pub fn singular_extension_rate(&self) -> f64 {
        if self.nodes == 0 { return 0.0; }
        self.singular_extensions as f64 / self.nodes as f64
    }
}

// -------------------------------------------------------------------------
// Helper functions
// -------------------------------------------------------------------------

/// True if the side to move has at least one piece besides pawns and the king.
///
/// NMP is unsafe in positions where having the move is a liability (zugzwang).
/// This guard rules out pure pawn-and-king endgames, where zugzwang is realistic.
#[inline]
fn has_non_pawn_material(board: &Board) -> bool {
    let side = board.side_to_move;
    let our = board.color(side);
    let pawns = board.pieces(PieceType::Pawn);
    let kings = board.pieces(PieceType::King);
    // Any own piece that is neither a pawn nor a king.
    !(our & !pawns & !kings).is_empty()
}

/// Compute the LMR depth reduction for a move at the given depth and move index.
///
/// Formula: `floor(ln(depth) × ln(move_index + 1) / 2)`
///
/// Returns 0 for very small inputs (depth < 2 or move_index < 1).
/// The caller adds `.max(1)` to ensure at least a 1-ply reduction.
///
/// Approximate table (values above 0):
/// ```text
/// depth\index |  2   3   4   5   6   8  10  15  20
/// ------------|------------------------------------------
///  3          |  0   0   0   0   0   0   0   1   1
///  4          |  0   0   1   1   1   1   1   1   2
///  5          |  0   1   1   1   1   1   1   2   2
///  7          |  1   1   1   1   1   2   2   2   3
/// 10          |  1   1   1   2   2   2   2   3   3
/// 15          |  1   1   2   2   2   2   3   3   4
/// ```
fn lmr_reduction(depth: u32, move_index: usize) -> u32 {
    if depth < 2 || move_index < 1 {
        return 0;
    }
    ((depth as f64).ln() * ((move_index + 1) as f64).ln() / 2.0) as u32
}

// -------------------------------------------------------------------------
// Score / TT helpers
// -------------------------------------------------------------------------

/// Convert a root-relative mate score to a node-relative one for TT storage.
fn score_to_tt(score: i32, ply: i32) -> i16 {
    let adjusted = if score >= MATE_IN_MAX {
        score + ply
    } else if score <= -MATE_IN_MAX {
        score - ply
    } else {
        score
    };
    adjusted as i16
}

/// Inverse of [`score_to_tt`]: node-relative back to root-relative.
fn score_from_tt(score: i16, ply: i32) -> i32 {
    let s = score as i32;
    if s >= MATE_IN_MAX {
        s - ply
    } else if s <= -MATE_IN_MAX {
        s + ply
    } else {
        s
    }
}

/// Selection-sort step: pull the highest-scoring move in `slice[i..]` to `i`,
/// keeping `scores` aligned. Lazy — we only sort as far as cutoffs let us reach.
#[inline]
fn select_next(slice: &mut [Move], scores: &mut [i32; 256], i: usize) {
    let mut best_idx = i;
    for j in (i + 1)..slice.len() {
        if scores[j] > scores[best_idx] {
            best_idx = j;
        }
    }
    slice.swap(i, best_idx);
    scores.swap(i, best_idx);
}

/// Convenience: a synchronous fixed-depth search with its own small TT.
///
/// Uses the [`HandcraftedEvaluator`] by default. To use a different evaluator,
/// construct a [`Searcher`] directly with the evaluator of your choice.
pub fn search(board: &mut Board, depth: u32) -> SearchResult {
    let depth = depth.max(1);
    let tt = TranspositionTable::new(8);
    let mut searcher = Searcher::new(None, Arc::new(AtomicBool::new(false)), None, HandcraftedEvaluator);
    let (best_move, score) = searcher.search_root(board, depth, None, &tt, &[], -INFINITY, INFINITY);
    SearchResult {
        best_move,
        score,
        depth,
        nodes: searcher.nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{is_mate_score, mate_distance_plies};

    #[test]
    fn finds_mate_in_one() {
        let mut board = Board::from_fen("7k/8/6K1/8/8/8/8/Q7 w - - 0 1").unwrap();
        let result = search(&mut board, 2);
        assert!(is_mate_score(result.score), "expected a mate score");
        assert_eq!(mate_distance_plies(result.score), 1);
        let mv = result.best_move.expect("must have a move");
        board.make_move(mv);
        assert!(board.is_checkmate(), "best move should deliver mate");
    }

    #[test]
    fn finds_mate_in_two() {
        // Qh5# is forced in two moves; NMP/LMR must not prune away the line.
        let mut board = Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
            .unwrap();
        let result = search(&mut board, 4);
        assert!(is_mate_score(result.score), "should find forced mate; got {}", result.score);
    }

    #[test]
    fn grabs_a_hanging_queen() {
        let mut board = Board::from_fen("3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1").unwrap();
        let result = search(&mut board, 1);
        assert_eq!(result.best_move.map(|m| m.to_uci()), Some("d1d8".to_string()));
        assert!(result.score > 700, "winning a queen should score high");
    }

    #[test]
    fn avoids_losing_capture_via_quiescence() {
        let mut board =
            Board::from_fen("rnbqkbnr/pp2pppp/2p5/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
                .unwrap();
        let result = search(&mut board, 4);
        assert!(result.score.abs() < 150, "score was {}", result.score);
    }

    #[test]
    fn reports_stalemate_as_draw() {
        let mut board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert!(board.is_stalemate());
        let result = search(&mut board, 3);
        assert_eq!(result.best_move, None);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn tt_preserves_result_correctness() {
        let mut board = Board::from_fen("3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1").unwrap();
        let shallow = search(&mut board, 3);
        let deep = search(&mut board, 5);
        assert_eq!(shallow.best_move.map(|m| m.to_uci()), Some("d1d8".to_string()));
        assert_eq!(deep.best_move.map(|m| m.to_uci()), Some("d1d8".to_string()));
    }

    #[test]
    fn mate_score_survives_tt_round_trip() {
        for ply in 0..20 {
            let s = MATE - 5;
            assert_eq!(score_from_tt(score_to_tt(s, ply), ply), s);
            assert_eq!(score_from_tt(score_to_tt(-s, ply), ply), -s);
        }
        assert_eq!(score_from_tt(score_to_tt(42, 7), 7), 42);
    }

    #[test]
    fn node_limit_stops_search() {
        use crate::eval::HandcraftedEvaluator;
        let mut board = Board::from_fen(crate::board::STARTING_FEN).unwrap();
        let mut tt = TranspositionTable::new(1);
        let mut searcher = Searcher::new(None, Arc::new(AtomicBool::new(false)), Some(1000), HandcraftedEvaluator);
        searcher.search_root(&mut board, 12, None, &mut tt, &[], -INFINITY, INFINITY);
        assert!(searcher.stopped);
        assert!(searcher.nodes < 200_000);
    }

    #[test]
    fn fifty_move_rule_returns_draw_in_search() {
        // KR vs K with halfmove_clock = 100: every quiet child at ply=1 triggers
        // the fifty-move rule.  With no captures available, all root moves score 0.
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 1").unwrap();
        let result = search(&mut board, 3);
        assert_eq!(result.score, 0, "fifty-move rule should return 0, got {}", result.score);
    }

    #[test]
    fn insufficient_material_returns_draw_in_search() {
        // KN vs K: a knight and king cannot force mate.
        // At every child node (ply ≥ 1) the position is still KN vs K → 0.
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4KN2 w - - 0 1").unwrap();
        let result = search(&mut board, 4);
        assert_eq!(result.score, 0, "KN vs K should be a draw, got {}", result.score);
    }

    #[test]
    fn is_repetition_detects_match() {
        let history = vec![0xDEAD_BEEF_u64, 0xCAFE_BABE, 0xFEED_BEEF];
        let searcher = Searcher::new_with_history(
            &history,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            HandcraftedEvaluator,
        );
        // 0xDEAD_BEEF is at index 0 → found with lookback=10
        assert!(searcher.is_repetition(0xDEAD_BEEF, 10));
        // 0xFEED_BEEF is at index 2 → found within the last 1 entry
        assert!(searcher.is_repetition(0xFEED_BEEF, 1));
        // 0xDEAD_BEEF is at index 0 but lookback=2 only covers indices [1,2]
        assert!(!searcher.is_repetition(0xDEAD_BEEF, 2));
        // Hash never in history
        assert!(!searcher.is_repetition(0x1234_5678, 10));
        // halfmove_clock = 0 → empty slice → no repetition
        assert!(!searcher.is_repetition(0xDEAD_BEEF, 0));
    }

    #[test]
    fn nmp_skipped_in_pawn_endgame() {
        // King+pawns only: NMP would be unsafe (zugzwang is possible).
        // The engine must still find the correct result without pruning.
        let mut board = Board::from_fen("8/8/8/8/8/4K3/4P3/7k w - - 0 1").unwrap();
        let r1 = search(&mut board, 4);
        // White is ahead (pawn + king advantage).
        assert!(r1.score > 0, "white should be better in this K+P endgame");
    }

    #[test]
    fn nmp_does_not_corrupt_normal_search() {
        // Verify NMP gives the same best move as the existing regression tests.
        let mut board = Board::from_fen("3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1").unwrap();
        let result = search(&mut board, 5);
        assert_eq!(
            result.best_move.map(|m| m.to_uci()),
            Some("d1d8".to_string()),
            "NMP must not corrupt the obvious best move"
        );
    }

    // ── Singular extension tests ──────────────────────────────────────────────

    #[test]
    fn singular_extension_counter_fires_at_sufficient_depth() {
        // SE requires TT data from a prior iteration to know the expected score
        // for the node.  Simulate iterative deepening by doing a depth-5 warmup
        // pass (which populates the TT), then a depth-7 pass on the same TT.
        let mut board = Board::from_fen(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        )
        .unwrap();
        let tt = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));

        // Warmup: populate TT so the deeper pass has data to check singularity.
        let mut warmup = Searcher::new(None, Arc::clone(&stop), None, HandcraftedEvaluator);
        warmup.search_root(&mut board, 5, None, &tt, &[], -INFINITY, INFINITY);

        // Deep pass: SE can now find TT entries with sufficient depth.
        tt.new_generation();
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);
        s.search_root(&mut board, 7, None, &tt, &[], -INFINITY, INFINITY);
        assert!(
            s.singular_extensions > 0,
            "expected at least one singular extension at depth 7 after TT warmup"
        );
    }

    #[test]
    fn singular_extension_does_not_change_obvious_best_move() {
        // The best move in a clearly won position must be the same with or
        // without the SE counter incrementing — correctness check.
        let mut board = Board::from_fen("3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1").unwrap();
        let result = search(&mut board, 7);
        assert_eq!(
            result.best_move.map(|m| m.to_uci()),
            Some("d1d8".to_string()),
            "SE must not corrupt the obvious best move"
        );
    }

    #[test]
    fn lmr_reduction_table() {
        // Reduction is 0 at very small inputs.
        assert_eq!(lmr_reduction(1, 5), 0);
        assert_eq!(lmr_reduction(5, 0), 0);
        // Reduction grows with depth and move index.
        assert!(lmr_reduction(10, 10) >= lmr_reduction(5, 5));
        assert!(lmr_reduction(10, 10) >= lmr_reduction(10, 3));
        // Reasonable upper bound: no absurd reductions at achievable depths.
        assert!(lmr_reduction(64, 256) <= 20);
    }

    #[test]
    fn has_non_pawn_material_detects_pieces() {
        // Position with queens: has non-pawn material.
        let board = Board::from_fen("3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1").unwrap();
        assert!(has_non_pawn_material(&board));

        // King + pawn vs King: no non-pawn material for White.
        let board = Board::from_fen("8/8/8/8/8/4K3/4P3/7k w - - 0 1").unwrap();
        assert!(!has_non_pawn_material(&board));

        // King + rook vs King: has non-pawn material.
        let board = Board::from_fen("8/8/8/8/8/8/8/R3K2k w - - 0 1").unwrap();
        assert!(has_non_pawn_material(&board));
    }

    #[test]
    fn null_move_make_unmake_restores_position() {
        use crate::board::STARTING_FEN;
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let before_fen = board.to_fen();
        let before_hash = board.hash;

        let undo = board.make_null_move();
        // Side should have flipped.
        assert_ne!(board.side_to_move, Board::from_fen(STARTING_FEN).unwrap().side_to_move);
        // Hash should have changed.
        assert_ne!(board.hash, before_hash);

        board.unmake_null_move(undo);
        assert_eq!(board.to_fen(), before_fen, "position not restored after null move");
        assert_eq!(board.hash, before_hash, "hash not restored after null move");
    }

    #[test]
    fn null_move_clears_ep_square() {
        // Position with an active EP square.
        let fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        let mut board = Board::from_fen(fen).unwrap();
        assert!(board.ep_square.is_some(), "EP should be set before null move");

        let undo = board.make_null_move();
        assert!(board.ep_square.is_none(), "EP should be cleared by null move");

        board.unmake_null_move(undo);
        assert!(board.ep_square.is_some(), "EP should be restored after unmake");
    }

    // ── Check extension tests ─────────────────────────────────────────────────

    #[test]
    fn check_extension_counter_increments_when_in_check() {
        // A position where White is immediately in check (Black queen on f3 gives check
        // after some path). Use a direct Searcher to inspect the counter.
        // "7k/8/8/8/8/5q2/8/4K3 w - - 0 1": White king on e1, Black queen f3 — White in check.
        let mut board = Board::from_fen("7k/8/8/8/8/5q2/8/4K3 w - - 0 1").unwrap();
        let tt = TranspositionTable::new(1);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);
        s.search_root(&mut board, 3, None, &tt, &[], -INFINITY, INFINITY);
        assert!(
            s.check_extensions > 0,
            "expected check extensions to fire in a position starting in check"
        );
    }

    #[test]
    fn check_extension_does_not_change_best_move_on_quiet_position() {
        // In a quiet non-check position, check extensions should not affect which
        // move is chosen — they only add depth in forced lines.
        let mut board = Board::from_fen("3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1").unwrap();
        let tt = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);
        let (mv, _) = s.search_root(&mut board, 4, None, &tt, &[], -INFINITY, INFINITY);
        assert_eq!(mv.map(|m| m.to_uci()), Some("d1d8".to_string()));
    }

    #[test]
    fn check_extension_handles_check_evasion_correctly() {
        // Starting from a position where White is in check, the engine must
        // find a legal evasion (not crash, not return None, not return a junk score).
        // "7k/8/8/8/8/5q2/8/4K3 w - - 0 1": White king e1 in check from Qf3.
        let mut board = Board::from_fen("7k/8/8/8/8/5q2/8/4K3 w - - 0 1").unwrap();
        let result = search(&mut board, 4);
        // Engine is king-only vs. king+queen — must be losing.
        assert!(result.best_move.is_some(), "must find a check evasion from in-check position");
        assert!(result.score < 0, "White should be losing here, got {}", result.score);
    }

    // ── Aspiration window tests ───────────────────────────────────────────────

    #[test]
    fn aspiration_window_same_best_move_as_full_window() {
        // A full-window search and a narrow aspiration search must agree on
        // the best move (correctness check).
        let mut board = Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
            .unwrap();
        let tt1 = TranspositionTable::new(4);
        let tt2 = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));

        // Full window
        let mut s1 = Searcher::new(None, Arc::clone(&stop), None, HandcraftedEvaluator);
        let (mv_full, score_full) =
            s1.search_root(&mut board, 5, None, &tt1, &[], -INFINITY, INFINITY);

        // Aspiration window centred on the full-window score
        let delta = 50;
        let lo = score_full - delta;
        let hi = score_full + delta;
        let mut s2 = Searcher::new(None, stop, None, HandcraftedEvaluator);
        let (mv_asp, _) = s2.search_root(&mut board, 5, None, &tt2, &[], lo, hi);

        assert_eq!(
            mv_full.map(|m| m.to_uci()),
            mv_asp.map(|m| m.to_uci()),
            "aspiration and full-window should agree on best move when window is exact"
        );
    }

    #[test]
    fn aspiration_fail_low_returns_score_at_most_alpha() {
        // If the window sits well above the true score the search fails low:
        // the returned score must be <= alpha (the lower bound of the window).
        // The starting position scores near 0; a window of [500, 600] is far above it.
        use crate::board::STARTING_FEN;
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let tt = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);

        let alpha = 500;
        let beta = 600;
        let (_, score) = s.search_root(&mut board, 4, None, &tt, &[], alpha, beta);
        assert!(
            score <= alpha,
            "expected fail-low with window [{alpha},{beta}], got score={score}"
        );
    }

    #[test]
    fn aspiration_fail_high_returns_score_above_beta() {
        // If beta is set too low (below the actual score), search returns score >= beta.
        let mut board = Board::from_fen("3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1").unwrap();
        let tt = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);

        // Full-window first to find the actual score.
        let (_, actual) = s.search_root(&mut board, 4, None, &tt, &[], -INFINITY, INFINITY);

        // Now search with a window far below the actual score.
        let tt2 = TranspositionTable::new(4);
        let stop2 = Arc::new(AtomicBool::new(false));
        let mut s2 = Searcher::new(None, stop2, None, HandcraftedEvaluator);
        let beta = actual - 50;
        let alpha = beta - 100;
        let (_, score) = s2.search_root(&mut board, 4, None, &tt2, &[], alpha, beta);
        assert!(
            score >= beta,
            "expected fail-high with beta={beta}, actual={actual}, got score={score}"
        );
    }
}
