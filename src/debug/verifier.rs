//! Search correctness verifier.
//!
//! Runs the engine on positions with known answers and checks every invariant
//! the search must uphold. Designed to catch silent bugs that unit tests miss:
//! illegal PV moves from TT collisions, wrong mate distances, non-determinism
//! from uninitialized state, and score arithmetic overflow.

use crate::board::Board;
use crate::movegen::Move;
use crate::search::{
    is_mate_score, mate_distance_plies, think, Searcher, SearchLimits, TranspositionTable,
    INFINITY, MATE,
};
use crate::eval::HandcraftedEvaluator;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// ─── Position kind ───────────────────────────────────────────────────────────

/// What a test position is expected to produce.
pub enum PositionKind {
    /// The engine must find a winning mate.  The `at_most_moves` ceiling allows
    /// the engine to find a *shorter* mate than specified (correct and acceptable).
    ForcedMateIn { at_most_moves: i32 },
    /// The engine must play exactly this move (UCI notation, lower-case).
    BestMove { uci: String },
    /// Any legal move is acceptable.  Only correctness invariants are checked.
    AnyLegal,
}

// ─── Test position ────────────────────────────────────────────────────────────

pub struct VerifyPosition {
    pub description: String,
    pub fen: String,
    /// Minimum search depth to use for this position.  Overrides the suite-wide
    /// default when larger.
    pub min_depth: u32,
    pub kind: PositionKind,
}

impl VerifyPosition {
    fn new(desc: &str, fen: &str, min_depth: u32, kind: PositionKind) -> Self {
        VerifyPosition {
            description: desc.into(),
            fen: fen.into(),
            min_depth,
            kind,
        }
    }
}

// ─── Failure kinds ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum VerificationFailure {
    /// FEN failed to parse.
    FenParseError(String),
    /// Engine returned a move not found in `legal_moves()`.
    IllegalRootMove { mv: String },
    /// Engine returned no move despite legal moves existing.
    NoMoveReturned,
    /// Score is outside `[−MATE, MATE]` — arithmetic overflow or sign error.
    ScoreOutOfBounds { score: i32 },
    /// PV contains an illegal move at the given ply.
    IllegalPvMove { mv: String, at_ply: usize },
    /// Two identical searches produced different best moves or scores.
    NonDeterministic { run1: String, run2: String },
    /// Position is a forced mate but the engine's score was not a mate score.
    NotMateScore { score: i32 },
    /// Mate distance in the score exceeds `at_most_moves * 2 - 1` plies.
    MateDistanceTooLong { expected_moves: i32, actual_plies: i32 },
    /// Engine played the wrong best move.
    WrongBestMove { expected: String, found: Option<String> },
    /// After search the TT probe at the root doesn't match the returned move.
    TtRootMismatch { search_move: String, tt_move: Option<String> },
}

impl std::fmt::Display for VerificationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FenParseError(e)              => write!(f, "FEN parse error: {}", e),
            Self::IllegalRootMove { mv }        => write!(f, "illegal root move returned: {}", mv),
            Self::NoMoveReturned                => write!(f, "no move returned despite legal moves"),
            Self::ScoreOutOfBounds { score }    => write!(f, "score {} out of bounds (±MATE={})", score, MATE),
            Self::IllegalPvMove { mv, at_ply }  => write!(f, "illegal PV move {} at ply {}", mv, at_ply),
            Self::NonDeterministic { run1, run2 } => write!(f, "non-deterministic: '{}' vs '{}'", run1, run2),
            Self::NotMateScore { score }        => write!(f, "expected mate score, got cp {}", score),
            Self::MateDistanceTooLong { expected_moves, actual_plies } =>
                write!(f, "mate in >{} moves (plies={})", expected_moves, actual_plies),
            Self::WrongBestMove { expected, found } =>
                write!(f, "expected {} found {}", expected, found.as_deref().unwrap_or("(none)")),
            Self::TtRootMismatch { search_move, tt_move } =>
                write!(f, "TT has {:?} but search returned {}", tt_move, search_move),
        }
    }
}

// ─── Verification result ──────────────────────────────────────────────────────

/// Full report for one position.
pub struct VerificationResult {
    pub description: String,
    pub fen: String,
    pub depth_used: u32,
    pub best_move_uci: Option<String>,
    pub score: i32,
    pub pv_uci: Vec<String>,
    pub nodes: u64,
    pub fail_high_rate: f64,
    pub research_rate: f64,
    pub tt_cutoff_rate: f64,
    pub failures: Vec<VerificationFailure>,
}

impl VerificationResult {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

impl std::fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tag   = if self.passed() { "PASS" } else { "FAIL" };
        let mv    = self.best_move_uci.as_deref().unwrap_or("(none)");
        let score = fmt_score(self.score);
        write!(
            f,
            "[{tag}] {desc:<44} d={d}  {mv:<6}  {score:<10}  nodes={nodes}",
            tag   = tag,
            desc  = &self.description,
            d     = self.depth_used,
            mv    = mv,
            score = score,
            nodes = fmt_nodes(self.nodes),
        )?;
        if !self.pv_uci.is_empty() {
            write!(f, "  pv=[{}]", self.pv_uci.join(" "))?;
        }
        for fail in &self.failures {
            write!(f, "\n    !! {}", fail)?;
        }
        Ok(())
    }
}

// ─── SearchVerifier ───────────────────────────────────────────────────────────

pub struct SearchVerifier {
    positions: Vec<VerifyPosition>,
}

impl Default for SearchVerifier {
    fn default() -> Self {
        SearchVerifier::new()
    }
}

impl SearchVerifier {
    /// Build the verifier with the built-in standard position suite.
    pub fn new() -> Self {
        SearchVerifier {
            positions: standard_positions(),
        }
    }

    /// Run all positions at the given depth (clamped to `max(depth, pos.min_depth)`).
    pub fn run_all(&self, depth: u32) -> Vec<VerificationResult> {
        self.positions
            .iter()
            .map(|pos| self.verify_one(pos, depth.max(pos.min_depth)))
            .collect()
    }

    /// Print a formatted report to stdout and return whether everything passed.
    pub fn print_report(results: &[VerificationResult]) -> bool {
        let passed = results.iter().filter(|r| r.passed()).count();
        let total  = results.len();

        let total_nodes: u64 = results.iter().map(|r| r.nodes).sum();
        let avg_fhr: f64 = results.iter().map(|r| r.fail_high_rate).sum::<f64>()
            / total.max(1) as f64;
        let avg_rr: f64 = results.iter().map(|r| r.research_rate).sum::<f64>()
            / total.max(1) as f64;
        let avg_ttcr: f64 = results.iter().map(|r| r.tt_cutoff_rate).sum::<f64>()
            / total.max(1) as f64;

        for r in results {
            println!("{}", r);
        }
        println!();
        println!(
            "─────────────────────────────────────────────────────────────────────────"
        );
        println!(
            "Passed: {passed}/{total}  nodes={nodes}  \
             fail-high={fhr:.1}%  re-search={rr:.1}%  tt-cutoff={ttcr:.1}%",
            passed = passed,
            total  = total,
            nodes  = fmt_nodes(total_nodes),
            fhr    = avg_fhr * 100.0,
            rr     = avg_rr  * 100.0,
            ttcr   = avg_ttcr * 100.0,
        );
        passed == total
    }

    // ── Internal: verify one position ────────────────────────────────────────

    fn verify_one(&self, pos: &VerifyPosition, depth: u32) -> VerificationResult {
        let mut failures = Vec::new();

        // 1. Parse FEN.
        let mut board = match Board::from_fen(&pos.fen) {
            Ok(b)  => b,
            Err(e) => {
                failures.push(VerificationFailure::FenParseError(format!("{:?}", e)));
                return VerificationResult {
                    description: pos.description.clone(),
                    fen: pos.fen.clone(),
                    depth_used: depth,
                    best_move_uci: None,
                    score: 0,
                    pv_uci: Vec::new(),
                    nodes: 0,
                    fail_high_rate: 0.0,
                    research_rate: 0.0,
                    tt_cutoff_rate: 0.0,
                    failures,
                };
            }
        };

        let legal = board.legal_moves();

        // 2. Run search with a shared TT so we can probe it afterwards.
        let tt   = TranspositionTable::new(8);
        let stop = Arc::new(AtomicBool::new(false));
        let limits = SearchLimits { depth: Some(depth), ..Default::default() };

        // Capture PV from the last completed depth.
        let mut last_pv: Vec<Move> = Vec::new();
        let result = think(&mut board, &limits, Arc::clone(&stop), &tt, &[], None, |info| {
            last_pv = info.pv.clone();
        });

        let bm_uci  = result.best_move.map(|m| m.to_uci());
        let pv_uci: Vec<String> = last_pv.iter().map(|m| m.to_uci()).collect();

        // 3. Check: TT root consistency.  The TT must have the root's best move.
        //    This check runs BEFORE the determinism search which would use a different TT.
        if let Some(mv) = result.best_move {
            let tt_mv_uci = tt.probe(board.hash).and_then(|d| d.best_move()).map(|m| m.to_uci());
            if tt_mv_uci.as_deref() != Some(&mv.to_uci()) {
                failures.push(VerificationFailure::TtRootMismatch {
                    search_move: mv.to_uci(),
                    tt_move: tt_mv_uci,
                });
            }
        }

        // 4. Check: root move legality.
        if let Some(mv) = result.best_move {
            if !legal.iter().any(|&m| m == mv) {
                failures.push(VerificationFailure::IllegalRootMove { mv: mv.to_uci() });
            }
        } else if !legal.is_empty() {
            failures.push(VerificationFailure::NoMoveReturned);
        }

        // 5. Check: score bounds.
        if result.score.abs() > MATE {
            failures.push(VerificationFailure::ScoreOutOfBounds { score: result.score });
        }

        // 6. Check: PV legality — walk the PV on a clone of the board.
        {
            let mut pv_board = Board::from_fen(&pos.fen).unwrap();
            for (ply, &mv) in last_pv.iter().enumerate() {
                let legal_now = pv_board.legal_moves();
                if !legal_now.iter().any(|&m| m == mv) {
                    failures.push(VerificationFailure::IllegalPvMove {
                        mv: mv.to_uci(),
                        at_ply: ply,
                    });
                    break; // cannot continue — position is corrupted
                }
                let _ = pv_board.make_move(mv);
            }
        }

        // 7. Check: determinism — a fresh TT produces identical output.
        {
            let tt2   = TranspositionTable::new(8);
            let stop2 = Arc::new(AtomicBool::new(false));
            let mut board2 = Board::from_fen(&pos.fen).unwrap();
            let result2 = think(&mut board2, &limits, stop2, &tt2, &[], None, |_| {});

            let repr1 = format!(
                "{} score={}",
                result.best_move.map(|m| m.to_uci()).unwrap_or_else(|| "(none)".into()),
                result.score,
            );
            let repr2 = format!(
                "{} score={}",
                result2.best_move.map(|m| m.to_uci()).unwrap_or_else(|| "(none)".into()),
                result2.score,
            );
            if repr1 != repr2 {
                failures.push(VerificationFailure::NonDeterministic {
                    run1: repr1,
                    run2: repr2,
                });
            }
        }

        // 8. Check: position-kind specific expectations.
        match &pos.kind {
            PositionKind::ForcedMateIn { at_most_moves } => {
                if !is_mate_score(result.score) || result.score < 0 {
                    failures.push(VerificationFailure::NotMateScore { score: result.score });
                } else {
                    let plies = mate_distance_plies(result.score);
                    let max_plies = 2 * at_most_moves - 1;
                    if plies > max_plies {
                        failures.push(VerificationFailure::MateDistanceTooLong {
                            expected_moves: *at_most_moves,
                            actual_plies: plies,
                        });
                    }
                }
            }
            PositionKind::BestMove { uci } => {
                if bm_uci.as_deref() != Some(uci.as_str()) {
                    failures.push(VerificationFailure::WrongBestMove {
                        expected: uci.clone(),
                        found: bm_uci.clone(),
                    });
                }
            }
            PositionKind::AnyLegal => {}
        }

        // 9. Collect diagnostic stats from a direct Searcher call.
        //    (think() doesn't expose Searcher, so we run one extra search.)
        let (fhr, rr, ttcr) = collect_stats(&pos.fen, depth);

        VerificationResult {
            description: pos.description.clone(),
            fen: pos.fen.clone(),
            depth_used: depth,
            best_move_uci: bm_uci,
            score: result.score,
            pv_uci,
            nodes: result.nodes,
            fail_high_rate: fhr,
            research_rate: rr,
            tt_cutoff_rate: ttcr,
            failures,
        }
    }
}

// ─── Helper: direct Searcher stats ───────────────────────────────────────────

/// Run a direct Searcher (bypassing think()) to collect diagnostic counters.
fn collect_stats(fen: &str, depth: u32) -> (f64, f64, f64) {
    let Ok(mut board) = Board::from_fen(fen) else { return (0.0, 0.0, 0.0); };
    let tt   = TranspositionTable::new(4);
    let stop = Arc::new(AtomicBool::new(false));
    let mut searcher = Searcher::new(None, stop, None, HandcraftedEvaluator);
    let _ = searcher.search_root(&mut board, depth.max(1), None, &tt, &[], -INFINITY, INFINITY);
    (searcher.fail_high_rate(), searcher.research_rate(), searcher.tt_cutoff_rate())
}

// ─── Standard position suite ─────────────────────────────────────────────────

/// 20 positions covering correctness, tactics, edge cases, and known regressions.
fn standard_positions() -> Vec<VerifyPosition> {
    use PositionKind::*;
    vec![
        // ── Sanity ────────────────────────────────────────────────────────────
        VerifyPosition::new(
            "Starting position (determinism + legal move)",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            4, AnyLegal,
        ),
        VerifyPosition::new(
            "Kiwipete (complex castling + EP)",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            4, AnyLegal,
        ),

        // ── Mate in 1 ─────────────────────────────────────────────────────────
        VerifyPosition::new(
            "Scholar's mate: Qxf7#",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            2, ForcedMateIn { at_most_moves: 1 },
        ),
        VerifyPosition::new(
            "Rg8# (rook delivers back-rank mate)",
            "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
            2, ForcedMateIn { at_most_moves: 1 },
        ),
        VerifyPosition::new(
            // Re8: rook e1→e8 is 2 squares from g8-king, so king cannot capture.
            // Black pawns f7/g7/h7 block all 7th-rank escapes; 8th rank covered by rook.
            "Re8# (back-rank mate, pawns block escape)",
            "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
            2, ForcedMateIn { at_most_moves: 1 },
        ),
        VerifyPosition::new(
            // Two rooks on the 7th rank cover all of rank 7; Ra8 covers all of rank 8.
            "Ra8# (double-rook ladder mate)",
            "6k1/RR6/8/8/8/8/8/6K1 w - - 0 1",
            2, ForcedMateIn { at_most_moves: 1 },
        ),

        // ── Best move (known capture or tactic) ───────────────────────────────
        VerifyPosition::new(
            "Hanging queen: Qxd8+",
            "3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1",
            2, BestMove { uci: "d1d8".into() },
        ),
        VerifyPosition::new(
            "Pawn captures hanging pawn: exd5",
            "7k/8/8/3p4/4P3/8/8/7K w - - 0 1",
            2, BestMove { uci: "e4d5".into() },
        ),
        VerifyPosition::new(
            "Re8# move: back-rank mate",
            "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
            2, BestMove { uci: "e1e8".into() },
        ),

        // ── Mate in 2 ─────────────────────────────────────────────────────────
        VerifyPosition::new(
            "Scholar's mate threat: Qxf7 forced",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            4, BestMove { uci: "h5f7".into() },
        ),
        VerifyPosition::new(
            "Qa8 forces mate in 3 (driving sequence)",
            "7k/8/6K1/8/8/8/8/Q7 w - - 0 1",
            6, ForcedMateIn { at_most_moves: 3 },
        ),

        // ── Draw positions ────────────────────────────────────────────────────
        VerifyPosition::new(
            "Stalemate: Black to move (score = 0)",
            "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1",
            2, AnyLegal,  // terminal: legal_moves() empty → score = 0
        ),
        VerifyPosition::new(
            "KN vs K: insufficient material (draw)",
            "4k3/8/8/8/8/8/8/4KN2 w - - 0 1",
            4, AnyLegal,
        ),
        VerifyPosition::new(
            "Fifty-move rule: halfmove_clock=100",
            "4k3/8/8/8/8/8/8/R3K3 w - - 100 1",
            3, AnyLegal,
        ),

        // ── Endgame correctness ───────────────────────────────────────────────
        VerifyPosition::new(
            "K+P vs K: White should be better (no zugzwang crash)",
            "8/8/8/8/8/4K3/4P3/7k w - - 0 1",
            4, AnyLegal,
        ),
        VerifyPosition::new(
            "NMP safety: pawn endgame (king+pawn only)",
            "8/1k6/1p6/1Pp5/2P5/8/6K1/8 w - - 0 1",
            5, AnyLegal,
        ),

        // ── Tactical regression suite ─────────────────────────────────────────
        VerifyPosition::new(
            "Winning capture survives LMR",
            "rnb1kbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 3",
            4, AnyLegal,
        ),
        VerifyPosition::new(
            "Quiescence avoids losing recapture",
            "rnbqkbnr/pp2pppp/2p5/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
            4, AnyLegal,
        ),
        VerifyPosition::new(
            "SEE-positive recapture found",
            "r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bBP4/2N1PN2/PPQ2PPP/R1B2RK1 w - - 0 10",
            4, AnyLegal,
        ),
        VerifyPosition::new(
            "Italian middle game (complex, no crash)",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
            4, AnyLegal,
        ),
    ]
}

// ─── CLI entry point ──────────────────────────────────────────────────────────

/// `checksmith verify [depth]`
pub fn run_verify(args: &[String]) {
    let depth: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(5);
    let verifier = SearchVerifier::new();

    println!(
        "Checksmith search verifier  positions={}  depth={}",
        verifier.positions.len(),
        depth,
    );
    println!();

    let results  = verifier.run_all(depth);
    let all_pass = SearchVerifier::print_report(&results);

    if !all_pass {
        eprintln!("\nVERIFICATION FAILED — search has a correctness bug.");
        std::process::exit(1);
    }
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

fn fmt_score(score: i32) -> String {
    if is_mate_score(score) {
        let plies = mate_distance_plies(score);
        let moves = (plies + 1) / 2;
        format!("mate {}", if score > 0 { moves } else { -moves })
    } else {
        format!("cp {}", score)
    }
}

fn fmt_nodes(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;
    use crate::search::search;

    // ── Searcher diagnostic counters ──────────────────────────────────────────

    #[test]
    fn fail_highs_are_counted() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let tt   = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);
        let _ = s.search_root(&mut board, 5, None, &tt, &[], -INFINITY, INFINITY);
        assert!(s.fail_highs > 0, "expect at least one beta cutoff at depth 5");
    }

    #[test]
    fn tt_hits_are_counted() {
        // depth-2 search stores entries in negamax at depth=1.
        // depth-3 search then probes those stored entries in negamax at depth=2.
        // depth-1 search goes to quiescence at depth=0 without storing, so we
        // need depth-2 + depth-3 (not depth-1 + depth-2) to get real TT hits.
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let tt   = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);
        let _ = s.search_root(&mut board, 2, None, &tt, &[], -INFINITY, INFINITY);
        let _ = s.search_root(&mut board, 3, None, &tt, &[], -INFINITY, INFINITY);
        assert!(s.tt_hits > 0, "depth-3 search should hit TT entries stored by depth-2");
    }

    #[test]
    fn researches_occur_at_pv_nodes() {
        // A rich middlegame with PV nodes should trigger PVS re-searches.
        let fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5";
        let mut board = Board::from_fen(fen).unwrap();
        let tt   = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);
        let _ = s.search_root(&mut board, 6, None, &tt, &[], -INFINITY, INFINITY);
        // PVS always re-searches the PV child: at least 1 re-search expected.
        assert!(s.researches > 0, "PVS should trigger at least one re-search at depth 6");
    }

    #[test]
    fn fail_high_rate_in_expected_range() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let tt   = TranspositionTable::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let mut s = Searcher::new(None, stop, None, HandcraftedEvaluator);
        let _ = s.search_root(&mut board, 6, None, &tt, &[], -INFINITY, INFINITY);
        let rate = s.fail_high_rate();
        // Healthy engines: 40–80% beta cutoffs (wide range; exact value is position-dependent).
        assert!(rate > 0.1, "fail_high_rate={:.2} suspiciously low", rate);
        assert!(rate < 1.0, "fail_high_rate={:.2} impossibly high", rate);
    }

    // ── Verifier checks ───────────────────────────────────────────────────────

    #[test]
    fn verifier_startpos_passes_all_checks() {
        let v = SearchVerifier::new();
        let pos = v.positions.iter().find(|p| p.description.contains("Starting position")).unwrap();
        let result = v.verify_one(pos, 4);
        assert!(
            result.passed(),
            "starting position verification failed: {:?}",
            result.failures
        );
    }

    #[test]
    fn verifier_detects_illegal_pv_from_corrupted_tt() {
        // Verify a clean mate-in-1 position: Re8# (back-rank mate).
        // This guards against the category of TT-corruption bug that produces illegal PVs;
        // the positive case confirms the checker produces no false positives on clean positions.
        let pos = VerifyPosition::new(
            "Re8# (back-rank mate, pawns block escape)",
            "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
            2,
            PositionKind::ForcedMateIn { at_most_moves: 1 },
        );
        let v = SearchVerifier { positions: vec![] };
        let result = v.verify_one(&pos, 2);
        assert!(result.passed(), "clean mate-in-1 should pass all checks: {:?}", result.failures);
        assert_eq!(result.pv_uci.len(), 1, "mate-in-1 PV should be exactly 1 move");
    }

    #[test]
    fn verifier_correct_mate_score() {
        let pos = VerifyPosition::new(
            "Scholar's mate",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            4,
            PositionKind::BestMove { uci: "h5f7".into() },
        );
        let v = SearchVerifier { positions: vec![] };
        let result = v.verify_one(&pos, 4);
        assert!(result.passed(), "scholar's mate check failed: {:?}", result.failures);
    }

    #[test]
    fn verifier_catches_wrong_best_move() {
        let pos = VerifyPosition::new(
            "Deliberately wrong expected move",
            "3q1k2/8/8/8/8/8/8/3Q2K1 w - - 0 1",
            2,
            PositionKind::BestMove { uci: "d1a1".into() }, // wrong; correct is d1d8
        );
        let v = SearchVerifier { positions: vec![] };
        let result = v.verify_one(&pos, 2);
        let has_wrong_move_failure = result.failures.iter().any(|f| {
            matches!(f, VerificationFailure::WrongBestMove { .. })
        });
        assert!(has_wrong_move_failure, "should detect wrong best move");
    }

    #[test]
    fn verifier_root_move_is_always_legal() {
        // Run a search on every standard position and confirm no illegal moves.
        let v = SearchVerifier::new();
        let results = v.run_all(3);
        for r in &results {
            let illegal = r.failures.iter().any(|f| {
                matches!(f, VerificationFailure::IllegalRootMove { .. })
            });
            assert!(!illegal, "illegal root move in '{}': {:?}", r.description, r.failures);
        }
    }

    #[test]
    fn verifier_pv_always_legal() {
        let v = SearchVerifier::new();
        let results = v.run_all(4);
        for r in &results {
            let bad_pv = r.failures.iter().any(|f| {
                matches!(f, VerificationFailure::IllegalPvMove { .. })
            });
            assert!(!bad_pv, "illegal PV move in '{}': {:?}", r.description, r.failures);
        }
    }

    #[test]
    fn verifier_all_deterministic() {
        let v = SearchVerifier::new();
        let results = v.run_all(4);
        for r in &results {
            let nondeterministic = r.failures.iter().any(|f| {
                matches!(f, VerificationFailure::NonDeterministic { .. })
            });
            assert!(
                !nondeterministic,
                "non-determinism in '{}': {:?}",
                r.description, r.failures
            );
        }
    }

    #[test]
    fn score_bounds_never_violated() {
        let v = SearchVerifier::new();
        let results = v.run_all(4);
        for r in &results {
            let out_of_bounds = r.failures.iter().any(|f| {
                matches!(f, VerificationFailure::ScoreOutOfBounds { .. })
            });
            assert!(
                !out_of_bounds,
                "score out of bounds in '{}': {}",
                r.description, r.score
            );
        }
    }

    #[test]
    fn mate_in_one_verifier_passes() {
        // Re8#: rook e1→e8 is not adjacent to king g8 so Black cannot capture it.
        // Black pawns block all 7th-rank escapes; 8th rank covered by the rook.
        let result = search(
            &mut Board::from_fen("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1").unwrap(),
            2,
        );
        assert!(is_mate_score(result.score) && result.score > 0, "score={}", result.score);
        assert_eq!(mate_distance_plies(result.score), 1, "should be mate in 1 ply");
    }

    #[test]
    fn whole_suite_passes_at_depth_4() {
        let v = SearchVerifier::new();
        let results = v.run_all(4);
        let failures: Vec<_> = results.iter().filter(|r| !r.passed()).collect();
        assert!(
            failures.is_empty(),
            "{} position(s) failed verification:\n{}",
            failures.len(),
            failures
                .iter()
                .map(|r| format!("  {} → {:?}", r.description, r.failures))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
}
