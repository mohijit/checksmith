//! Engine testing: self-play matches, illegal-move detection, and regression checks.
//!
//! ## Architecture
//!
//! A **match** consists of `N` games played from fixed opening positions
//! (see [`opening`]).  Both sides of each game use the same internal engine
//! instance, exercising correctness rather than relative strength.  Plug the
//! resulting [`MatchResult`] into [`sprt::SprtTest`] to decide whether a code
//! change improves or regresses the engine.
//!
//! ## Game termination
//!
//! Games end by any of:
//! * Checkmate or stalemate (detected via empty legal-move list + check test).
//! * 50-move rule (`halfmove_clock ≥ 100`).
//! * Threefold-repetition (same Zobrist hash appears 3 times in game history).
//! * Score-based draw adjudication (score stays within ±`adj_threshold` cp for
//!   `adj_min_moves` consecutive half-moves).
//! * Hard move-count cap (`max_moves` half-moves) — safety net for non-terminating
//!   positions.
//!
//! ## Illegal-move and crash detection
//!
//! After every engine call the returned move is checked against `board.legal_moves()`.
//! A move absent from that list triggers `AbortReason::IllegalMove`.  A missing
//! move when legal moves exist triggers `AbortReason::NoMoveFound`.  Both abort
//! the current game and are counted separately in [`MatchResult::aborts`].
//!
//! ## Cute Chess CLI (external testing)
//!
//! For process-level testing against a compiled baseline binary, use Cute Chess CLI:
//!
//! ```sh
//! cutechess-cli \
//!   -engine cmd=./target/release/checksmith name=dev \
//!   -engine cmd=./target/release/checksmith_base name=base \
//!   -each  proto=uci tc=10+0.1 \
//!   -games 2000 -concurrency 4 \
//!   -openings file=openings.epd format=epd order=random \
//!   -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
//!   -ratinginterval 200
//! ```
//!
//! Cute Chess handles process crashes, time forfeit, and pentanomial SPRT
//! automatically.  The internal [`sprt`] module is for in-process match analysis
//! where you control the game loop directly.

pub mod opening;
pub mod sprt;

use crate::board::{Board, Color};
use crate::movegen::Move;
use crate::search::{think, SearchLimits, TranspositionTable};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// ─── Time control ────────────────────────────────────────────────────────────

/// How long each side may think per half-move.
#[derive(Clone, Debug)]
pub enum TimeControl {
    /// Fixed search depth (deterministic; preferred for regression testing).
    FixedDepth(u32),
    /// Fixed thinking time in milliseconds per move (closer to tournament play).
    MoveTime(u64),
}

impl Default for TimeControl {
    fn default() -> Self {
        TimeControl::FixedDepth(5)
    }
}

// ─── Game outcome types ───────────────────────────────────────────────────────

/// Why a game was drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawReason {
    Stalemate,
    FiftyMoveDraw,
    Repetition,
    /// Score-based draw adjudication fired before natural termination.
    Adjudicated,
}

/// Why a game was aborted (not a normal result).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbortReason {
    /// The engine returned a move not found in `board.legal_moves()`.
    IllegalMove,
    /// The engine returned `None` despite legal moves being available.
    NoMoveFound,
}

/// The final result of a single game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameOutcome {
    /// The side playing White delivered checkmate.
    WhiteWins,
    /// The side playing Black delivered checkmate.
    BlackWins,
    Draw(DrawReason),
    /// Game ended abnormally (engine bug detected).
    Aborted(AbortReason),
}

impl std::fmt::Display for GameOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GameOutcome::WhiteWins              => write!(f, "White wins"),
            GameOutcome::BlackWins              => write!(f, "Black wins"),
            GameOutcome::Draw(DrawReason::Stalemate)    => write!(f, "Draw (stalemate)"),
            GameOutcome::Draw(DrawReason::FiftyMoveDraw)=> write!(f, "Draw (50-move)"),
            GameOutcome::Draw(DrawReason::Repetition)   => write!(f, "Draw (repetition)"),
            GameOutcome::Draw(DrawReason::Adjudicated)  => write!(f, "Draw (adjudicated)"),
            GameOutcome::Aborted(AbortReason::IllegalMove)  => write!(f, "ABORT: illegal move"),
            GameOutcome::Aborted(AbortReason::NoMoveFound)  => write!(f, "ABORT: no move"),
        }
    }
}

// ─── GameRecord ───────────────────────────────────────────────────────────────

/// Complete record of a single played game.
pub struct GameRecord {
    /// Moves played in UCI notation (may be partial for aborted games).
    pub moves: Vec<Move>,
    pub outcome: GameOutcome,
}

// ─── MatchConfig ─────────────────────────────────────────────────────────────

/// Configuration for a multi-game self-play match.
pub struct MatchConfig {
    /// Total number of games to play.
    pub games: usize,
    /// Time/depth budget per half-move.
    pub tc: TimeControl,
    /// Index of the first opening (wraps via `% OPENINGS.len()`).
    pub opening_start: usize,
    /// Absolute centipawn score below which the position is considered drawish.
    pub adj_threshold: i32,
    /// Number of consecutive half-moves below `adj_threshold` before adjudication.
    pub adj_min_moves: usize,
    /// Hard cap on half-moves per game (prevents infinite games).
    pub max_moves: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        MatchConfig {
            games:         20,
            tc:            TimeControl::FixedDepth(5),
            opening_start: 0,
            adj_threshold: 10,   // |score| < 10 cp → near equal
            adj_min_moves: 12,   // 6 full moves of near-equal play → draw
            max_moves:     400,  // 200 full moves hard cap
        }
    }
}

// ─── MatchResult ──────────────────────────────────────────────────────────────

/// Aggregate statistics from a completed match.
#[derive(Clone, Debug, Default)]
pub struct MatchResult {
    /// Games won by the side playing White.
    pub white_wins: u64,
    /// Games drawn for any reason.
    pub draws:      u64,
    /// Games won by the side playing Black.
    pub black_wins: u64,
    /// Games that ended due to an engine bug (illegal move or no move).
    pub aborts:     u64,
}

impl MatchResult {
    /// Total games recorded (including aborts).
    pub fn total(&self) -> u64 {
        self.white_wins + self.draws + self.black_wins + self.aborts
    }

    /// White's score as a fraction in [0, 1], excluding aborted games.
    ///
    /// Returns 0.5 when no completed games have been recorded yet.
    pub fn white_score(&self) -> f64 {
        let n = self.white_wins + self.draws + self.black_wins;
        if n == 0 {
            return 0.5;
        }
        (self.white_wins as f64 + self.draws as f64 / 2.0) / n as f64
    }
}

impl std::fmt::Display for MatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "W:{} D:{} L:{}  aborts:{}  score:{:.1}%",
            self.white_wins,
            self.draws,
            self.black_wins,
            self.aborts,
            self.white_score() * 100.0,
        )
    }
}

// ─── play_game ────────────────────────────────────────────────────────────────

/// Play one game from `opening_fen` using `config` and return the record.
///
/// The transposition table is shared for the whole game (same as in UCI play)
/// but a new generation is started before each move search.
pub fn play_game(opening_fen: &str, config: &MatchConfig) -> GameRecord {
    let mut board = match Board::from_fen(opening_fen) {
        Ok(b)  => b,
        Err(_) => {
            return GameRecord {
                moves:   Vec::new(),
                outcome: GameOutcome::Aborted(AbortReason::NoMoveFound),
            };
        }
    };

    let mut moves:   Vec<Move> = Vec::new();
    // game_history: all hashes visited so far, NOT including the current root.
    // Matches the convention expected by `think()`.
    let mut game_history: Vec<u64> = Vec::new();
    let mut adj_streak: usize = 0;
    let tt = TranspositionTable::new(4);

    loop {
        let legal = board.legal_moves();

        // ── Terminal: no legal moves ─────────────────────────────────────────
        if legal.is_empty() {
            let outcome = if board.is_in_check() {
                if board.side_to_move == Color::White {
                    GameOutcome::BlackWins
                } else {
                    GameOutcome::WhiteWins
                }
            } else {
                GameOutcome::Draw(DrawReason::Stalemate)
            };
            return GameRecord { moves, outcome };
        }

        // ── 50-move rule ──────────────────────────────────────────────────────
        if board.halfmove_clock >= 100 {
            return GameRecord { moves, outcome: GameOutcome::Draw(DrawReason::FiftyMoveDraw) };
        }

        // ── Threefold repetition ──────────────────────────────────────────────
        // game_history holds all prior positions (NOT the current root).
        // If the current position appeared ≥ 2 times in history, adding this
        // occurrence makes 3 total → draw.
        let prior_reps = game_history.iter().filter(|&&h| h == board.hash).count();
        if prior_reps >= 2 {
            return GameRecord { moves, outcome: GameOutcome::Draw(DrawReason::Repetition) };
        }

        // ── Max-moves safety cap ──────────────────────────────────────────────
        if moves.len() >= config.max_moves {
            return GameRecord { moves, outcome: GameOutcome::Draw(DrawReason::Adjudicated) };
        }

        // ── Ask the engine for a move ─────────────────────────────────────────
        tt.new_generation();
        let (mv_opt, score) = engine_move(&mut board, &config.tc, &tt, &game_history);

        let mv = match mv_opt {
            None => {
                return GameRecord {
                    moves,
                    outcome: GameOutcome::Aborted(AbortReason::NoMoveFound),
                };
            }
            Some(m) => m,
        };

        // ── Legality verification (crash/bug detection) ───────────────────────
        if !legal.iter().any(|&m| m == mv) {
            return GameRecord {
                moves,
                outcome: GameOutcome::Aborted(AbortReason::IllegalMove),
            };
        }

        // ── Score-based draw adjudication ────────────────────────────────────
        if score.abs() < config.adj_threshold {
            adj_streak += 1;
            if adj_streak >= config.adj_min_moves {
                moves.push(mv);
                return GameRecord {
                    moves,
                    outcome: GameOutcome::Draw(DrawReason::Adjudicated),
                };
            }
        } else {
            adj_streak = 0;
        }

        // ── Apply move ────────────────────────────────────────────────────────
        game_history.push(board.hash); // record current root BEFORE it changes
        let _ = board.make_move(mv);
        moves.push(mv);
    }
}

/// Invoke the engine for one half-move.  Returns `(move, score)`.
fn engine_move(
    board: &mut Board,
    tc: &TimeControl,
    tt: &TranspositionTable,
    game_history: &[u64],
) -> (Option<Move>, i32) {
    let stop = Arc::new(AtomicBool::new(false));
    let limits = match tc {
        TimeControl::FixedDepth(d) => SearchLimits {
            depth: Some(*d),
            ..Default::default()
        },
        TimeControl::MoveTime(ms) => SearchLimits {
            movetime: Some(*ms),
            ..Default::default()
        },
    };
    // Silence per-depth info callbacks — self-play doesn't emit UCI output.
    let result = think(board, &limits, stop, tt, game_history, None, |_| {});
    (result.best_move, result.score)
}

// ─── play_match ──────────────────────────────────────────────────────────────

/// Play a match of `config.games` games, calling `on_game` after each one.
///
/// `on_game(game_index, record, running_totals)` — use it to print progress or
/// update a GUI.
pub fn play_match(
    config: &MatchConfig,
    mut on_game: impl FnMut(usize, &GameRecord, &MatchResult),
) -> MatchResult {
    let mut result = MatchResult::default();
    for i in 0..config.games {
        let fen = opening::pick(config.opening_start + i);
        let record = play_game(fen, config);
        match record.outcome {
            GameOutcome::WhiteWins   => result.white_wins += 1,
            GameOutcome::BlackWins   => result.black_wins += 1,
            GameOutcome::Draw(_)     => result.draws += 1,
            GameOutcome::Aborted(_)  => result.aborts += 1,
        }
        on_game(i, &record, &result);
    }
    result
}

// ─── CLI entry point ─────────────────────────────────────────────────────────

/// `checksmith match [games] [depth]`
///
/// Runs an internal self-play match and prints per-game progress plus a final
/// SPRT summary.
pub fn run_match(args: &[String]) {
    let games: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
    let depth: u32   = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);

    let config = MatchConfig {
        games,
        tc: TimeControl::FixedDepth(depth),
        ..Default::default()
    };

    println!(
        "Checksmith self-play match  games={}  depth={}  openings={}",
        games,
        depth,
        opening::OPENINGS.len(),
    );
    println!();

    let mut sprt_test = sprt::SprtTest::new(sprt::SprtConfig::default());

    let result = play_match(&config, |i, record, running| {
        // Update SPRT incrementally.
        match record.outcome {
            GameOutcome::WhiteWins => sprt_test.update(1, 0, 0),
            GameOutcome::BlackWins => sprt_test.update(0, 0, 1),
            GameOutcome::Draw(_)   => sprt_test.update(0, 1, 0),
            GameOutcome::Aborted(_)=> {}
        }
        println!(
            "  [{:3}/{}] {:>3} moves  {}  ({})",
            i + 1,
            games,
            record.moves.len(),
            record.outcome,
            running,
        );
    });

    println!();
    println!("======================================");
    println!("Match result  {}", result);
    println!("{}", sprt_test);
    println!("======================================");

    if result.aborts > 0 {
        eprintln!("WARNING: {} game(s) aborted — engine bug detected!", result.aborts);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;

    #[test]
    fn game_from_startpos_terminates() {
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(2),
            max_moves: 400,
            ..Default::default()
        };
        let record = play_game(STARTING_FEN, &config);
        assert!(
            !matches!(record.outcome, GameOutcome::Aborted(_)),
            "game should not abort: {:?}",
            record.outcome
        );
        assert!(!record.moves.is_empty(), "at least one move must be played");
    }

    #[test]
    fn fifty_move_draw_detected() {
        // halfmove_clock == 100 → immediate 50-move draw before the first search.
        let fen = "8/8/4k3/8/8/4K3/8/8 w - - 100 50";
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            max_moves: 400,
            ..Default::default()
        };
        let record = play_game(fen, &config);
        assert_eq!(record.outcome, GameOutcome::Draw(DrawReason::FiftyMoveDraw));
        assert_eq!(record.moves.len(), 0, "no moves should be played before 50-move detection");
    }

    #[test]
    fn adjudication_fires_when_threshold_maxed() {
        // Set threshold to maximum so *every* move triggers the score condition.
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            adj_threshold: i32::MAX,
            adj_min_moves: 2,
            max_moves: 400,
            ..Default::default()
        };
        let record = play_game(STARTING_FEN, &config);
        assert_eq!(
            record.outcome,
            GameOutcome::Draw(DrawReason::Adjudicated),
            "max threshold + 2 moves should adjudicate"
        );
        // Adjudication fires after adj_min_moves half-moves, so at least 2 moves.
        assert!(record.moves.len() >= 2);
    }

    #[test]
    fn max_moves_cap_fires() {
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            adj_threshold: 0,   // disable score adjudication
            adj_min_moves: 9999,
            max_moves: 4,       // very small cap
            ..Default::default()
        };
        let record = play_game(STARTING_FEN, &config);
        // With no score adjudication and 4-move cap, should hit the cap (or a
        // natural terminal before it).
        assert!(record.moves.len() <= 4, "move count must respect max_moves");
    }

    #[test]
    fn match_3_games_no_aborts() {
        let config = MatchConfig {
            games: 3,
            tc: TimeControl::FixedDepth(2),
            max_moves: 200,
            ..Default::default()
        };
        let result = play_match(&config, |_, _, _| {});
        assert_eq!(result.aborts, 0, "self-play should produce no illegal moves");
        assert_eq!(result.total(), 3);
    }

    #[test]
    fn match_result_display() {
        let r = MatchResult {
            white_wins: 4,
            draws: 6,
            black_wins: 3,
            aborts: 1,
        };
        let s = r.to_string();
        assert!(s.contains("W:4"), "must show white wins");
        assert!(s.contains("D:6"), "must show draws");
        assert!(s.contains("aborts:1"), "must show aborts");
    }

    #[test]
    fn stalemate_detected() {
        // Stalemate: Black king on a8 with White queen on b6, White king on c6,
        // White to… wait, we need Black to move and be stalemated.
        // Black king on a8, White queen on b6, White king on b8 — but kings can't be adjacent.
        // Classic stalemate: Black king a8, White queen c7, White king c6, Black to move.
        let fen = "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1";
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            max_moves: 400,
            ..Default::default()
        };
        let record = play_game(fen, &config);
        assert_eq!(record.outcome, GameOutcome::Draw(DrawReason::Stalemate));
        assert_eq!(record.moves.len(), 0);
    }

    #[test]
    fn checkmate_detected() {
        // Black king on g8, White rook on e8, White king on g6.
        // Black to move, in check from Re8 with no escape squares → White wins.
        let fen = "4R1k1/6pp/6K1/8/8/8/8/8 b - - 0 1";
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            max_moves: 400,
            ..Default::default()
        };
        let record = play_game(fen, &config);
        assert_eq!(record.outcome, GameOutcome::WhiteWins);
        assert_eq!(record.moves.len(), 0);
    }

    #[test]
    fn on_game_callback_invoked_for_each_game() {
        let config = MatchConfig {
            games: 5,
            tc: TimeControl::FixedDepth(1),
            max_moves: 100,
            ..Default::default()
        };
        let mut call_count = 0usize;
        play_match(&config, |_, _, _| call_count += 1);
        assert_eq!(call_count, 5);
    }

    #[test]
    fn game_outcome_display() {
        assert_eq!(GameOutcome::WhiteWins.to_string(), "White wins");
        assert_eq!(GameOutcome::Draw(DrawReason::Stalemate).to_string(), "Draw (stalemate)");
        assert_eq!(GameOutcome::Draw(DrawReason::FiftyMoveDraw).to_string(), "Draw (50-move)");
        assert_eq!(GameOutcome::Aborted(AbortReason::IllegalMove).to_string(), "ABORT: illegal move");
    }
}
