//! Engine testing: self-play matches, SPRT, and external-engine benchmarking.
//!
//! ## Architecture
//!
//! A **match** is a sequence of games between two [`ChessEngine`] instances.
//! The built-in [`InternalEngine`] uses the current binary's own search.
//! [`ExternalEngine`] wraps a UCI subprocess, enabling `dev` vs `baseline`
//! testing without Cute Chess CLI.
//!
//! ## SPRT auto-stop
//!
//! When `MatchConfig::sprt` is `Some(config)`, the match stops as soon as the
//! log-likelihood ratio crosses the accept-H0 or accept-H1 boundary — saving
//! the time required to play the remaining fixed game count.
//!
//! ## Paired games
//!
//! With `MatchConfig::paired = true`, each opening is played **twice**: once
//! with the primary engine as White and once as Black.  Results are recorded
//! from the primary engine's perspective so opening-color bias cancels out.
//!
//! ## EPD openings
//!
//! `OpeningSource::Epd(fens)` replaces the 24 hardcoded openings with any
//! list of FEN strings, typically loaded from a standard EPD test file.
//!
//! ## Save / Resume
//!
//! Set `MatchConfig::save_file` to a path.  After each game a single line
//! (`W`, `D`, or `L` from the primary engine's perspective) is appended.
//! On the next run with the same `--resume` file, existing lines are counted
//! and the match continues from that game index.
//!
//! ## Cute Chess CLI (external testing)
//!
//! For process-level SPRT against any UCI engine, Cute Chess CLI remains an
//! option and requires no changes to this codebase:
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

pub mod opening;
pub mod sprt;
pub mod uci_process;

use crate::board::{Board, Color};
use crate::movegen::Move;
use crate::search::{think, SearchLimits, TranspositionTable};
use sprt::{SprtConfig, SprtOutcome, SprtTest};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use uci_process::UciProcess;

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

// ─── Opening source ───────────────────────────────────────────────────────────

/// Where to draw opening positions from.
#[derive(Clone, Debug, Default)]
pub enum OpeningSource {
    /// Use the 24 built-in hardcoded openings (default, backward-compatible).
    #[default]
    Internal,
    /// Use positions from a loaded EPD file (FEN strings only; ops ignored).
    Epd(Vec<String>),
}

impl OpeningSource {
    /// Pick the opening FEN for game index `i` (wraps around).
    pub fn pick(&self, i: usize) -> &str {
        match self {
            OpeningSource::Internal => opening::pick(i),
            OpeningSource::Epd(fens) if !fens.is_empty() => &fens[i % fens.len()],
            OpeningSource::Epd(_)   => opening::pick(i),
        }
    }

    /// Load a source from an EPD file.  Returns `Internal` if the file cannot
    /// be read or contains no valid positions.
    pub fn from_epd_file(path: &str) -> Self {
        let fens = load_epd_fens(path);
        if fens.is_empty() {
            eprintln!("WARNING: no valid positions found in {:?}; using internal openings", path);
            OpeningSource::Internal
        } else {
            OpeningSource::Epd(fens)
        }
    }

    pub fn len(&self) -> usize {
        match self {
            OpeningSource::Internal   => opening::OPENINGS.len(),
            OpeningSource::Epd(fens) => fens.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Read an EPD file and extract the FEN part of each position (first 4 + "0 1").
fn load_epd_fens(path: &str) -> Vec<String> {
    let file = match File::open(path) {
        Ok(f)  => f,
        Err(e) => { eprintln!("cannot open EPD file '{}': {}", path, e); return Vec::new(); }
    };
    BufReader::new(file)
        .lines()
        .filter_map(|l| l.ok())
        .filter_map(|line| {
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with('#') { return None; }
            // Take first 4 space-separated tokens (board, side, castling, ep).
            let mut tokens = line.splitn(5, ' ');
            let board    = tokens.next()?;
            let side     = tokens.next()?;
            let castling = tokens.next()?;
            let ep       = tokens.next()?;
            Some(format!("{} {} {} {} 0 1", board, side, castling, ep))
        })
        .collect()
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
            GameOutcome::WhiteWins                               => write!(f, "White wins"),
            GameOutcome::BlackWins                               => write!(f, "Black wins"),
            GameOutcome::Draw(DrawReason::Stalemate)             => write!(f, "Draw (stalemate)"),
            GameOutcome::Draw(DrawReason::FiftyMoveDraw)         => write!(f, "Draw (50-move)"),
            GameOutcome::Draw(DrawReason::Repetition)            => write!(f, "Draw (repetition)"),
            GameOutcome::Draw(DrawReason::Adjudicated)           => write!(f, "Draw (adjudicated)"),
            GameOutcome::Aborted(AbortReason::IllegalMove)       => write!(f, "ABORT: illegal move"),
            GameOutcome::Aborted(AbortReason::NoMoveFound)       => write!(f, "ABORT: no move"),
        }
    }
}

// ─── GameRecord ───────────────────────────────────────────────────────────────

/// Complete record of a single played game.
pub struct GameRecord {
    /// Moves played in UCI notation (may be partial for aborted games).
    pub moves:   Vec<Move>,
    pub outcome: GameOutcome,
}

// ─── MatchConfig ─────────────────────────────────────────────────────────────

/// Configuration for a multi-game match.
pub struct MatchConfig {
    /// Total number of games to play (upper bound; SPRT may stop earlier).
    pub games: usize,
    /// Time/depth budget per half-move.
    pub tc: TimeControl,
    /// Opening source.  Defaults to the built-in 24 positions.
    pub openings: OpeningSource,
    /// Play each opening twice (colors swapped) to reduce color bias.
    /// When `true`, `games` should be even.
    pub paired: bool,
    /// If `Some`, stop the match as soon as the LLR crosses a boundary.
    pub sprt: Option<SprtConfig>,
    /// Absolute centipawn score below which the position is considered drawish.
    pub adj_threshold: i32,
    /// Number of consecutive half-moves below `adj_threshold` before adjudication.
    pub adj_min_moves: usize,
    /// Hard cap on half-moves per game (prevents infinite games).
    pub max_moves: usize,
    /// If set, append one `W`/`D`/`L` line per game to this file.
    pub save_file: Option<String>,
}

impl Default for MatchConfig {
    fn default() -> Self {
        MatchConfig {
            games:         20,
            tc:            TimeControl::FixedDepth(5),
            openings:      OpeningSource::Internal,
            paired:        false,
            sprt:          None,
            adj_threshold: 10,
            adj_min_moves: 12,
            max_moves:     400,
            save_file:     None,
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
        if n == 0 { return 0.5; }
        (self.white_wins as f64 + self.draws as f64 / 2.0) / n as f64
    }
}

impl std::fmt::Display for MatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "W:{} D:{} L:{}  aborts:{}  score:{:.1}%",
            self.white_wins, self.draws, self.black_wins,
            self.aborts,
            self.white_score() * 100.0,
        )
    }
}

// ─── ChessEngine trait ────────────────────────────────────────────────────────

/// Abstraction over the internal engine and external UCI subprocesses.
pub trait ChessEngine {
    /// Called at the start of each game with the opening FEN.
    fn new_game(&mut self, opening_fen: &str);

    /// Return the best move and its centipawn score for the current position.
    ///
    /// `moves_played` — all moves made from the opening FEN so far (for
    /// position reconstruction in external engines).
    /// `game_history` — Zobrist hashes for threefold-repetition detection
    /// (used by the internal engine only).
    fn get_move(
        &mut self,
        board:        &mut Board,
        moves_played: &[Move],
        tc:           &TimeControl,
        game_history: &[u64],
    ) -> (Option<Move>, i32);
}

// ─── InternalEngine ───────────────────────────────────────────────────────────

/// The current binary's own search, used as one side in a game.
pub struct InternalEngine {
    tt: TranspositionTable,
}

impl InternalEngine {
    pub fn new() -> Self {
        InternalEngine { tt: TranspositionTable::new(4) }
    }
}

impl Default for InternalEngine {
    fn default() -> Self { Self::new() }
}

impl ChessEngine for InternalEngine {
    fn new_game(&mut self, _opening_fen: &str) {
        self.tt.new_generation();
    }

    fn get_move(
        &mut self,
        board:        &mut Board,
        _moves_played: &[Move],
        tc:            &TimeControl,
        game_history:  &[u64],
    ) -> (Option<Move>, i32) {
        let stop   = Arc::new(AtomicBool::new(false));
        let limits = tc_to_limits(tc);
        self.tt.new_generation();
        let result = think(board, &limits, stop, &self.tt, game_history, None, |_| {});
        (result.best_move, result.score)
    }
}

// ─── ExternalEngine ───────────────────────────────────────────────────────────

/// A UCI engine subprocess used as one side in a game.
pub struct ExternalEngine {
    process: UciProcess,
}

impl ExternalEngine {
    /// Launch the engine at `path`.  Returns `Err` if the binary cannot be
    /// started or the UCI handshake fails.
    pub fn launch(path: &str) -> std::io::Result<Self> {
        Ok(ExternalEngine { process: UciProcess::launch(path)? })
    }

    /// Wrap an already-initialised `UciProcess` (e.g. one that has had
    /// setoptions applied before the first game).
    pub fn from_process(process: UciProcess) -> Self {
        ExternalEngine { process }
    }
}

impl ChessEngine for ExternalEngine {
    fn new_game(&mut self, opening_fen: &str) {
        let _ = self.process.new_game(opening_fen);
    }

    fn get_move(
        &mut self,
        board:        &mut Board,
        moves_played: &[Move],
        tc:           &TimeControl,
        _game_history: &[u64],
    ) -> (Option<Move>, i32) {
        self.process.get_move(board, moves_played, tc)
    }
}

// ─── play_game_dyn ────────────────────────────────────────────────────────────

/// Play one game between `white` and `black` from `opening_fen`.
///
/// Results from the *white* engine's perspective (White wins = primary wins).
pub fn play_game_dyn(
    opening_fen: &str,
    config:      &MatchConfig,
    white:       &mut dyn ChessEngine,
    black:       &mut dyn ChessEngine,
) -> GameRecord {
    let mut board = match Board::from_fen(opening_fen) {
        Ok(b)  => b,
        Err(_) => return GameRecord {
            moves:   Vec::new(),
            outcome: GameOutcome::Aborted(AbortReason::NoMoveFound),
        },
    };

    white.new_game(opening_fen);
    black.new_game(opening_fen);

    let mut moves_played: Vec<Move> = Vec::new();
    let mut game_history: Vec<u64>  = Vec::new();
    let mut adj_streak:   usize     = 0;

    loop {
        let legal = board.legal_moves();

        // ── Terminal: no legal moves ─────────────────────────────────────────
        if legal.is_empty() {
            let outcome = if board.is_in_check() {
                if board.side_to_move == Color::White { GameOutcome::BlackWins }
                else                                  { GameOutcome::WhiteWins }
            } else {
                GameOutcome::Draw(DrawReason::Stalemate)
            };
            return GameRecord { moves: moves_played, outcome };
        }

        // ── Draw conditions ───────────────────────────────────────────────────
        if board.halfmove_clock >= 100 {
            return GameRecord { moves: moves_played, outcome: GameOutcome::Draw(DrawReason::FiftyMoveDraw) };
        }
        let prior_reps = game_history.iter().filter(|&&h| h == board.hash).count();
        if prior_reps >= 2 {
            return GameRecord { moves: moves_played, outcome: GameOutcome::Draw(DrawReason::Repetition) };
        }
        if moves_played.len() >= config.max_moves {
            return GameRecord { moves: moves_played, outcome: GameOutcome::Draw(DrawReason::Adjudicated) };
        }

        // ── Engine move ───────────────────────────────────────────────────────
        let engine: &mut dyn ChessEngine = if board.side_to_move == Color::White { white } else { black };
        let (mv_opt, score) = engine.get_move(&mut board, &moves_played, &config.tc, &game_history);

        let mv = match mv_opt {
            None    => return GameRecord { moves: moves_played, outcome: GameOutcome::Aborted(AbortReason::NoMoveFound) },
            Some(m) => m,
        };

        if !legal.iter().any(|&m| m == mv) {
            return GameRecord { moves: moves_played, outcome: GameOutcome::Aborted(AbortReason::IllegalMove) };
        }

        // ── Score-based draw adjudication ────────────────────────────────────
        if score.abs() < config.adj_threshold {
            adj_streak += 1;
            if adj_streak >= config.adj_min_moves {
                moves_played.push(mv);
                return GameRecord { moves: moves_played, outcome: GameOutcome::Draw(DrawReason::Adjudicated) };
            }
        } else {
            adj_streak = 0;
        }

        // ── Apply move ────────────────────────────────────────────────────────
        game_history.push(board.hash);
        let _ = board.make_move(mv);
        moves_played.push(mv);
    }
}

// ─── play_game ───────────────────────────────────────────────────────────────

/// Play one game from `opening_fen` using the internal engine for both sides.
///
/// The transposition table is shared for the whole game (same as in UCI play)
/// but a new generation is started before each move search.
pub fn play_game(opening_fen: &str, config: &MatchConfig) -> GameRecord {
    let mut white = InternalEngine::new();
    let mut black = InternalEngine::new();
    play_game_dyn(opening_fen, config, &mut white, &mut black)
}

// ─── play_match ──────────────────────────────────────────────────────────────

/// Play a self-play match (internal engine vs itself).
///
/// `on_game(game_index, record, running_totals)` is called after each game.
///
/// If `config.sprt` is `Some`, the match stops early once the LLR crosses
/// either boundary.  Results are always from White's perspective (symmetric in
/// pure self-play).
pub fn play_match(
    config:   &MatchConfig,
    mut on_game: impl FnMut(usize, &GameRecord, &MatchResult),
) -> MatchResult {
    let mut result = MatchResult::default();
    let mut sprt   = config.sprt.as_ref().map(|c| SprtTest::new(c.clone()));
    let mut save   = config.save_file.as_deref().and_then(open_save_file);

    let opening_step = if config.paired { 2 } else { 1 };

    let mut game_idx = 0usize;
    let mut opening_idx = 0usize;

    while game_idx < config.games {
        let fen = config.openings.pick(opening_idx);
        let flipped = config.paired && (game_idx % 2 == 1);

        let record = if flipped {
            // Paired second game: run same opening but flip whose result we record.
            let raw = play_game(fen, config);
            // Invert the outcome so SPRT sees it from the primary engine's perspective.
            GameRecord { moves: raw.moves, outcome: flip_outcome(raw.outcome) }
        } else {
            play_game(fen, config)
        };

        match record.outcome {
            GameOutcome::WhiteWins => { result.white_wins += 1; record_result(&mut save, 'W'); sprt_update(&mut sprt, 1, 0, 0); }
            GameOutcome::BlackWins => { result.black_wins += 1; record_result(&mut save, 'L'); sprt_update(&mut sprt, 0, 0, 1); }
            GameOutcome::Draw(_)   => { result.draws      += 1; record_result(&mut save, 'D'); sprt_update(&mut sprt, 0, 1, 0); }
            GameOutcome::Aborted(_)=> { result.aborts     += 1; }
        }

        on_game(game_idx, &record, &result);
        game_idx += 1;

        // Advance opening index after each full pair (or after each game if unpaired).
        if !config.paired || game_idx % 2 == 0 {
            opening_idx += opening_step;
        }

        // SPRT auto-stop.
        if let Some(ref s) = sprt {
            if s.outcome() != SprtOutcome::Continue { break; }
        }
    }

    result
}

// ─── play_match_vs ───────────────────────────────────────────────────────────

/// Play a match between the internal engine and an external UCI engine at
/// `baseline_path`.
///
/// Results are tracked from the **internal** engine's perspective.  Paired
/// games alternate which color the internal engine plays.  SPRT auto-stop
/// fires as soon as a conclusion is reached.
///
/// `on_game(game_index, record, running_totals, sprt_state)` — `sprt_state`
/// is `None` when `config.sprt` is not set.
pub fn play_match_vs(
    config:        &MatchConfig,
    baseline_path: &str,
    mut on_game:   impl FnMut(usize, &GameRecord, &MatchResult, Option<&SprtTest>),
) -> Result<MatchResult, String> {
    let mut baseline = ExternalEngine::launch(baseline_path)
        .map_err(|e| format!("cannot launch baseline '{}': {}", baseline_path, e))?;

    let mut result = MatchResult::default();
    let mut sprt   = config.sprt.as_ref().map(|c| SprtTest::new(c.clone()));
    let mut save   = config.save_file.as_deref().and_then(open_save_file);

    let mut game_idx    = 0usize;
    let mut opening_idx = 0usize;

    // Resume from save file if it exists.
    let resume_count = config.save_file.as_deref()
        .map(count_saved_results)
        .unwrap_or(0);
    if resume_count > 0 {
        eprintln!("Resuming from {} saved game(s).", resume_count);
        // Fast-forward SPRT and result counts from the existing log.
        if let Some(ref path) = config.save_file {
            let (rw, rd, rl) = read_saved_results(path);
            result.white_wins = rw;
            result.draws      = rd;
            result.black_wins = rl;
            sprt_update(&mut sprt, rw, rd, rl);
        }
        game_idx    = resume_count;
        opening_idx = if config.paired { resume_count / 2 } else { resume_count };
    }

    while game_idx < config.games {
        let fen = config.openings.pick(opening_idx);

        // In paired mode: even games → internal=White, odd games → internal=Black.
        let internal_is_white = !config.paired || game_idx % 2 == 0;

        let raw = if internal_is_white {
            let mut internal = InternalEngine::new();
            play_game_dyn(fen, config, &mut internal, &mut baseline)
        } else {
            let mut internal = InternalEngine::new();
            play_game_dyn(fen, config, &mut baseline, &mut internal)
        };

        // Translate outcome to internal engine's perspective.
        let outcome_for_internal = if internal_is_white {
            raw.outcome
        } else {
            flip_outcome(raw.outcome)
        };

        let record = GameRecord { moves: raw.moves, outcome: outcome_for_internal };

        match outcome_for_internal {
            GameOutcome::WhiteWins  => { result.white_wins += 1; record_result(&mut save, 'W'); sprt_update(&mut sprt, 1, 0, 0); }
            GameOutcome::BlackWins  => { result.black_wins += 1; record_result(&mut save, 'L'); sprt_update(&mut sprt, 0, 0, 1); }
            GameOutcome::Draw(_)    => { result.draws      += 1; record_result(&mut save, 'D'); sprt_update(&mut sprt, 0, 1, 0); }
            GameOutcome::Aborted(_) => { result.aborts     += 1; }
        }

        on_game(game_idx, &record, &result, sprt.as_ref());
        game_idx += 1;

        if !config.paired || game_idx % 2 == 0 {
            opening_idx += 1;
        }

        if let Some(ref s) = sprt {
            if s.outcome() != SprtOutcome::Continue { break; }
        }
    }

    Ok(result)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn tc_to_limits(tc: &TimeControl) -> SearchLimits {
    match tc {
        TimeControl::FixedDepth(d) => SearchLimits { depth: Some(*d), ..Default::default() },
        TimeControl::MoveTime(ms)  => SearchLimits { movetime: Some(*ms), ..Default::default() },
    }
}

fn flip_outcome(o: GameOutcome) -> GameOutcome {
    match o {
        GameOutcome::WhiteWins  => GameOutcome::BlackWins,
        GameOutcome::BlackWins  => GameOutcome::WhiteWins,
        other                   => other,
    }
}

fn sprt_update(sprt: &mut Option<SprtTest>, w: u64, d: u64, l: u64) {
    if let Some(s) = sprt { s.update(w, d, l); }
}

fn open_save_file(path: &str) -> Option<File> {
    OpenOptions::new().create(true).append(true).open(path).ok()
}

fn record_result(file: &mut Option<File>, ch: char) {
    if let Some(f) = file {
        let _ = writeln!(f, "{}", ch);
    }
}

fn count_saved_results(path: &str) -> usize {
    File::open(path)
        .map(|f| BufReader::new(f).lines().filter(|l| l.is_ok()).count())
        .unwrap_or(0)
}

fn read_saved_results(path: &str) -> (u64, u64, u64) {
    let (mut w, mut d, mut l) = (0u64, 0u64, 0u64);
    if let Ok(f) = File::open(path) {
        for line in BufReader::new(f).lines().filter_map(|l| l.ok()) {
            match line.trim() {
                "W" => w += 1,
                "D" => d += 1,
                "L" => l += 1,
                _   => {}
            }
        }
    }
    (w, d, l)
}

// ─── CLI entry point ─────────────────────────────────────────────────────────

/// `checksmith match [games] [depth] [--vs path] [--openings file.epd]
///                   [--paired] [--time N] [--save file] [--resume file]`
pub fn run_match(args: &[String]) {
    let mut games:     usize  = 20;
    let mut depth:     u32    = 5;
    let mut time_ms:   u64    = 0;
    let mut use_time           = false;
    let mut vs_path:   Option<String> = None;
    let mut epd_path:  Option<String> = None;
    let mut paired             = false;
    let mut save_path: Option<String> = None;
    let mut resume_path: Option<String> = None;
    let mut sprt_stop          = true; // default on when using --vs

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--vs" => {
                i += 1;
                vs_path = args.get(i).cloned();
            }
            "--openings" => {
                i += 1;
                epd_path = args.get(i).cloned();
            }
            "--paired" => { paired = true; }
            "--time" | "-t" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    time_ms  = v.parse().unwrap_or(1000);
                    use_time = true;
                }
            }
            "--depth" | "-d" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    depth = v.parse().unwrap_or(5);
                }
            }
            "--save" => {
                i += 1;
                save_path = args.get(i).cloned();
            }
            "--resume" => {
                i += 1;
                resume_path = args.get(i).cloned();
            }
            "--no-sprt-stop" => { sprt_stop = false; }
            s => {
                // Positional: first = games, second = depth.
                if let Ok(n) = s.parse::<usize>() {
                    if games == 20 { games = n; }
                } else if let Ok(d) = s.parse::<u32>() {
                    if depth == 5 { depth = d; }
                }
            }
        }
        i += 1;
    }

    let tc = if use_time {
        TimeControl::MoveTime(time_ms)
    } else {
        TimeControl::FixedDepth(depth)
    };

    let openings = match epd_path.as_deref() {
        Some(p) => OpeningSource::from_epd_file(p),
        None    => OpeningSource::Internal,
    };

    // Use the same file for save and resume if --resume is given but --save is not.
    let save_file = save_path.or(resume_path);

    let config = MatchConfig {
        games,
        tc,
        openings,
        paired,
        sprt: if sprt_stop && vs_path.is_some() { Some(SprtConfig::default()) } else { None },
        save_file,
        ..Default::default()
    };

    let tc_label = if use_time { format!("time={}ms", time_ms) } else { format!("depth={}", depth) };

    match vs_path.as_deref() {
        None => {
            // Pure self-play.
            println!(
                "Checksmith self-play  games={}  {}  openings={}{}",
                games, tc_label, config.openings.len(),
                if paired { "  paired" } else { "" },
            );
            println!();

            let mut sprt_display = config.sprt.as_ref().map(|c| SprtTest::new(c.clone()));

            let result = play_match(&config, |idx, record, running| {
                if let Some(ref mut s) = sprt_display {
                    match record.outcome {
                        GameOutcome::WhiteWins => s.update(1, 0, 0),
                        GameOutcome::BlackWins => s.update(0, 0, 1),
                        GameOutcome::Draw(_)   => s.update(0, 1, 0),
                        _                      => {}
                    }
                }
                let sprt_str = sprt_display.as_ref().map(|s| format!("  {}", s)).unwrap_or_default();
                println!(
                    "  [{:3}/{}] {:>3} moves  {}  ({}){}", idx + 1, games,
                    record.moves.len(), record.outcome, running, sprt_str,
                );
            });

            print_match_summary(&result, sprt_display.as_ref());
        }
        Some(path) => {
            // vs external engine.
            println!(
                "Checksmith SPRT  dev=internal  base={}  games={}  {}  openings={}{}",
                path, games, tc_label, config.openings.len(),
                if paired { "  paired" } else { "" },
            );
            if config.sprt.is_some() {
                println!("SPRT: elo0=0 elo1=5 alpha=0.05 beta=0.05  (stops on decision)");
            }
            println!();

            match play_match_vs(&config, path, |idx, record, running, sprt| {
                let sprt_str = sprt.map(|s| format!("  {}", s)).unwrap_or_default();
                println!(
                    "  [{:3}/{}] {:>3} moves  {}  ({}){}", idx + 1, games,
                    record.moves.len(), record.outcome, running, sprt_str,
                );
            }) {
                Ok(result) => {
                    let sprt = config.sprt.as_ref().map(|c| {
                        let mut s = SprtTest::new(c.clone());
                        s.update(result.white_wins, result.draws, result.black_wins);
                        s
                    });
                    print_match_summary(&result, sprt.as_ref());
                }
                Err(e) => eprintln!("Match aborted: {}", e),
            }
        }
    }
}

fn print_match_summary(result: &MatchResult, sprt: Option<&SprtTest>) {
    println!();
    println!("======================================");
    println!("Match result  {}", result);
    if let Some(s) = sprt {
        println!("{}", s);
        let verdict = match s.outcome() {
            SprtOutcome::H1Accepted => "ACCEPTED  (improvement confirmed)",
            SprtOutcome::H0Accepted => "REJECTED  (no improvement detected)",
            SprtOutcome::Continue   => "INCONCLUSIVE (games limit reached)",
        };
        println!("Decision: {}", verdict);
    }
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
        let fen = "8/8/4k3/8/8/4K3/8/8 w - - 100 50";
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            max_moves: 400,
            ..Default::default()
        };
        let record = play_game(fen, &config);
        assert_eq!(record.outcome, GameOutcome::Draw(DrawReason::FiftyMoveDraw));
        assert_eq!(record.moves.len(), 0);
    }

    #[test]
    fn adjudication_fires_when_threshold_maxed() {
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            adj_threshold: i32::MAX,
            adj_min_moves: 2,
            max_moves: 400,
            ..Default::default()
        };
        let record = play_game(STARTING_FEN, &config);
        assert_eq!(record.outcome, GameOutcome::Draw(DrawReason::Adjudicated));
        assert!(record.moves.len() >= 2);
    }

    #[test]
    fn max_moves_cap_fires() {
        let config = MatchConfig {
            tc: TimeControl::FixedDepth(1),
            adj_threshold: 0,
            adj_min_moves: 9999,
            max_moves: 4,
            ..Default::default()
        };
        let record = play_game(STARTING_FEN, &config);
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

    #[test]
    fn flip_outcome_swaps_wins() {
        assert_eq!(flip_outcome(GameOutcome::WhiteWins), GameOutcome::BlackWins);
        assert_eq!(flip_outcome(GameOutcome::BlackWins), GameOutcome::WhiteWins);
        assert_eq!(
            flip_outcome(GameOutcome::Draw(DrawReason::Stalemate)),
            GameOutcome::Draw(DrawReason::Stalemate)
        );
    }

    #[test]
    fn opening_source_epd_wraps_around() {
        let fens = vec!["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()];
        let src  = OpeningSource::Epd(fens);
        assert_eq!(src.pick(0), src.pick(1)); // wraps to index 0
        assert_eq!(src.len(), 1);
    }

    #[test]
    fn paired_games_play_requested_count() {
        let config = MatchConfig {
            games:  6,
            paired: true,
            tc:     TimeControl::FixedDepth(1),
            max_moves: 50,
            ..Default::default()
        };
        let mut count = 0usize;
        let result = play_match(&config, |_, _, _| count += 1);
        assert_eq!(count, 6);
        assert_eq!(result.total(), 6);
    }

    #[test]
    fn sprt_auto_stop_fires() {
        // With forced draws (s=0.5) and elo1=200, LLR drifts ~−0.16 per game
        // toward H0; lo_bound ≈ −2.94, so auto-stop fires in ~20 games.
        // Using elo1=5 would need ~30k games — far too slow for a unit test.
        let config = MatchConfig {
            games: 10_000,
            tc:    TimeControl::FixedDepth(1),
            sprt:  Some(SprtConfig { elo0: 0.0, elo1: 200.0, alpha: 0.05, beta: 0.05 }),
            max_moves: 50,
            adj_threshold: i32::MAX, // force every game to a 2-move draw
            adj_min_moves: 2,
            ..Default::default()
        };
        let mut games_played = 0usize;
        play_match(&config, |_, _, _| games_played += 1);
        assert!(games_played < 10_000, "SPRT auto-stop did not fire");
        assert!(games_played < 500,    "SPRT should stop in ~20 games with elo1=200");
    }

    #[test]
    fn save_and_resume_counts_results() {
        use std::env;
        let path = env::temp_dir().join("checksmith_test_save.txt");
        let path_str = path.to_str().unwrap().to_string();
        // Remove any leftover file.
        let _ = std::fs::remove_file(&path);

        let config = MatchConfig {
            games: 4,
            tc:    TimeControl::FixedDepth(1),
            max_moves: 50,
            adj_threshold: i32::MAX,
            adj_min_moves: 2,
            save_file: Some(path_str.clone()),
            ..Default::default()
        };
        play_match(&config, |_, _, _| {});

        let count = count_saved_results(&path_str);
        assert_eq!(count, 4, "save file should contain exactly 4 lines");

        let (w, d, l) = read_saved_results(&path_str);
        assert_eq!(w + d + l, 4, "W+D+L must equal games played");

        let _ = std::fs::remove_file(&path);
    }
}
