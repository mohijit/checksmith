//! UCI subprocess wrapper for external-engine testing.
//!
//! Spawns an engine binary, performs the UCI handshake, and provides a
//! simple request/response interface for playing individual moves.  The
//! engine's stderr is discarded; only stdout is consumed.
//!
//! ## Lifecycle
//!
//! 1. [`UciProcess::launch`] — spawn the process and complete the handshake.
//! 2. Call [`UciProcess::new_game`] before each game to reset engine state.
//! 3. Call [`UciProcess::get_move`] once per half-move.
//! 4. `UciProcess` is dropped at the end of the match; `quit` is sent
//!    automatically.
//!
//! ## Error handling
//!
//! All I/O errors and illegal moves (moves not in `board.legal_moves()`) cause
//! `get_move` to return `None`, which the game loop treats as
//! `AbortReason::NoMoveFound`.

use crate::board::Board;
use crate::movegen::Move;
use crate::selfplay::TimeControl;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

// ── UciProcess ────────────────────────────────────────────────────────────────

/// A live UCI engine subprocess.
pub struct UciProcess {
    child:       Child,
    stdin:       BufWriter<ChildStdin>,
    reader:      BufReader<ChildStdout>,
    opening_fen: String,
}

impl UciProcess {
    /// Spawn the engine at `path`, run the `uci` / `isready` handshake, and
    /// return a ready handle.  Returns `Err` if the process cannot be spawned
    /// or the handshake times out / fails.
    pub fn launch(path: &str) -> std::io::Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| std::io::Error::new(e.kind(), format!("cannot launch '{}': {}", path, e)))?;

        let stdin  = BufWriter::new(child.stdin.take().expect("stdin piped"));
        let reader = BufReader::new(child.stdout.take().expect("stdout piped"));

        let mut p = UciProcess { child, stdin, reader, opening_fen: String::new() };
        p.handshake().map_err(|e| {
            std::io::Error::new(e.kind(), format!("UCI handshake with '{}' failed: {}", path, e))
        })?;
        Ok(p)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn send(&mut self, cmd: &str) -> std::io::Result<()> {
        writeln!(self.stdin, "{}", cmd)?;
        self.stdin.flush()
    }

    fn read_raw(&mut self) -> std::io::Result<String> {
        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        Ok(line.trim_end_matches(['\r', '\n']).to_string())
    }

    fn drain_until(&mut self, token: &str) -> std::io::Result<()> {
        loop {
            let line = self.read_raw()?;
            if line == token { return Ok(()); }
        }
    }

    fn handshake(&mut self) -> std::io::Result<()> {
        self.send("uci")?;
        // Drain option/id lines until "uciok".
        loop {
            let line = self.read_raw()?;
            if line == "uciok" { break; }
        }
        self.send("isready")?;
        self.drain_until("readyok")
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Send a `setoption name N value V` command.
    ///
    /// Call this after [`launch`](Self::launch) but before [`new_game`](Self::new_game)
    /// to configure engine-specific parameters (e.g., SPSA-perturbed values).
    pub fn send_setoption(&mut self, name: &str, value: i32) -> std::io::Result<()> {
        self.send(&format!("setoption name {} value {}", name, value))
    }

    /// Signal the start of a new game and store `opening_fen` for subsequent
    /// [`get_move`](Self::get_move) calls.
    pub fn new_game(&mut self, opening_fen: &str) -> std::io::Result<()> {
        self.opening_fen = opening_fen.to_string();
        self.send("ucinewgame")?;
        self.send("isready")?;
        self.drain_until("readyok")
    }

    /// Ask the engine for its best move from the current position.
    ///
    /// Sends `position fen <opening> moves <m0> <m1> …` then `go depth N` or
    /// `go movetime N`.  Reads `info` lines (capturing the last `score cp`
    /// value) until `bestmove` arrives.
    ///
    /// Returns `(best_move, score_cp)`.  `best_move` is `None` if the engine
    /// sends `bestmove (none)`, gives an illegal move, or if an I/O error occurs.
    pub fn get_move(
        &mut self,
        board:        &Board,
        moves_played: &[Move],
        tc:           &TimeControl,
    ) -> (Option<Move>, i32) {
        if self.send_position(moves_played).is_err() {
            return (None, 0);
        }
        let go_cmd = match tc {
            TimeControl::FixedDepth(d) => format!("go depth {}", d),
            TimeControl::MoveTime(ms)  => format!("go movetime {}", ms),
        };
        if self.send(&go_cmd).is_err() {
            return (None, 0);
        }
        self.read_bestmove(board)
    }

    fn send_position(&mut self, moves_played: &[Move]) -> std::io::Result<()> {
        let cmd = if moves_played.is_empty() {
            format!("position fen {}", self.opening_fen)
        } else {
            let ms = moves_played.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ");
            format!("position fen {} moves {}", self.opening_fen, ms)
        };
        self.send(&cmd)
    }

    fn read_bestmove(&mut self, board: &Board) -> (Option<Move>, i32) {
        let mut last_score: i32 = 0;
        loop {
            let line = match self.read_raw() {
                Ok(l)  => l,
                Err(_) => return (None, 0),
            };
            // Capture the latest "score cp N" from info lines.
            if line.starts_with("info") {
                if let Some(s) = parse_score_cp(&line) {
                    last_score = s;
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("bestmove ") {
                let uci = rest.split_whitespace().next().unwrap_or("");
                if uci.is_empty() || uci == "(none)" {
                    return (None, last_score);
                }
                let mv = board.legal_moves().iter().find(|&&m| m.to_uci() == uci).copied();
                return (mv, last_score);
            }
        }
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract the `score cp N` value from a UCI `info` line, if present.
fn parse_score_cp(info: &str) -> Option<i32> {
    let mut parts = info.split_whitespace();
    while let Some(tok) = parts.next() {
        if tok == "score" {
            let kind = parts.next()?;
            if kind == "cp" {
                return parts.next()?.parse().ok();
            }
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_score_cp_basic() {
        let line = "info depth 10 seldepth 14 score cp 25 nodes 12345 nps 1234567 pv e2e4";
        assert_eq!(parse_score_cp(line), Some(25));
    }

    #[test]
    fn parse_score_cp_negative() {
        let line = "info depth 8 score cp -42 nodes 999";
        assert_eq!(parse_score_cp(line), Some(-42));
    }

    #[test]
    fn parse_score_cp_mate_line_returns_none() {
        // "score mate N" — we don't parse mate scores here, return None.
        let line = "info depth 5 score mate 3 nodes 1000";
        assert_eq!(parse_score_cp(line), None);
    }

    #[test]
    fn parse_score_cp_absent_returns_none() {
        assert_eq!(parse_score_cp("info depth 1 nodes 100"), None);
    }
}
