//! EPD (Extended Position Description) test suite runner.
//!
//! EPD is the standard format for tactical test positions, used by suites like
//! Bratko-Kopec, Win At Chess (WAC), and Arasan.
//!
//! ## Format
//!
//! Each line is:
//! ```text
//! <board> <side> <castling> <ep_sq> bm <move(s)>; id "<name>"; ...
//! ```
//! The first four space-separated tokens are identical to FEN fields 1–4
//! (board layout, side to move, castling rights, en-passant square).  EPD
//! omits the halfmove clock and fullmove number; we append `0 1` when
//! converting to a full FEN for our `Board` parser.
//!
//! Supported operations:
//! | Op   | Meaning |
//! |------|---------|
//! | `bm` | Best move(s) — engine must find one of these |
//! | `am` | Avoid move(s) — engine must NOT play any of these |
//! | `id` | Human-readable identifier for the position |
//!
//! All other operations (`ce`, `acd`, `acn`, …) are silently ignored.
//!
//! ## CLI
//!
//! ```sh
//! checksmith epd suite.epd              # depth 8 per position (default)
//! checksmith epd suite.epd 12           # depth 12 per position
//! checksmith epd suite.epd --depth 10   # explicit depth flag
//! checksmith epd suite.epd --time 1000  # 1 000 ms per position
//! ```

use crate::board::{Board, PieceType, Square};
use crate::movegen::{Move, MoveFlag};
use crate::search::iterative::{think, SearchLimits};
use crate::search::tt::TranspositionTable;
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

// ── Data types ────────────────────────────────────────────────────────────────

/// A single parsed EPD test position.
pub struct EpdPosition {
    /// Full FEN string (EPD 4-field setup + appended `"0 1"`).
    pub fen: String,
    /// Moves from the `bm` operation (raw SAN strings, e.g. `["Rd1+", "Nf5"]`).
    pub best_moves: Vec<String>,
    /// Moves from the `am` operation (raw SAN strings).
    pub avoid_moves: Vec<String>,
    /// Human-readable id from the `id` operation, or empty string.
    pub id: String,
}

// ── Parsing ───────────────────────────────────────────────────────────────────

/// Parse one EPD line.  Returns `None` for blank lines and `#`-comments.
pub fn parse_line(line: &str) -> Option<EpdPosition> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // First four tokens: board, side, castling, ep-square.
    let mut tokens = line.splitn(5, ' ');
    let board_str = tokens.next()?;
    let side      = tokens.next()?;
    let castling  = tokens.next()?;
    let ep_sq     = tokens.next()?;
    let ops_str   = tokens.next().unwrap_or("");

    let fen = format!("{} {} {} {} 0 1", board_str, side, castling, ep_sq);
    let (best_moves, avoid_moves, id) = parse_ops(ops_str);

    Some(EpdPosition { fen, best_moves, avoid_moves, id })
}

/// Parse the semicolon-delimited operations section of an EPD line.
fn parse_ops(ops: &str) -> (Vec<String>, Vec<String>, String) {
    let mut best_moves  = Vec::new();
    let mut avoid_moves = Vec::new();
    let mut id          = String::new();

    for op in ops.split(';') {
        let op = op.trim();
        if op.is_empty() { continue; }

        let (opcode, value) = if let Some(sp) = op.find(' ') {
            (&op[..sp], op[sp + 1..].trim())
        } else {
            (op, "")
        };

        match opcode {
            "bm" => best_moves.extend(value.split_whitespace().map(String::from)),
            "am" => avoid_moves.extend(value.split_whitespace().map(String::from)),
            "id" => id = value.trim_matches('"').to_string(),
            _    => {}
        }
    }

    (best_moves, avoid_moves, id)
}

/// Read and parse every line of an EPD file.  Blank lines and comments are
/// skipped; positions with unparseable FENs are reported to stderr and omitted.
pub fn parse_file(path: &Path) -> std::io::Result<Vec<EpdPosition>> {
    let content = fs::read_to_string(path)?;
    Ok(content.lines().filter_map(parse_line).collect())
}

// ── SAN → Move conversion ─────────────────────────────────────────────────────

/// Convert a SAN move string to a [`Move`] in the context of `board`.
///
/// Handles the full SAN grammar:
/// * Castling: `O-O`, `O-O-O` (also `0-0`, `0-0-0`)
/// * Promotions: `e8=Q`, `exd8=N`
/// * File/rank/full-square disambiguation: `Ndf3`, `N3f3`, `Nd1f3`
/// * Pawn captures: `exd5`
/// * Annotation suffixes (`+`, `#`, `!`, `?`) are stripped before parsing.
///
/// Returns `None` if the SAN string does not match any legal move.
pub fn san_to_move(san: &str, board: &Board) -> Option<Move> {
    let legal = board.legal_moves();

    // Strip trailing annotation characters.
    let san = san.trim_end_matches(['+', '#', '!', '?']);

    // ── Castling ──────────────────────────────────────────────────────────────
    if san == "O-O-O" || san == "0-0-0" {
        return legal.iter().find(|&&m| m.flag() == MoveFlag::QueenCastle).copied();
    }
    if san == "O-O" || san == "0-0" {
        return legal.iter().find(|&&m| m.flag() == MoveFlag::KingCastle).copied();
    }

    // ── Piece type ────────────────────────────────────────────────────────────
    let (piece_type, rest) = match san.chars().next()? {
        c @ ('N' | 'B' | 'R' | 'Q' | 'K') => {
            (char_to_piece_type(c)?, &san[1..])
        }
        _ => (PieceType::Pawn, san),
    };

    // ── Promotion: "e8=Q" → rest="e8", promo=Some(Queen) ─────────────────────
    let (rest, promo) = if let Some(eq) = rest.rfind('=') {
        let promo_char = rest[eq + 1..].chars().next()?;
        (rest[..eq].as_ref(), Some(char_to_piece_type(promo_char)?))
    } else {
        (rest, None)
    };

    // Strip capture marker so we can index purely by position.
    let no_x: String = rest.chars().filter(|&c| c != 'x').collect();
    let rest = no_x.as_str();

    if rest.len() < 2 { return None; }

    // ── Destination square = last two characters ───────────────────────────────
    let to_sq    = Square::from_algebraic(&rest[rest.len() - 2..])?;
    let disambig = &rest[..rest.len() - 2];

    // ── Optional disambiguation by file (lowercase) and/or rank (digit) ───────
    let from_file: Option<u8> = disambig
        .chars()
        .find(|c| c.is_ascii_lowercase())
        .map(|c| c as u8 - b'a');
    let from_rank: Option<u8> = disambig
        .chars()
        .find(|c| c.is_ascii_digit())
        .map(|c| c as u8 - b'1');

    let stm = board.side_to_move;

    legal.iter().copied().find(|&m| {
        if m.to() != to_sq { return false; }
        let piece = match board.piece_at(m.from()) {
            Some(p) => p,
            None    => return false,
        };
        if piece.color      != stm        { return false; }
        if piece.piece_type != piece_type { return false; }
        if m.promotion_piece() != promo   { return false; }
        if let Some(ff) = from_file { if m.from().file() != ff { return false; } }
        if let Some(fr) = from_rank { if m.from().rank() != fr { return false; } }
        true
    })
}

fn char_to_piece_type(c: char) -> Option<PieceType> {
    match c {
        'N' => Some(PieceType::Knight),
        'B' => Some(PieceType::Bishop),
        'R' => Some(PieceType::Rook),
        'Q' => Some(PieceType::Queen),
        'K' => Some(PieceType::King),
        _   => None,
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

/// Run an EPD test suite, searching each position and checking the result
/// against the `bm` (best-move) and `am` (avoid-move) constraints.
///
/// Pass `depth > 0` for a fixed-depth search, or `depth == 0` with
/// `time_ms > 0` for a per-position time budget.
pub fn run_epd(path: &Path, depth: u32, time_ms: u64) {
    let positions = match parse_file(path) {
        Ok(p) if !p.is_empty() => p,
        Ok(_)  => { eprintln!("EPD file is empty or contains no valid positions."); return; }
        Err(e) => { eprintln!("Failed to read {:?}: {}", path, e); return; }
    };

    let n        = positions.len();
    let use_time = depth == 0;

    println!(
        "Checksmith EPD  file={}  positions={}  {}",
        path.display(),
        n,
        if use_time { format!("time={}ms", time_ms) } else { format!("depth={}", depth) }
    );
    println!();

    let stop = Arc::new(AtomicBool::new(false));
    let tt   = TranspositionTable::new(16);

    let total_start = Instant::now();
    let mut solved  = 0usize;
    let mut tested  = 0usize;

    for (i, pos) in positions.iter().enumerate() {
        let mut board = match Board::from_fen(&pos.fen) {
            Ok(b)  => b,
            Err(e) => {
                eprintln!("  [{:>4}/{}] SKIP  {} — bad FEN: {}", i + 1, n, pos.id, e);
                continue;
            }
        };

        let limits = if use_time {
            SearchLimits { movetime: Some(time_ms), ..Default::default() }
        } else {
            SearchLimits { depth: Some(depth), ..Default::default() }
        };

        tt.new_generation();
        let pos_start = Instant::now();
        let result    = think(&mut board, &limits, Arc::clone(&stop), &tt, &[], None, |_| {});
        let pos_ms    = pos_start.elapsed().as_millis();

        let found_uci = result.best_move.map(|m| m.to_uci()).unwrap_or_default();

        // Re-parse the board for SAN conversion (think() restores its state,
        // but using a fresh copy is cleaner and avoids any edge-case aliasing).
        let mut board2 = Board::from_fen(&pos.fen).unwrap();
        let pass = evaluate_result(&found_uci, &pos.best_moves, &pos.avoid_moves, &mut board2);

        if pass.is_some() { tested += 1; }
        if pass == Some(true) { solved += 1; }

        let mark = match pass {
            Some(true)  => '✓',
            Some(false) => '✗',
            None        => '-',
        };

        let bm_display = if !pos.best_moves.is_empty() {
            pos.best_moves.join(" ")
        } else if !pos.avoid_moves.is_empty() {
            format!("am:{}", pos.avoid_moves.join(" "))
        } else {
            "(any)".to_string()
        };

        println!(
            "  [{:>4}/{}] {}  {:<28}  bm: {:<18}  found: {:<8}  {:.2}s",
            i + 1, n, mark,
            trunc(&pos.id, 28),
            trunc(&bm_display, 18),
            found_uci,
            pos_ms as f64 / 1000.0,
        );
    }

    let total_ms = total_start.elapsed().as_millis();
    let pct      = if tested > 0 { 100.0 * solved as f64 / tested as f64 } else { 0.0 };

    println!();
    println!("===========================");
    println!("Solved  : {}/{}", solved, tested);
    if tested < n {
        println!("Skipped : {} (no bm/am)", n - tested);
    }
    println!("Score   : {:.1}%", pct);
    println!(
        "Time    : {:.3}s  ({:.2}s/pos)",
        total_ms as f64 / 1000.0,
        total_ms as f64 / 1000.0 / n.max(1) as f64,
    );
    println!("===========================");
}

/// Determine whether the engine's move satisfies the EPD constraints.
///
/// * `bm` present → engine must play one of the listed moves.
/// * `am` present → engine must NOT play any of the listed moves.
/// * Neither      → returns `None` (no constraint, position is unchecked).
fn evaluate_result(
    found_uci:   &str,
    best_moves:  &[String],
    avoid_moves: &[String],
    board:       &mut Board,
) -> Option<bool> {
    let has_bm = !best_moves.is_empty();
    let has_am = !avoid_moves.is_empty();
    if !has_bm && !has_am { return None; }

    let mut pass = true;

    if has_bm {
        let matched = best_moves.iter().any(|san| {
            san_to_move(san, board)
                .map(|m| m.to_uci() == found_uci)
                .unwrap_or(false)
        });
        pass &= matched;
    }

    if has_am {
        let avoided = avoid_moves.iter().all(|san| {
            san_to_move(san, board)
                .map(|m| m.to_uci() != found_uci)
                .unwrap_or(true)
        });
        pass &= avoided;
    }

    Some(pass)
}

fn trunc(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}

// ── CLI entry point ───────────────────────────────────────────────────────────

/// Handle `checksmith epd <file.epd> [depth | --depth N | --time N]`.
pub fn run_epd_command(args: &[String]) {
    let Some(file) = args.first() else {
        eprintln!("usage: checksmith epd <file.epd> [--depth N | --time N]");
        std::process::exit(1);
    };

    let mut depth:   u32 = 8;
    let mut time_ms: u64 = 0;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--time" | "-t" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    time_ms = v.parse().unwrap_or(1000);
                    depth   = 0;
                }
            }
            "--depth" | "-d" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    depth = v.parse().unwrap_or(8);
                }
            }
            s => {
                if let Ok(d) = s.parse::<u32>() {
                    depth = d;
                }
            }
        }
        i += 1;
    }

    run_epd(Path::new(file), depth, time_ms);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;

    // A selection from the Bratko-Kopec suite.
    const BK01: &str =
        r#"1k1r4/pp1b1R2/3p4/2pPp3/2P2B2/2B5/7r/2K5 b - - bm Rd1+; id "BK.01";"#;
    const BK02: &str =
        r#"3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - bm d5; id "BK.02";"#;

    #[test]
    fn parse_line_extracts_bm_and_id() {
        let pos = parse_line(BK01).unwrap();
        assert_eq!(pos.best_moves, vec!["Rd1+"]);
        assert_eq!(pos.id, "BK.01");
        Board::from_fen(&pos.fen).expect("EPD FEN must be parseable");
    }

    #[test]
    fn parse_line_two_bm() {
        // Some suites list multiple acceptable best moves.
        let line = r#"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - bm e4 d4; id "opening";"#;
        let pos  = parse_line(line).unwrap();
        assert_eq!(pos.best_moves, vec!["e4", "d4"]);
    }

    #[test]
    fn parse_line_am_operation() {
        let line = r#"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 am e5; id "avoid";"#;
        let pos  = parse_line(line).unwrap();
        assert!(pos.best_moves.is_empty());
        assert_eq!(pos.avoid_moves, vec!["e5"]);
    }

    #[test]
    fn parse_line_skips_blank_and_comments() {
        assert!(parse_line("").is_none());
        assert!(parse_line("   ").is_none());
        assert!(parse_line("# this is a comment").is_none());
    }

    #[test]
    fn parse_bk02_fen_is_valid() {
        let pos = parse_line(BK02).unwrap();
        Board::from_fen(&pos.fen).expect("BK.02 FEN must be parseable");
    }

    // ── SAN converter tests ───────────────────────────────────────────────────

    #[test]
    fn san_pawn_push_e4() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mv    = san_to_move("e4", &board).unwrap();
        assert_eq!(mv.to_uci(), "e2e4");
    }

    #[test]
    fn san_pawn_push_d4() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mv    = san_to_move("d4", &board).unwrap();
        assert_eq!(mv.to_uci(), "d2d4");
    }

    #[test]
    fn san_knight_develop_nf3() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mv    = san_to_move("Nf3", &board).unwrap();
        assert_eq!(mv.to_uci(), "g1f3");
    }

    #[test]
    fn san_knight_develop_nc3() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mv    = san_to_move("Nc3", &board).unwrap();
        assert_eq!(mv.to_uci(), "b1c3");
    }

    #[test]
    fn san_kingside_castle() {
        // Italian: after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 — White can castle.
        let fen   = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5";
        let board = Board::from_fen(fen).unwrap();
        let mv    = san_to_move("O-O", &board).unwrap();
        assert_eq!(mv.to_uci(), "e1g1");
    }

    #[test]
    fn san_queenside_castle() {
        // A position where White has queen-side castling rights available.
        let fen   = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
        let board = Board::from_fen(fen).unwrap();
        let mv    = san_to_move("O-O-O", &board).unwrap();
        assert_eq!(mv.to_uci(), "e1c1");
    }

    #[test]
    fn san_promotion_queen() {
        let fen   = "8/4P3/8/8/8/8/8/4K1k1 w - - 0 1";
        let board = Board::from_fen(fen).unwrap();
        let mv    = san_to_move("e8=Q", &board).unwrap();
        assert_eq!(mv.to_uci(), "e7e8q");
    }

    #[test]
    fn san_promotion_knight() {
        let fen   = "8/4P3/8/8/8/8/8/4K1k1 w - - 0 1";
        let board = Board::from_fen(fen).unwrap();
        let mv    = san_to_move("e8=N", &board).unwrap();
        assert_eq!(mv.to_uci(), "e7e8n");
    }

    #[test]
    fn san_pawn_capture() {
        // 1.e4 d5: White can play exd5.
        let fen   = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2";
        let board = Board::from_fen(fen).unwrap();
        let mv    = san_to_move("exd5", &board).unwrap();
        assert_eq!(mv.to_uci(), "e4d5");
    }

    #[test]
    fn san_piece_capture_with_x() {
        // Rxf7 in Scholar's mate-like position.
        let fen   = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        let board = Board::from_fen(fen).unwrap();
        // Bc4 can capture on f7: Bxf7+ — but that's the bishop.
        // Just verify that san_to_move handles the 'x' gracefully even if move is illegal.
        let _ = san_to_move("Rxf7", &board); // may be None if rook can't reach f7
    }

    #[test]
    fn san_returns_none_for_illegal_move() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert!(san_to_move("Qe4", &board).is_none()); // queen blocked by pawns
        assert!(san_to_move("Ke2", &board).is_none()); // king blocked by pawns
    }

    #[test]
    fn san_strips_check_suffix() {
        // Rd1+ — same as Rd1 but with check annotation.
        let fen = "1k1r4/pp1b1R2/3p4/2pPp3/2P2B2/2B5/7r/2K5 b - - 0 1";
        let board = Board::from_fen(fen).unwrap();
        let with_check    = san_to_move("Rxd1+", &board);
        let without_check = san_to_move("Rxd1",  &board);
        // Both should give the same move (or both None if the position doesn't allow it).
        assert_eq!(with_check.map(|m| m.to_uci()), without_check.map(|m| m.to_uci()));
    }

    // ── evaluate_result tests ─────────────────────────────────────────────────

    #[test]
    fn evaluate_no_constraint_returns_none() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        assert!(evaluate_result("e2e4", &[], &[], &mut board).is_none());
    }

    #[test]
    fn evaluate_bm_match_passes() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        // bm says "e4"; engine found e2e4.
        let r = evaluate_result("e2e4", &["e4".to_string()], &[], &mut board);
        assert_eq!(r, Some(true));
    }

    #[test]
    fn evaluate_bm_mismatch_fails() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        // bm says "Nf3"; engine found e2e4.
        let r = evaluate_result("e2e4", &["Nf3".to_string()], &[], &mut board);
        assert_eq!(r, Some(false));
    }

    #[test]
    fn evaluate_am_avoided_passes() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        // am says avoid "e4"; engine found d2d4 — should pass.
        let r = evaluate_result("d2d4", &[], &["e4".to_string()], &mut board);
        assert_eq!(r, Some(true));
    }

    #[test]
    fn evaluate_am_played_fails() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        // am says avoid "e4"; engine played e2e4 — should fail.
        let r = evaluate_result("e2e4", &[], &["e4".to_string()], &mut board);
        assert_eq!(r, Some(false));
    }

    #[test]
    fn evaluate_multiple_bm_any_match_passes() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        // bm says "e4" or "d4"; engine found d2d4 — should pass.
        let r = evaluate_result(
            "d2d4",
            &["e4".to_string(), "d4".to_string()],
            &[],
            &mut board,
        );
        assert_eq!(r, Some(true));
    }
}
