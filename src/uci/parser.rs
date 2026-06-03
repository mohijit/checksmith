//! Parsing of UCI command arguments: moves, `position`, and `go`.
//!
//! UCI uses *long algebraic* move notation — source square, target square, and
//! an optional promotion letter (`e2e4`, `e7e8q`, `e1g1` for castling, `e5d6`
//! for en passant). We resolve such a string to one of our [`Move`]s by
//! matching it against the position's legal moves; this makes castling, en
//! passant, and promotion fall out automatically without special cases.

use crate::board::{Board, Square, STARTING_FEN};
use crate::movegen::Move;
use crate::search::SearchLimits;

/// Resolve a UCI move string against a pre-computed legal `MoveList`.
pub fn parse_move(text: &str, legal: &crate::movegen::MoveList) -> Option<Move> {
    let bytes = text.as_bytes();
    if bytes.len() < 4 { return None; }
    let from  = Square::from_algebraic(&text[0..2])?;
    let to    = Square::from_algebraic(&text[2..4])?;
    let promo = bytes.get(4).map(|&b| b.to_ascii_lowercase() as char);
    legal.iter().copied().find(|&mv| {
        mv.from() == from
            && mv.to() == to
            && match (promo, mv.promotion_piece()) {
                (None, None) => true,
                (Some(c), Some(pt)) => promo_char(pt) == c,
                _ => false,
            }
    })
}

/// Resolve a UCI move string against `board`'s legal moves.
pub fn parse_uci_move(board: &Board, text: &str) -> Option<Move> {
    let bytes = text.as_bytes();
    if bytes.len() < 4 {
        return None;
    }
    let from = Square::from_algebraic(&text[0..2])?;
    let to = Square::from_algebraic(&text[2..4])?;
    let promo = bytes.get(4).map(|&b| b.to_ascii_lowercase() as char);

    for &mv in board.legal_moves().iter() {
        if mv.from() != from || mv.to() != to {
            continue;
        }
        // Match the promotion piece too (a non-promotion move has none).
        match (promo, mv.promotion_piece()) {
            (None, None) => return Some(mv),
            (Some(c), Some(pt)) if promo_char(pt) == c => return Some(mv),
            _ => {}
        }
    }
    None
}

fn promo_char(pt: crate::board::PieceType) -> char {
    use crate::board::PieceType::*;
    match pt {
        Knight => 'n',
        Bishop => 'b',
        Rook => 'r',
        Queen => 'q',
        _ => '?',
    }
}

/// Build the board described by a `position` command's arguments (everything
/// after the word `position`).
///
/// Forms: `startpos [moves ...]` or `fen <6 fields> [moves ...]`.
pub fn parse_position(tokens: &[&str]) -> Option<Board> {
    let mut board;
    let moves_start;

    match tokens.first()? {
        &"startpos" => {
            board = Board::from_fen(STARTING_FEN).ok()?;
            moves_start = 1;
        }
        &"fen" => {
            // A FEN is six space-separated fields immediately after `fen`.
            if tokens.len() < 7 {
                return None;
            }
            let fen = tokens[1..7].join(" ");
            board = Board::from_fen(&fen).ok()?;
            moves_start = 7;
        }
        _ => return None,
    }

    // Apply any moves listed after the `moves` keyword.
    if let Some(pos) = tokens[moves_start..].iter().position(|&t| t == "moves") {
        let first_move = moves_start + pos + 1;
        for &mv_str in &tokens[first_move..] {
            let mv = parse_uci_move(&board, mv_str)?;
            board.make_move(mv);
        }
    }

    Some(board)
}

/// Parse a `go` command's arguments (everything after the word `go`).
pub fn parse_go(tokens: &[&str]) -> SearchLimits {
    let mut limits = SearchLimits::default();
    let mut i = 0;
    // Helper to read the integer following a keyword.
    let next_u64 = |i: &mut usize| -> Option<u64> {
        *i += 1;
        tokens.get(*i).and_then(|t| t.parse().ok())
    };

    while i < tokens.len() {
        match tokens[i] {
            "depth" => limits.depth = next_u64(&mut i).map(|v| v as u32),
            "movetime" => limits.movetime = next_u64(&mut i),
            "wtime" => limits.wtime = next_u64(&mut i),
            "btime" => limits.btime = next_u64(&mut i),
            "winc" => limits.winc = next_u64(&mut i),
            "binc" => limits.binc = next_u64(&mut i),
            "movestogo" => limits.movestogo = next_u64(&mut i).map(|v| v as u32),
            "nodes" => limits.nodes = next_u64(&mut i),
            "infinite" => limits.infinite = true,
            "ponder"   => limits.ponder   = true,
            _ => {} // ignore unknown / unsupported tokens (e.g. `searchmoves`)
        }
        i += 1;
    }
    limits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_normal_and_special_moves() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(
            parse_uci_move(&board, "e2e4").map(|m| m.to_uci()),
            Some("e2e4".to_string())
        );
        assert!(parse_uci_move(&board, "e2e5").is_none()); // illegal

        // Castling is a king two-square move.
        let castle = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        assert!(parse_uci_move(&castle, "e1g1").unwrap().is_castle());

        // Promotion.
        let promo = Board::from_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").unwrap();
        let mv = parse_uci_move(&promo, "a7a8q").unwrap();
        assert_eq!(mv.to_uci(), "a7a8q");
    }

    #[test]
    fn position_startpos_with_moves() {
        let board = parse_position(&["startpos", "moves", "e2e4", "e7e5", "g1f3"]).unwrap();
        // After 1.e4 e5 2.Nf3, a knight sits on f3 and it's Black to move.
        assert_eq!(board.to_fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
    }

    #[test]
    fn position_fen_form() {
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
        let command = format!("fen {}", fen);
        let tokens: Vec<&str> = command.split_whitespace().collect();
        let board = parse_position(&tokens).unwrap();
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn go_parses_clock_and_depth() {
        let limits = parse_go(&["wtime", "300000", "btime", "300000", "winc", "2000"]);
        assert_eq!(limits.wtime, Some(300_000));
        assert_eq!(limits.winc, Some(2_000));

        let limits = parse_go(&["depth", "8"]);
        assert_eq!(limits.depth, Some(8));

        let limits = parse_go(&["infinite"]);
        assert!(limits.infinite);
        assert!(!limits.ponder);
    }

    #[test]
    fn go_parses_ponder() {
        let limits = parse_go(&["ponder"]);
        assert!(limits.ponder, "ponder flag should be set");
        assert!(!limits.infinite);
    }

    #[test]
    fn go_ponder_is_treated_as_infinite() {
        let limits = parse_go(&["ponder"]);
        assert!(limits.is_infinite(), "ponder implies no time pressure");
    }

    #[test]
    fn go_parses_nodes() {
        let limits = parse_go(&["nodes", "1000000"]);
        assert_eq!(limits.nodes, Some(1_000_000));
    }

    #[test]
    fn go_parses_movetime() {
        let limits = parse_go(&["movetime", "5000"]);
        assert_eq!(limits.movetime, Some(5_000));
        assert!(limits.wtime.is_none());
    }

    #[test]
    fn go_parses_movestogo() {
        let limits = parse_go(&["wtime", "60000", "btime", "60000", "movestogo", "20"]);
        assert_eq!(limits.movestogo, Some(20));
        assert_eq!(limits.wtime, Some(60_000));
    }

    #[test]
    fn position_invalid_input_returns_none() {
        assert!(parse_position(&[]).is_none());
        assert!(parse_position(&["garbage"]).is_none());
        // FEN with too few fields.
        assert!(parse_position(&["fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"]).is_none());
    }
}
