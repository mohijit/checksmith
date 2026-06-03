//! Integration tests for Milestone 1: board representation and FEN handling.

use checksmith::board::{Board, CastlingRights, Color, Piece, PieceType, Square, STARTING_FEN};

#[test]
fn starting_position_piece_counts() {
    let b = Board::starting_position();
    assert_eq!(b.occupancy().count(), 32);
    assert_eq!(b.color(Color::White).count(), 16);
    assert_eq!(b.color(Color::Black).count(), 16);
    assert_eq!(b.pieces(PieceType::Pawn).count(), 16);
    assert_eq!(b.pieces(PieceType::King).count(), 2);
    assert_eq!(b.pieces_colored(Color::White, PieceType::Queen).count(), 1);
}

#[test]
fn starting_position_metadata() {
    let b = Board::starting_position();
    assert_eq!(b.side_to_move, Color::White);
    assert_eq!(b.castling, CastlingRights::ALL);
    assert_eq!(b.ep_square, None);
    assert_eq!(b.halfmove_clock, 0);
    assert_eq!(b.fullmove_number, 1);
}

#[test]
fn starting_position_piece_placement() {
    let b = Board::starting_position();
    assert_eq!(
        b.piece_at(Square::E1),
        Some(Piece::new(Color::White, PieceType::King))
    );
    assert_eq!(
        b.piece_at(Square::D8),
        Some(Piece::new(Color::Black, PieceType::Queen))
    );
    assert_eq!(
        b.piece_at(Square::A1),
        Some(Piece::new(Color::White, PieceType::Rook))
    );
    assert_eq!(b.piece_at(Square::E4), None);
    assert_eq!(b.king_square(Color::Black), Some(Square::E8));
}

#[test]
fn fen_round_trips() {
    // Each of these is already in canonical form, so to_fen must reproduce it.
    let fens = [
        STARTING_FEN,
        // "Kiwipete" — a famous perft position, exercises castling on both sides.
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        // En passant available.
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        // No castling rights, Black to move, advanced clocks.
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 13 47",
    ];
    for fen in fens {
        let b = Board::from_fen(fen).expect("valid FEN should parse");
        assert_eq!(b.to_fen(), fen, "round-trip mismatch for: {}", fen);
    }
}

#[test]
fn parses_en_passant_and_clocks() {
    let b =
        Board::from_fen("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2").unwrap();
    assert_eq!(b.ep_square, Square::from_algebraic("c6"));
    assert_eq!(b.fullmove_number, 2);
    assert_eq!(b.side_to_move, Color::White);
}

#[test]
fn zobrist_hash_is_consistent() {
    let a = Board::from_fen(STARTING_FEN).unwrap();
    let b = Board::starting_position();
    // Same position -> same hash, and it's non-trivial.
    assert_eq!(a.hash, b.hash);
    assert_ne!(a.hash, 0);

    // Only the side to move differs -> the hashes must differ.
    let black_to_move =
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
    assert_ne!(a.hash, black_to_move.hash);
}

#[test]
fn rejects_invalid_fens() {
    assert!(Board::from_fen("").is_err());
    // Only 7 ranks.
    assert!(Board::from_fen("8/8/8/8/8/8/8 w - - 0 1").is_err());
    // Rank with too many files (9).
    assert!(Board::from_fen("rnbqkbnrp/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    // Illegal side-to-move.
    assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1").is_err());
    // Illegal piece letter.
    assert!(Board::from_fen("xnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
}
