//! FEN parsing and serialization.
//!
//! **FEN** (Forsyth–Edwards Notation) is the standard one-line description of a
//! chess position. It has six space-separated fields:
//!
//! ```text
//! rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
//! └─────── piece placement ───────┘ │  │   │  │ └ fullmove number
//!                                   │  │   │  └── halfmove clock
//!                                   │  │   └───── en passant target ('-' = none)
//!                                   │  └───────── castling rights ('-' = none)
//!                                   └──────────── side to move (w/b)
//! ```
//!
//! Piece placement is given rank 8 first down to rank 1, files a→h within each
//! rank. Letters are pieces (uppercase = White); digits are runs of empty
//! squares. The last two fields are optional in many tools; we default them to
//! `0` and `1`.

use super::piece::{Color, Piece};
use super::square::Square;
use super::{Board, CastlingRights};
use std::fmt;

/// Why a FEN string could not be parsed.
#[derive(Debug, PartialEq, Eq)]
pub enum FenError {
    /// A required field was absent.
    MissingField(&'static str),
    /// Piece placement did not have exactly 8 ranks.
    BadRankCount(usize),
    /// A rank did not describe exactly 8 files.
    BadFileCount { rank: u8, found: usize },
    /// An unrecognized piece letter.
    UnknownPiece(char),
    /// Side-to-move field was not `w` or `b`.
    BadSideToMove(String),
    /// Castling field had an illegal character.
    BadCastling(String),
    /// En-passant field was neither `-` nor a valid square.
    BadEnPassant(String),
    /// A clock field was not a valid number.
    BadNumber(String),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenError::MissingField(name) => write!(f, "missing FEN field: {}", name),
            FenError::BadRankCount(n) => {
                write!(f, "piece placement has {} ranks, expected 8", n)
            }
            FenError::BadFileCount { rank, found } => {
                write!(f, "rank {} describes {} files, expected 8", rank, found)
            }
            FenError::UnknownPiece(c) => write!(f, "unknown piece letter '{}'", c),
            FenError::BadSideToMove(s) => write!(f, "invalid side to move '{}'", s),
            FenError::BadCastling(s) => write!(f, "invalid castling field '{}'", s),
            FenError::BadEnPassant(s) => write!(f, "invalid en passant field '{}'", s),
            FenError::BadNumber(s) => write!(f, "invalid number '{}'", s),
        }
    }
}

impl std::error::Error for FenError {}

impl Board {
    /// Parse a position from FEN.
    pub fn from_fen(fen: &str) -> Result<Board, FenError> {
        let mut fields = fen.split_whitespace();

        let placement = fields.next().ok_or(FenError::MissingField("piece placement"))?;
        let side = fields.next().ok_or(FenError::MissingField("side to move"))?;
        let castling = fields.next().ok_or(FenError::MissingField("castling"))?;
        let ep = fields.next().ok_or(FenError::MissingField("en passant"))?;
        // The clock fields are optional; default them when absent.
        let halfmove = fields.next().unwrap_or("0");
        let fullmove = fields.next().unwrap_or("1");

        let mut board = Board::empty();

        // --- Field 1: piece placement -------------------------------------
        let ranks: Vec<&str> = placement.split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError::BadRankCount(ranks.len()));
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            // ranks[0] describes rank 8 (rank index 7), and so on down.
            let rank = 7 - i as u8;
            let mut file: u8 = 0;
            for ch in rank_str.chars() {
                if let Some(skip) = ch.to_digit(10) {
                    file += skip as u8;
                } else {
                    let piece = Piece::from_char(ch).ok_or(FenError::UnknownPiece(ch))?;
                    if file > 7 {
                        return Err(FenError::BadFileCount {
                            rank: rank + 1,
                            found: file as usize + 1,
                        });
                    }
                    board.put_piece(Square::from_file_rank(file, rank), piece);
                    file += 1;
                }
            }
            if file != 8 {
                return Err(FenError::BadFileCount {
                    rank: rank + 1,
                    found: file as usize,
                });
            }
        }

        // --- Field 2: side to move ----------------------------------------
        board.side_to_move = match side {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(FenError::BadSideToMove(other.to_string())),
        };

        // --- Field 3: castling rights -------------------------------------
        let mut rights = CastlingRights::NONE;
        if castling != "-" {
            for ch in castling.chars() {
                match ch {
                    'K' => rights.add(CastlingRights::WHITE_KING),
                    'Q' => rights.add(CastlingRights::WHITE_QUEEN),
                    'k' => rights.add(CastlingRights::BLACK_KING),
                    'q' => rights.add(CastlingRights::BLACK_QUEEN),
                    _ => return Err(FenError::BadCastling(castling.to_string())),
                }
            }
        }
        board.castling = rights;

        // --- Field 4: en passant target -----------------------------------
        board.ep_square = if ep == "-" {
            None
        } else {
            Some(Square::from_algebraic(ep).ok_or_else(|| FenError::BadEnPassant(ep.to_string()))?)
        };

        // --- Fields 5 & 6: clocks -----------------------------------------
        board.halfmove_clock = halfmove
            .parse()
            .map_err(|_| FenError::BadNumber(halfmove.to_string()))?;
        board.fullmove_number = fullmove
            .parse()
            .map_err(|_| FenError::BadNumber(fullmove.to_string()))?;

        board.hash = board.compute_hash();
        Ok(board)
    }

    /// Serialize the position back to FEN. Round-trips with [`Board::from_fen`].
    pub fn to_fen(&self) -> String {
        let mut s = String::new();

        // Field 1: piece placement, rank 8 down to rank 1.
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                match self.piece_at(sq) {
                    Some(p) => {
                        if empty > 0 {
                            s.push_str(&empty.to_string());
                            empty = 0;
                        }
                        s.push(p.to_char());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }

        // Field 2: side to move.
        s.push(' ');
        s.push(if self.side_to_move == Color::White { 'w' } else { 'b' });

        // Field 3: castling.
        s.push(' ');
        s.push_str(&self.castling_string());

        // Field 4: en passant.
        s.push(' ');
        match self.ep_square {
            Some(sq) => s.push_str(&sq.to_string()),
            None => s.push('-'),
        }

        // Fields 5 & 6: clocks.
        s.push(' ');
        s.push_str(&self.halfmove_clock.to_string());
        s.push(' ');
        s.push_str(&self.fullmove_number.to_string());

        s
    }
}
