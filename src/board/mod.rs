//! Board representation.
//!
//! [`Board`] is the central data structure of the engine. It holds:
//!
//! * **Piece bitboards** — `[Bitboard; 6]`, one per [`PieceType`], merging both
//!   colors. Combined with the color bitboards this answers "where are all the
//!   rooks?" and "where is everything White owns?" in one operation each.
//! * **Color bitboards** — `[Bitboard; 2]`. Their union is the occupancy.
//! * **A mailbox** — `[Option<Piece>; 64]`, a redundant square-indexed view so
//!   "what is on e4?" is O(1) without scanning bitboards. The two
//!   representations are kept in sync by [`Board::put_piece`] /
//!   [`Board::remove_piece`].
//! * **Game state** — side to move, castling rights, en-passant square,
//!   halfmove clock, fullmove number, and the Zobrist [`hash`](Board::hash).

pub mod bitboard;
pub mod fen;
pub mod piece;
pub mod square;
pub mod zobrist;

pub use bitboard::Bitboard;
pub use fen::FenError;
pub use piece::{Color, Piece, PieceType};
pub use square::Square;

use std::fmt;
use zobrist::zobrist;

/// Standard chess starting position in FEN.
pub const STARTING_FEN: &str =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Castling availability, stored as a 4-bit mask.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash, Debug)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const WHITE_KING: u8 = 0b0001;
    pub const WHITE_QUEEN: u8 = 0b0010;
    pub const BLACK_KING: u8 = 0b0100;
    pub const BLACK_QUEEN: u8 = 0b1000;

    /// No castling rights.
    pub const NONE: CastlingRights = CastlingRights(0);
    /// All four castling rights (`KQkq`).
    pub const ALL: CastlingRights = CastlingRights(0b1111);

    /// True if the given flag (e.g. [`CastlingRights::WHITE_KING`]) is set.
    #[inline]
    pub const fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    /// Grant a castling right.
    #[inline]
    pub fn add(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Revoke a castling right.
    #[inline]
    pub fn remove(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}

/// A full chess position.
#[derive(Clone)]
pub struct Board {
    /// Piece occupancy by type, both colors merged. Index with `PieceType::index`.
    pieces: [Bitboard; 6],
    /// Piece occupancy by color. Index with `Color::index`.
    colors: [Bitboard; 2],
    /// Square-indexed view, kept in sync with the bitboards.
    mailbox: [Option<Piece>; 64],

    pub side_to_move: Color,
    pub castling: CastlingRights,
    /// The square a pawn may be captured on via en passant, if any.
    pub ep_square: Option<Square>,
    /// Plies since the last capture or pawn move (for the 50-move rule).
    pub halfmove_clock: u16,
    /// Starts at 1, incremented after each Black move.
    pub fullmove_number: u16,
    /// Zobrist hash of the position.
    pub hash: u64,
}

impl Board {
    /// An empty board with White to move and no rights.
    pub fn empty() -> Board {
        Board {
            pieces: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            mailbox: [None; 64],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
        }
    }

    /// The standard chess starting position.
    pub fn starting_position() -> Board {
        // STARTING_FEN is a compile-time constant we know is valid.
        Board::from_fen(STARTING_FEN).expect("STARTING_FEN must be valid")
    }

    // --- Queries -----------------------------------------------------------

    /// All pieces of a given type (both colors).
    #[inline]
    pub fn pieces(&self, pt: PieceType) -> Bitboard {
        self.pieces[pt.index()]
    }

    /// All pieces of a given color.
    #[inline]
    pub fn color(&self, c: Color) -> Bitboard {
        self.colors[c.index()]
    }

    /// Pieces of a given color and type, e.g. white rooks.
    #[inline]
    pub fn pieces_colored(&self, c: Color, pt: PieceType) -> Bitboard {
        self.pieces[pt.index()] & self.colors[c.index()]
    }

    /// Every occupied square.
    #[inline]
    pub fn occupancy(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    /// The piece on `sq`, if any.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.index()]
    }

    /// The square the king of color `c` stands on, if present.
    pub fn king_square(&self, c: Color) -> Option<Square> {
        self.pieces_colored(c, PieceType::King).lsb()
    }

    // --- Mutation (keeps both representations in sync) ----------------------
    //
    // These are the *only* two methods that change which pieces are on the
    // board. They keep the bitboards, the mailbox, and the Zobrist hash all in
    // sync, which is what lets `make_move` update the hash incrementally just by
    // calling them.

    /// Place `piece` on `sq`. Assumes `sq` is currently empty.
    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        self.pieces[piece.piece_type.index()].set(sq);
        self.colors[piece.color.index()].set(sq);
        self.mailbox[sq.index()] = Some(piece);
        self.hash ^= zobrist().piece(piece.color, piece.piece_type, sq);
    }

    /// Remove and return whatever piece is on `sq`.
    pub fn remove_piece(&mut self, sq: Square) -> Option<Piece> {
        let removed = self.mailbox[sq.index()].take();
        if let Some(piece) = removed {
            self.pieces[piece.piece_type.index()].clear(sq);
            self.colors[piece.color.index()].clear(sq);
            self.hash ^= zobrist().piece(piece.color, piece.piece_type, sq);
        }
        removed
    }

    // --- Hashing -----------------------------------------------------------

    /// True when there is at least one pawn of `side_to_move` that can actually
    /// capture on `ep_sq`.
    ///
    /// Polyglot requires this check before including the en-passant file in
    /// the Zobrist hash: just having an ep square is not enough — a captor
    /// must exist.
    pub fn has_ep_captor(&self, ep_sq: Square) -> bool {
        let attacker = self.side_to_move;
        let ep_file = ep_sq.file();
        // The capturing pawn sits on the rank *behind* the ep square from the
        // attacker's perspective.  For White ep the ep square is on rank 6
        // (0-indexed 5), so the captor is on rank 5 (0-indexed 4); for Black
        // it is rank 3 (0-indexed 2) vs rank 4 (0-indexed 3).
        let captor_rank = if attacker == Color::White { ep_sq.rank() - 1 } else { ep_sq.rank() + 1 };
        // Check both adjacent files.
        for &delta in &[-1i8, 1] {
            let f = ep_file as i8 + delta;
            if !(0..8).contains(&f) { continue; }
            let sq = Square::from_file_rank(f as u8, captor_rank);
            if let Some(piece) = self.piece_at(sq) {
                if piece.color == attacker && piece.piece_type == PieceType::Pawn {
                    return true;
                }
            }
        }
        false
    }

    /// Recompute the Zobrist hash from scratch.
    ///
    /// Used after FEN load. Make/unmake (Milestone 2) will instead update the
    /// hash incrementally.
    pub fn compute_hash(&self) -> u64 {
        let z = zobrist();
        let mut h = 0u64;

        for index in 0..64 {
            if let Some(piece) = self.mailbox[index] {
                h ^= z.piece(piece.color, piece.piece_type, Square::new(index as u8));
            }
        }
        if self.side_to_move == Color::Black {
            h ^= z.side();
        }
        h ^= z.castling(self.castling.0);
        if let Some(ep) = self.ep_square {
            h ^= z.ep_file(ep.file());
        }
        h
    }

    /// The castling rights rendered as a FEN field (`"KQkq"`, `"-"`, ...).
    pub fn castling_string(&self) -> String {
        let c = self.castling;
        if c.0 == 0 {
            return "-".to_string();
        }
        let mut s = String::new();
        if c.has(CastlingRights::WHITE_KING) {
            s.push('K');
        }
        if c.has(CastlingRights::WHITE_QUEEN) {
            s.push('Q');
        }
        if c.has(CastlingRights::BLACK_KING) {
            s.push('k');
        }
        if c.has(CastlingRights::BLACK_QUEEN) {
            s.push('q');
        }
        s
    }
}

/// Human-readable board diagram plus the position's metadata.
impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  +------------------------+")?;
        for rank in (0..8).rev() {
            write!(f, "{} |", rank + 1)?;
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                match self.piece_at(sq) {
                    Some(p) => write!(f, " {} ", p.to_char())?,
                    None => write!(f, " . ")?,
                }
            }
            writeln!(f, "|")?;
        }
        writeln!(f, "  +------------------------+")?;
        writeln!(f, "    a  b  c  d  e  f  g  h")?;
        writeln!(f)?;
        writeln!(f, "Side to move : {:?}", self.side_to_move)?;
        writeln!(f, "Castling     : {}", self.castling_string())?;
        writeln!(
            f,
            "En passant   : {}",
            self.ep_square
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".to_string())
        )?;
        writeln!(f, "Halfmove     : {}", self.halfmove_clock)?;
        writeln!(f, "Fullmove     : {}", self.fullmove_number)?;
        write!(f, "Zobrist hash : {:#018x}", self.hash)
    }
}
