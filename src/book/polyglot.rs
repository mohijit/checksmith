//! Polyglot opening-book format support.
//!
//! A Polyglot `.bin` file is a sorted array of 16-byte records, each of the form:
//!
//! ```text
//! key    : u64   (Polyglot Zobrist hash of the position)
//! move   : u16   (packed from/to/promo in bits 0-14)
//! weight : u16   (relative frequency; 0 is legal but discouraged)
//! learn  : u32   (ignored — learning data written by some GUIs)
//! ```
//!
//! The file is big-endian and contains no header.

use super::polyglot_random::RANDOM;
use crate::board::{Board, CastlingRights, Color, PieceType, Square};
use crate::movegen::{Move, MoveFlag, MoveList};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

// ── Piece index in the Polyglot table ─────────────────────────────────────────

fn piece_index(color: Color, pt: PieceType) -> usize {
    let type_idx = match pt {
        PieceType::Pawn   => 0,
        PieceType::Knight => 2,
        PieceType::Bishop => 4,
        PieceType::Rook   => 6,
        PieceType::Queen  => 8,
        PieceType::King   => 10,
    };
    // Polyglot: black=even, white=odd.
    // Color::White.index()=0 (even), Color::Black.index()=1 (odd) — so we flip.
    type_idx + color.opposite().index()
}

// ── Polyglot Zobrist hash ──────────────────────────────────────────────────────

/// Compute the Polyglot Zobrist hash for `board`.
///
/// This uses the 781-constant Polyglot table — NOT the engine's own Zobrist
/// table.  The two are intentionally different; the engine hash only needs
/// internal consistency, while this hash must match every other Polyglot
/// implementation in the world so that `.bin` files are interoperable.
pub fn poly_hash(board: &Board) -> u64 {
    let mut h: u64 = 0;

    // Pieces
    for sq_idx in 0..64u8 {
        let sq = Square::new(sq_idx);
        if let Some(piece) = board.piece_at(sq) {
            let pi = piece_index(piece.color, piece.piece_type);
            h ^= RANDOM[pi * 64 + sq_idx as usize];
        }
    }

    // Castling rights
    if board.castling.has(CastlingRights::WHITE_KING)  { h ^= RANDOM[768]; }
    if board.castling.has(CastlingRights::WHITE_QUEEN) { h ^= RANDOM[769]; }
    if board.castling.has(CastlingRights::BLACK_KING)  { h ^= RANDOM[770]; }
    if board.castling.has(CastlingRights::BLACK_QUEEN) { h ^= RANDOM[771]; }

    // En passant — Polyglot only XORs the file when a capture is actually
    // possible; otherwise the ep square is ignored.
    if let Some(ep_sq) = board.ep_square {
        if board.has_ep_captor(ep_sq) {
            h ^= RANDOM[772 + ep_sq.file() as usize];
        }
    }

    // Side to move
    if board.side_to_move == Color::White {
        h ^= RANDOM[780];
    }

    h
}

// ── Move encoding / decoding ───────────────────────────────────────────────────

/// Decode a Polyglot 16-bit move word into a legal [`Move`] for `board`, or
/// `None` if it does not match any legal move.
///
/// Polyglot bit layout:
/// ```text
/// bits  0-2   to_file   (0=a … 7=h)
/// bits  3-5   to_rank   (0=1 … 7=8)
/// bits  6-8   from_file
/// bits  9-11  from_rank
/// bits 12-14  promotion  (0=none 1=N 2=B 3=R 4=Q)
/// ```
///
/// Castling is stored as the king moving **to the rook's square**
/// (e1h1, e1a1, e8h8, e8a8), which must be translated to our king-to-king
/// destination (g1, c1, g8, c8).
pub fn decode_move(poly_move: u16, legal: &MoveList) -> Option<Move> {
    let to_file  = (poly_move & 0x0007) as u8;
    let to_rank  = ((poly_move >> 3) & 0x0007) as u8;
    let fr_file  = ((poly_move >> 6) & 0x0007) as u8;
    let fr_rank  = ((poly_move >> 9) & 0x0007) as u8;
    let promo    = (poly_move >> 12) & 0x0007;

    let from = Square::from_file_rank(fr_file, fr_rank);
    let mut to = Square::from_file_rank(to_file, to_rank);

    // Remap castling: Polyglot stores e1→h1 (king-side) and e1→a1 (queen-side).
    // Our engine stores e1→g1 and e1→c1.
    let is_castle_wk = from == Square::E1 && to == Square::H1;
    let is_castle_wq = from == Square::E1 && to == Square::A1;
    let is_castle_bk = from == Square::E8 && to == Square::H8;
    let is_castle_bq = from == Square::E8 && to == Square::A8;
    if is_castle_wk { to = Square::G1; }
    if is_castle_wq { to = Square::C1; }
    if is_castle_bk { to = Square::G8; }
    if is_castle_bq { to = Square::C8; }

    // Map Polyglot promotion code to our MoveFlag-derived promotion piece.
    let poly_promo_pt: Option<PieceType> = match promo {
        1 => Some(PieceType::Knight),
        2 => Some(PieceType::Bishop),
        3 => Some(PieceType::Rook),
        4 => Some(PieceType::Queen),
        _ => None,
    };

    // Find the matching legal move by (from, to, promotion_piece).
    legal.iter().copied().find(|&mv| {
        mv.from() == from
            && mv.to() == to
            && mv.promotion_piece() == poly_promo_pt
    })
}

/// Encode a [`Move`] into a Polyglot 16-bit word.
///
/// Castling moves are re-encoded with the king landing on the rook's square.
pub fn encode_move(mv: Move) -> u16 {
    let mut to = mv.to();

    if mv.flag() == MoveFlag::KingCastle {
        to = match mv.from() {
            Square::E1 => Square::H1,
            Square::E8 => Square::H8,
            _ => to,
        };
    } else if mv.flag() == MoveFlag::QueenCastle {
        to = match mv.from() {
            Square::E1 => Square::A1,
            Square::E8 => Square::A8,
            _ => to,
        };
    }

    let promo: u16 = match mv.promotion_piece() {
        Some(PieceType::Knight) => 1,
        Some(PieceType::Bishop) => 2,
        Some(PieceType::Rook)   => 3,
        Some(PieceType::Queen)  => 4,
        _                       => 0,
    };

    (to.file() as u16)
        | ((to.rank() as u16) << 3)
        | ((mv.from().file() as u16) << 6)
        | ((mv.from().rank() as u16) << 9)
        | (promo << 12)
}

// ── PolyEntry ─────────────────────────────────────────────────────────────────

/// One record from a Polyglot `.bin` file.
#[derive(Clone, Copy, Debug)]
pub struct PolyEntry {
    pub key:    u64,
    pub mv:     u16,
    pub weight: u16,
    #[allow(dead_code)]
    pub learn:  u32,
}

impl PolyEntry {
    const BYTES: usize = 16;

    fn from_bytes(buf: &[u8; 16]) -> PolyEntry {
        PolyEntry {
            key:    u64::from_be_bytes(buf[0..8].try_into().unwrap()),
            mv:     u16::from_be_bytes(buf[8..10].try_into().unwrap()),
            weight: u16::from_be_bytes(buf[10..12].try_into().unwrap()),
            learn:  u32::from_be_bytes(buf[12..16].try_into().unwrap()),
        }
    }
}

// ── File loading ──────────────────────────────────────────────────────────────

/// Read and return all entries from a Polyglot `.bin` file.
pub fn load_polyglot<P: AsRef<Path>>(path: P) -> io::Result<Vec<PolyEntry>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();
    let mut buf = [0u8; PolyEntry::BYTES];
    loop {
        match reader.read_exact(&mut buf) {
            Ok(()) => entries.push(PolyEntry::from_bytes(&buf)),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(entries)
}

/// Return all entries whose `key` matches `hash`, using binary search.
pub fn probe_entries<'a>(entries: &'a [PolyEntry], hash: u64) -> &'a [PolyEntry] {
    let start = entries.partition_point(|e| e.key < hash);
    let end   = entries[start..].partition_point(|e| e.key == hash) + start;
    &entries[start..end]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::STARTING_FEN;

    #[test]
    fn poly_hash_startpos_matches_known_value() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let h = poly_hash(&board);
        assert_eq!(
            h, 0x463b96181691fc9c,
            "Polyglot startpos hash mismatch: got {:#018x}", h
        );
    }

    #[test]
    fn encode_decode_round_trips_quiet_move() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let legal = board.legal_moves();
        // Find e2e4
        let mv = legal.iter().copied()
            .find(|m| m.to_uci() == "e2e4")
            .expect("e2e4 must be legal from startpos");
        let encoded = encode_move(mv);
        let decoded = decode_move(encoded, &legal).expect("decode must succeed");
        assert_eq!(decoded, mv);
    }

    #[test]
    fn encode_decode_round_trips_castle() {
        // Position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.0-0 — just test encoding
        // We check that white kingside castle encodes to e1h1 in Polyglot.
        use crate::movegen::MoveFlag;
        let mv = Move::new(Square::E1, Square::G1, MoveFlag::KingCastle);
        let encoded = encode_move(mv);
        let to_file = (encoded & 0x7) as u8;
        let to_rank = ((encoded >> 3) & 0x7) as u8;
        assert_eq!(to_file, 7, "Polyglot castle should land on h-file");
        assert_eq!(to_rank, 0, "Polyglot castle should land on rank 1");
    }
}
