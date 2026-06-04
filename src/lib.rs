//! Checksmith — a classical chess engine built from scratch.
//!
//! Milestone 1 scope: board representation, piece/square/bitboard types,
//! Zobrist hashing, and FEN parsing/serialization.
//!
//! The crate is organized so that later milestones (move generation, search,
//! evaluation, UCI) slot in as sibling modules without disturbing this one.

pub mod bench;
pub mod board;
pub mod book;
pub mod debug;
pub mod epd;
pub mod eval;
pub mod movegen;
pub mod nnue;
pub mod search;
pub mod selfplay;
pub mod tablebase;
pub mod tune;
pub mod uci;

pub use board::Board;
pub use eval::{Evaluator, HandcraftedEvaluator};
pub use movegen::{Move, MoveFlag, MoveList};
pub use search::{search, SearchResult, Searcher};
