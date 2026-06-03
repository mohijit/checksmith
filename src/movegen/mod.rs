//! Move generation: move encoding, attack tables, make/unmake, and legal-move
//! generation validated by perft.
//!
//! Dependency direction: this layer sits on top of [`crate::board`] and adds
//! everything needed to *play* moves, without yet knowing anything about search
//! or evaluation.

pub mod attacks;
pub mod generate;
pub mod legal;
pub mod make_move;
pub mod moves;

pub use attacks::is_square_attacked;
pub use make_move::{NullUndo, Undo};
pub use moves::{Move, MoveFlag, MoveList};
