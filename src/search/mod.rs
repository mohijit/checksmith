//! Search: choosing a move by looking ahead.
//!
//! The engine plays by exploring the tree of move sequences and scoring leaf
//! positions with [`crate::eval`]. The pieces:
//!
//! * [`negamax`] — the alpha-beta core, wrapped in a stoppable [`Searcher`].
//! * [`iterative`] — time-managed iterative deepening ([`think`]) used by UCI.
//!
//! ## Score scale
//!
//! Scores are centipawns from the side-to-move's perspective. Sentinels sit far
//! outside any plausible material score:
//!
//! * [`INFINITY`] — alpha-beta window bounds; no real score reaches it.
//! * [`MATE`] — magnitude of a checkmate. A mate `n` plies away scores
//!   `MATE - n`, so the search prefers mating sooner (and surviving longer when
//!   losing).

pub mod draw;
pub mod iterative;
pub mod negamax;
pub mod ordering;
pub mod params;
pub mod quiescence;
pub mod see;
pub mod smp;
pub mod time;
pub mod tt;

pub use iterative::{think, SearchInfo, SearchLimits, MAX_DEPTH};
pub use negamax::{search, SearchResult, Searcher};
pub use see::see;
pub use time::TimeManager;
pub use tt::{Bound, TranspositionTable};

/// Window bound; larger than any achievable evaluation.
pub const INFINITY: i32 = 30_000;

/// Magnitude of a checkmate score.
pub const MATE: i32 = 29_000;

/// Scores with magnitude above this represent a forced mate, not a normal eval.
pub const MATE_IN_MAX: i32 = MATE - 1_000;

/// True if `score` encodes a forced mate rather than a positional evaluation.
pub fn is_mate_score(score: i32) -> bool {
    score.abs() > MATE_IN_MAX
}

/// For a mate score, the number of plies until mate.
pub fn mate_distance_plies(score: i32) -> i32 {
    MATE - score.abs()
}
