//! Search stability and debugging framework.
//!
//! [`SearchVerifier`] runs the engine on a suite of positions with known
//! expected results and checks every correctness invariant:
//!
//! * Root move is in `legal_moves()` — the most critical safety property.
//! * PV moves are legal in sequence — TT collisions can corrupt PVs.
//! * Two identical searches produce identical results — non-determinism flag.
//! * Mate positions score correctly — validates `score_to_tt`/`score_from_tt`.
//! * Score is within `[−MATE, MATE]` — arithmetic overflow sentinel.
//! * TT has the root's best move after search — store/probe round-trip.
//!
//! ## Usage
//!
//! ```sh
//! checksmith verify [depth]   # run standard suite (default depth 5)
//! ```
//!
//! Or in code:
//!
//! ```rust,ignore
//! let verifier = SearchVerifier::new();
//! let results = verifier.run_all(5);
//! verifier.print_report(&results);
//! ```

pub mod verifier;
pub use verifier::{run_verify, SearchVerifier, VerificationResult, VerifyPosition};
