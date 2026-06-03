//! NNUE (Efficiently Updatable Neural Network) infrastructure.
//!
//! ## Module layout
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`features`] | HalfKP feature indexing, active-feature extraction |
//! | [`accumulator`] | Incremental accumulator stack (`AccumulatorStack`) |
//! | [`network`] | Binary `.nnue` weight file format + quantized inference |
//!
//! ## Quick start
//!
//! ```ignore
//! // Load a trained network.
//! let net = Arc::new(Network::load(Path::new("my.nnue"))?);
//! // Or use a zero network for testing.
//! let net = Arc::new(Network::zeros());
//!
//! // Build an accumulator for a board position.
//! let mut stack = AccumulatorStack::new(&net.feature_weights);
//! stack.refresh(&board, &net.feature_weights);
//!
//! // Evaluate (returns centipawns from side-to-move's perspective).
//! let centipawns = net.evaluate(&stack.current, board.side_to_move);
//! ```
//!
//! ## Activating incremental updates in the search
//!
//! For production performance, maintain the accumulator inside
//! [`crate::search::negamax::Searcher`]:
//!
//! 1. Add `acc: AccumulatorStack` as a field alongside `hash_history`.
//! 2. Before each `board.make_move(mv)`, call:
//!    ```ignore
//!    let (rm_w, add_w, rm_b, add_b) = move_feature_deltas(&board, mv, king_sqs);
//!    self.acc.push_with_deltas(&rm_w, &add_w, &rm_b, &add_b, &net.feature_weights);
//!    // King move? call self.acc.current.refresh(board_after, &net.feature_weights)
//!    ```
//! 3. After each `board.unmake_move(mv, undo)`, call `self.acc.pop()`.
//!
//! This reduces feature-transformer work from O(40,960 × 256) per leaf to
//! O(30 × 256) per make/unmake — a ~1,365× reduction at the hottest call site.

pub mod accumulator;
pub mod features;
pub mod network;

pub use accumulator::AccumulatorStack;
pub use features::{active_features, INPUT_FEATURES, HIDDEN_SIZE, L1_SIZE, L2_SIZE, QA, QB};
pub use network::Network;
