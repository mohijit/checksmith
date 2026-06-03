//! NNUE weight file format and quantized inference.
//!
//! ## Weight file format (`.nnue`)
//!
//! Binary, little-endian, no compression.
//!
//! ```text
//! Offset  Size   Type    Field
//! ──────────────────────────────────────────────────────────────────
//!  0       4     u8[4]   Magic: b"CKSM"
//!  4       4     u32le   Version (currently 1)
//!  8       4     u32le   INPUT_FEATURES (must equal 40,960)
//! 12       4     u32le   HIDDEN_SIZE   (must equal 512)
//! 16       4     u32le   L1_SIZE       (must equal 32)
//! 20       4     u32le   L2_SIZE       (must equal 32)
//! ──────────────────────────────────────────────────────────────────
//!  24          Feature transformer
//!               bias   : HIDDEN_SIZE   × i16  (1,024 bytes)
//!               weight : INPUT_FEATURES × HIDDEN_SIZE × i16  (41,943,040 bytes)
//! ──────────────────────────────────────────────────────────────────
//!  L1 layer
//!               bias   : L1_SIZE × i32  (128 bytes)
//!               weight : (HIDDEN_SIZE × 2) × L1_SIZE × i8  (32,768 bytes)
//! ──────────────────────────────────────────────────────────────────
//!  L2 layer
//!               bias   : L2_SIZE × i32  (128 bytes)
//!               weight : L1_SIZE × L2_SIZE × i8  (1,024 bytes)
//! ──────────────────────────────────────────────────────────────────
//!  Output layer
//!               bias   : 1 × i32  (4 bytes)
//!               weight : L2_SIZE × i8  (32 bytes)
//! ──────────────────────────────────────────────────────────────────
//! ```
//!
//! ## Quantization scheme
//!
//! | Constant | Value | Role |
//! |----------|-------|------|
//! | `QA`     | 127   | ClampedReLU ceiling for feature-transformer output |
//! | `QB`     | 127   | Scale divisor between subsequent layers |
//!
//! **Feature transformer → L1 input**:
//! Each accumulator value is clamped to `[0, QA]` (ClampedReLU).
//! The two halves (STM and non-STM) are concatenated: `512` values.
//!
//! **L1 pre-activation**:
//! `l1[i] = clamp(Σ_j (input[j] × l1_weight[j][i]) / QA + l1_bias[i], 0, QB)`
//!
//! **L2 pre-activation**:
//! `l2[i] = clamp(Σ_j (l1[j] × l2_weight[j][i]) / QB + l2_bias[i], 0, QB)`
//!
//! **Output**:
//! `out = (Σ_j (l2[j] × out_weight[j]) + out_bias) / QB`
//!
//! The final value is returned as centipawns.  At training time the output
//! scale is calibrated so that the network's raw output in centipawns is
//! consistent with the handcrafted evaluator's scale.
//!
//! ## Inference ordering: STM first
//!
//! The side-to-move (STM) accumulator is placed first in the concatenated
//! input.  This lets the network learn "I am ahead" vs "opponent is ahead"
//! without needing separate output heads.

use super::accumulator::{Accumulator, FeatureWeights};
use super::features::{HIDDEN_SIZE, INPUT_FEATURES, L1_SIZE, L2_SIZE, QA, QB};
use crate::board::Color;
use std::io;
use std::path::Path;

/// The file-format magic bytes that identify a Checksmith `.nnue` weight file.
pub const MAGIC: &[u8; 4] = b"CKSM";
pub const VERSION: u32 = 1;

/// A fully-loaded NNUE network ready for inference.
///
/// Wrap in `Arc` to share a single loaded network across multiple search threads
/// without copying the large weight arrays.
pub struct Network {
    pub feature_weights: FeatureWeights,
    /// L1 layer biases (`L1_SIZE` values, i32).
    pub l1_bias:   Vec<i32>,
    /// L1 layer weights (`(HIDDEN_SIZE * 2) × L1_SIZE`, i8, row-major by input neuron).
    pub l1_weight: Vec<i8>,
    /// L2 layer biases (`L2_SIZE` values, i32).
    pub l2_bias:   Vec<i32>,
    /// L2 layer weights (`L1_SIZE × L2_SIZE`, i8, row-major by L1 neuron).
    pub l2_weight: Vec<i8>,
    /// Output layer bias (1 value, i32).
    pub out_bias:   i32,
    /// Output layer weights (`L2_SIZE` values, i8).
    pub out_weight: Vec<i8>,
}

impl Network {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a network with all weights and biases set to zero.
    ///
    /// A zero network evaluates every position as 0 centipawns, which is a
    /// valid (if useless) evaluator.  Useful for unit tests and as a baseline.
    pub fn zeros() -> Self {
        Network {
            feature_weights: FeatureWeights::zeros(),
            l1_bias:   vec![0i32; L1_SIZE],
            l1_weight: vec![0i8; (HIDDEN_SIZE * 2) * L1_SIZE],
            l2_bias:   vec![0i32; L2_SIZE],
            l2_weight: vec![0i8; L1_SIZE * L2_SIZE],
            out_bias:   0,
            out_weight: vec![0i8; L2_SIZE],
        }
    }

    /// Load a network from a Checksmith `.nnue` binary file.
    ///
    /// Returns an `io::Error` if:
    /// * The file cannot be read.
    /// * The magic bytes or version do not match.
    /// * The architecture constants in the file differ from the compiled-in ones.
    /// * The file is truncated.
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let mut cur = 0usize;

        macro_rules! consume {
            ($n:expr) => {{
                if cur + $n > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("NNUE file truncated at byte {}", cur),
                    ));
                }
                let slice = &data[cur..cur + $n];
                cur += $n;
                slice
            }};
        }

        // ── Header ──────────────────────────────────────────────────────────

        if consume!(4) != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a Checksmith .nnue file"));
        }
        let version = u32_le(consume!(4));
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .nnue version {} (expected {})", version, VERSION),
            ));
        }

        let file_input  = u32_le(consume!(4)) as usize;
        let file_hidden = u32_le(consume!(4)) as usize;
        let file_l1     = u32_le(consume!(4)) as usize;
        let file_l2     = u32_le(consume!(4)) as usize;

        if file_input != INPUT_FEATURES || file_hidden != HIDDEN_SIZE
            || file_l1 != L1_SIZE || file_l2 != L2_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "architecture mismatch: file=({},{},{},{}) compiled=({},{},{},{})",
                    file_input, file_hidden, file_l1, file_l2,
                    INPUT_FEATURES, HIDDEN_SIZE, L1_SIZE, L2_SIZE,
                ),
            ));
        }

        // ── Feature transformer ──────────────────────────────────────────────

        let ft_bias   = read_i16_vec(consume!(HIDDEN_SIZE * 2));
        let ft_weight = read_i16_vec(consume!(INPUT_FEATURES * HIDDEN_SIZE * 2));

        // ── L1 ──────────────────────────────────────────────────────────────

        let l1_bias   = read_i32_vec(consume!(L1_SIZE * 4));
        let l1_weight = consume!((HIDDEN_SIZE * 2) * L1_SIZE)
            .iter().map(|&b| b as i8).collect::<Vec<i8>>();

        // ── L2 ──────────────────────────────────────────────────────────────

        let l2_bias   = read_i32_vec(consume!(L2_SIZE * 4));
        let l2_weight = consume!(L1_SIZE * L2_SIZE)
            .iter().map(|&b| b as i8).collect::<Vec<i8>>();

        // ── Output ──────────────────────────────────────────────────────────

        let out_bias   = i32_le(consume!(4));
        let out_weight = consume!(L2_SIZE)
            .iter().map(|&b| b as i8).collect::<Vec<i8>>();

        Ok(Network {
            feature_weights: FeatureWeights { bias: ft_bias, weight: ft_weight },
            l1_bias, l1_weight,
            l2_bias, l2_weight,
            out_bias, out_weight,
        })
    }

    /// Serialize this network to a Checksmith `.nnue` binary file.
    ///
    /// Useful for saving a trained or procedurally-constructed network
    /// (e.g., a random initialisation for integration tests).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut buf = Vec::with_capacity(1 << 25); // ~42 MB for HIDDEN_SIZE=512

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(INPUT_FEATURES as u32).to_le_bytes());
        buf.extend_from_slice(&(HIDDEN_SIZE    as u32).to_le_bytes());
        buf.extend_from_slice(&(L1_SIZE        as u32).to_le_bytes());
        buf.extend_from_slice(&(L2_SIZE        as u32).to_le_bytes());

        // Feature transformer
        for &v in &self.feature_weights.bias   { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.feature_weights.weight { buf.extend_from_slice(&v.to_le_bytes()); }

        // L1
        for &v in &self.l1_bias   { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.l1_weight { buf.push(v as u8); }

        // L2
        for &v in &self.l2_bias   { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.l2_weight { buf.push(v as u8); }

        // Output
        buf.extend_from_slice(&self.out_bias.to_le_bytes());
        for &v in &self.out_weight { buf.push(v as u8); }

        std::fs::write(path, &buf)
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// Run quantized inference from the given accumulator.
    ///
    /// Returns an estimate in centipawns from `side_to_move`'s perspective
    /// (positive = the side to move stands better).
    ///
    /// The accumulator must be current for the position being evaluated
    /// (via [`AccumulatorStack::refresh`] or incremental updates).
    pub fn evaluate(&self, acc: &Accumulator, side_to_move: Color) -> i32 {
        // ── Step 1: ClampedReLU, STM first ──────────────────────────────────
        //
        // Concatenate side-to-move (STM) and opponent (NSTM) accumulators.
        // The STM accumulator is placed first; the network learns "my advantage"
        // directly from the ordering.
        let (stm, nstm) = match side_to_move {
            Color::White => (&acc.white.values, &acc.black.values),
            Color::Black => (&acc.black.values, &acc.white.values),
        };

        // input[j] ∈ [0, QA] after ClampedReLU.
        let mut input = [0i32; HIDDEN_SIZE * 2];
        for (j, &v) in stm.iter().enumerate() {
            input[j] = v.clamp(0, QA as i16) as i32;
        }
        for (j, &v) in nstm.iter().enumerate() {
            input[HIDDEN_SIZE + j] = v.clamp(0, QA as i16) as i32;
        }

        // ── Step 2: L1 linear + ClampedReLU ─────────────────────────────────
        //
        // l1_weight is row-major by input neuron: weight[input_neuron * L1_SIZE + l1_neuron]
        let mut l1 = [0i32; L1_SIZE];
        for j in 0..HIDDEN_SIZE * 2 {
            let base = j * L1_SIZE;
            let inp = input[j];
            for i in 0..L1_SIZE {
                l1[i] += inp * self.l1_weight[base + i] as i32;
            }
        }
        for i in 0..L1_SIZE {
            // Divide by QA to account for the [0,QA] scale of the input, then
            // add bias and clamp to [0, QB].
            l1[i] = (l1[i] / QA + self.l1_bias[i]).clamp(0, QB);
        }

        // ── Step 3: L2 linear + ClampedReLU ─────────────────────────────────
        let mut l2 = [0i32; L2_SIZE];
        for j in 0..L1_SIZE {
            let base = j * L2_SIZE;
            let lv = l1[j];
            for i in 0..L2_SIZE {
                l2[i] += lv * self.l2_weight[base + i] as i32;
            }
        }
        for i in 0..L2_SIZE {
            l2[i] = (l2[i] / QB + self.l2_bias[i]).clamp(0, QB);
        }

        // ── Step 4: Output linear ────────────────────────────────────────────
        let mut out = self.out_bias;
        for j in 0..L2_SIZE {
            out += l2[j] * self.out_weight[j] as i32;
        }

        // Divide by QB to get centipawns. The output_scale is embedded in
        // the out_weight values during training.
        out / QB
    }
}

// ─── Byte-reading helpers ─────────────────────────────────────────────────────

#[inline]
fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().expect("slice length 4"))
}

#[inline]
fn i32_le(b: &[u8]) -> i32 {
    i32::from_le_bytes(b.try_into().expect("slice length 4"))
}

fn read_i16_vec(b: &[u8]) -> Vec<i16> {
    b.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn read_i32_vec(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Color, STARTING_FEN};
    use crate::nnue::accumulator::AccumulatorStack;
    use std::sync::Arc;

    fn zero_net() -> Arc<Network> { Arc::new(Network::zeros()) }

    #[test]
    fn zero_network_outputs_zero() {
        let net = zero_net();
        let board = Board::from_fen(STARTING_FEN).unwrap();
        let mut stack = AccumulatorStack::new(&net.feature_weights);
        stack.refresh(&board, &net.feature_weights);
        let score = net.evaluate(&stack.current, board.side_to_move);
        assert_eq!(score, 0, "zero-weight network must evaluate to 0");
    }

    #[test]
    fn zero_network_symmetric() {
        // The starting position is symmetric; a zero network should return 0
        // regardless of which side is to move.
        let net = zero_net();
        let board_w = Board::from_fen(STARTING_FEN).unwrap();
        let board_b = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        ).unwrap();

        let score_w = {
            let mut s = AccumulatorStack::new(&net.feature_weights);
            s.refresh(&board_w, &net.feature_weights);
            net.evaluate(&s.current, Color::White)
        };
        let score_b = {
            let mut s = AccumulatorStack::new(&net.feature_weights);
            s.refresh(&board_b, &net.feature_weights);
            net.evaluate(&s.current, Color::Black)
        };
        assert_eq!(score_w, 0);
        assert_eq!(score_b, 0);
    }

    #[test]
    fn file_round_trip() {
        // Build a non-trivial network, save it, reload it, verify all weights match.
        let mut net = Network::zeros();
        // Set some distinguishable values.
        net.feature_weights.bias[0] = 42;
        net.feature_weights.bias[1] = -7;
        net.feature_weights.weight[100] = 123;
        net.l1_bias[0] = 1000;
        net.l1_weight[0] = 55;
        net.l2_bias[5] = -500;
        net.l2_weight[3] = -33;
        net.out_bias = 99;
        net.out_weight[0] = 11;

        let dir = std::env::temp_dir();
        let path = dir.join("checksmith_test_round_trip.nnue");
        net.save(&path).expect("save failed");

        let loaded = Network::load(&path).expect("load failed");
        assert_eq!(loaded.feature_weights.bias[0], 42);
        assert_eq!(loaded.feature_weights.bias[1], -7);
        assert_eq!(loaded.feature_weights.weight[100], 123);
        assert_eq!(loaded.l1_bias[0], 1000);
        assert_eq!(loaded.l1_weight[0], 55);
        assert_eq!(loaded.l2_bias[5], -500);
        assert_eq!(loaded.l2_weight[3], -33);
        assert_eq!(loaded.out_bias, 99);
        assert_eq!(loaded.out_weight[0], 11);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let bad: Vec<u8> = b"XXXX\x01\x00\x00\x00".to_vec();
        let dir = std::env::temp_dir();
        let path = dir.join("checksmith_bad_magic.nnue");
        std::fs::write(&path, &bad).unwrap();
        let result = Network::load(&path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn inference_does_not_overflow_with_saturated_inputs() {
        // Max possible accumulator values clamped to QA; verify no integer panic.
        let net = zero_net();
        let acc = crate::nnue::accumulator::Accumulator {
            white: crate::nnue::accumulator::HalfAccumulator { values: vec![i16::MAX; HIDDEN_SIZE] },
            black: crate::nnue::accumulator::HalfAccumulator { values: vec![i16::MAX; HIDDEN_SIZE] },
        };
        // Should not panic, even with saturated inputs (zero weights → output = 0).
        let score = net.evaluate(&acc, Color::White);
        assert_eq!(score, 0);
    }
}
