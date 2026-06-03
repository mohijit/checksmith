"""
Export a trained PyTorch NNUE checkpoint to Checksmith's .nnue binary format.

Binary layout (little-endian) — must match src/nnue/network.rs exactly:

  Offset   Size    Field
  ───────────────────────────────────────────────────────
    0        4     Magic: b"CKSM"
    4        4     Version: 1 (u32le)
    8        4     INPUT_FEATURES = 40960 (u32le)
   12        4     HIDDEN_SIZE   = 512   (u32le)
   16        4     L1_SIZE       = 32    (u32le)
   20        4     L2_SIZE       = 32    (u32le)
  ───────────────────────────────────────────────────────
   24       1024   Feature-transformer bias   (HIDDEN_SIZE × i16)
  1048   41943040  Feature-transformer weight (INPUT_FEATURES × HIDDEN_SIZE × i16)
  ───────────────────────────────────────────────────────
         128       L1 bias   (L1_SIZE × i32)
       32768       L1 weight ((HIDDEN_SIZE×2) × L1_SIZE × i8)
  ───────────────────────────────────────────────────────
         128       L2 bias   (L2_SIZE × i32)
        1024       L2 weight (L1_SIZE × L2_SIZE × i8)
  ───────────────────────────────────────────────────────
           4       Output bias   (1 × i32)
          32       Output weight (L2_SIZE × i8)

Quantization:
    FT weights  × QA (127) → i16  (range ~[-16000, 16000])
    L1 weights  × QB (127) → i8   (range [-127, 127])
    L2 weights  × QB (127) → i8
    Output weight × QB     → i8
    All biases: scaled to match quantized layer inputs, stored as i32.

Usage:
    python train/02_export.py --ckpt train/model/best.pt
                              --out  train/model/checksmith.nnue
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    sys.exit("Missing PyTorch: pip install torch")

import numpy as np

# ── Architecture constants ────────────────────────────────────────────────────

MAGIC          = b"CKSM"
VERSION        = 1
INPUT_FEATURES = 40_960
HIDDEN_SIZE    = 512
L1_SIZE        = 32
L2_SIZE        = 32
QA             = 127
QB             = 127


def clamp_i8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -127, 127).astype(np.int8)

def clamp_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def export(ckpt_path: str, out_path: str) -> None:
    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    def get(key: str) -> np.ndarray:
        return state[key].float().detach().numpy()

    # ── Feature transformer ───────────────────────────────────────────────────
    # ft_bias   shape: (HIDDEN_SIZE,)   float in [-1, 1] (trained)
    # ft_weight shape: (INPUT_FEATURES, HIDDEN_SIZE)
    ft_bias_f   = get("ft_bias")
    ft_weight_f = get("ft_weight.weight")  # (INPUT_FEATURES, HIDDEN_SIZE)

    # Scale to i16: weights × QA; bias × QA (so accumulator starts at bias×QA)
    ft_bias_q   = clamp_i16(ft_bias_f   * QA)
    ft_weight_q = clamp_i16(ft_weight_f * QA)

    # ── L1 ────────────────────────────────────────────────────────────────────
    # l1.weight shape: (L1_SIZE, HIDDEN_SIZE*2) in PyTorch (output_first)
    # Rust expects row-major by INPUT neuron: shape (HIDDEN_SIZE*2, L1_SIZE)
    l1_bias_f   = get("l1.bias")                      # (L1_SIZE,)
    l1_weight_f = get("l1.weight").T                   # (HIDDEN_SIZE*2, L1_SIZE)

    # The L1 receives ClampedReLU output in [0, QA].
    # Rust computes: sum(input[j] * weight[j][i]) / QA + bias → clamp [0, QB]
    # So weight must be in [−127, 127] and bias in int32 units (QB scale).
    l1_weight_q = clamp_i8(l1_weight_f * QB)
    # Bias: absorb the /QA scale from the accumulator side.
    # After the integer dot-product is divided by QA, bias is added as-is.
    # We store bias × QB so Rust's "+= bias" gives the right scale.
    l1_bias_q   = np.round(l1_bias_f * QB).astype(np.int32)

    # ── L2 ────────────────────────────────────────────────────────────────────
    l2_bias_f   = get("l2.bias")                       # (L2_SIZE,)
    l2_weight_f = get("l2.weight").T                   # (L1_SIZE, L2_SIZE)

    l2_weight_q = clamp_i8(l2_weight_f * QB)
    l2_bias_q   = np.round(l2_bias_f * QB).astype(np.int32)

    # ── Output ────────────────────────────────────────────────────────────────
    out_bias_f   = get("out.bias")                     # (1,)
    out_weight_f = get("out.weight").flatten()         # (L2_SIZE,)

    # The output layer takes ClampedReLU L2 activations in [0, 1] (float) and
    # produces centipawns directly.  In Rust, l2 values are in [0, QB] because
    # they equal QB * l2_float.  The output formula in Rust is:
    #
    #   out_cp = (Σ l2[j] * out_weight[j] + out_bias) / QB
    #          = Σ (QB * l2_float[j]) * out_weight_q[j] / QB + out_bias_q / QB
    #          = Σ l2_float[j] * out_weight_q[j]           + out_bias_q / QB
    #
    # For this to equal the float centipawn output (Σ l2_float[j] * out_weight_f[j] + out_bias_f):
    #   out_weight_q = round(out_weight_f)   [no extra scale]
    #   out_bias_q   = round(out_bias_f * QB)
    #
    # CP_SCALE is NOT applied here — it is already baked into the network's
    # output values (the loss uses sigmoid(out / CP_SCALE), so the trained
    # out_weight values already encode centipawn scale).
    out_weight_q = clamp_i8(np.round(out_weight_f))
    out_bias_q   = np.round(out_bias_f * QB).astype(np.int32)

    # ── Sanity-check shapes ───────────────────────────────────────────────────
    assert ft_bias_q.shape   == (HIDDEN_SIZE,),            f"FT bias shape: {ft_bias_q.shape}"
    assert ft_weight_q.shape == (INPUT_FEATURES, HIDDEN_SIZE), f"FT weight shape: {ft_weight_q.shape}"
    assert l1_bias_q.shape   == (L1_SIZE,),                f"L1 bias shape: {l1_bias_q.shape}"
    assert l1_weight_q.shape == (HIDDEN_SIZE * 2, L1_SIZE), f"L1 weight shape: {l1_weight_q.shape}"
    assert l2_bias_q.shape   == (L2_SIZE,),                f"L2 bias shape: {l2_bias_q.shape}"
    assert l2_weight_q.shape == (L1_SIZE, L2_SIZE),        f"L2 weight shape: {l2_weight_q.shape}"
    assert out_bias_q.shape  == (1,),                      f"Out bias shape: {out_bias_q.shape}"
    assert out_weight_q.shape == (L2_SIZE,),               f"Out weight shape: {out_weight_q.shape}"

    # ── Write binary ─────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", INPUT_FEATURES))
        f.write(struct.pack("<I", HIDDEN_SIZE))
        f.write(struct.pack("<I", L1_SIZE))
        f.write(struct.pack("<I", L2_SIZE))

        # Feature transformer
        f.write(ft_bias_q.astype("<i2").tobytes())
        f.write(ft_weight_q.astype("<i2").tobytes())

        # L1
        f.write(l1_bias_q.astype("<i4").tobytes())
        f.write(l1_weight_q.astype("i1").tobytes())

        # L2
        f.write(l2_bias_q.astype("<i4").tobytes())
        f.write(l2_weight_q.astype("i1").tobytes())

        # Output
        f.write(out_bias_q.astype("<i4").tobytes())
        f.write(out_weight_q.astype("i1").tobytes())

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")

    # ── Quick stats ───────────────────────────────────────────────────────────
    print(f"\nQuantization summary:")
    print(f"  FT  weight  range: [{ft_weight_q.min():6d}, {ft_weight_q.max():6d}]  (i16)")
    print(f"  FT  bias    range: [{ft_bias_q.min():6d}, {ft_bias_q.max():6d}]  (i16)")
    print(f"  L1  weight  range: [{l1_weight_q.min():6d}, {l1_weight_q.max():6d}]  (i8)")
    print(f"  L2  weight  range: [{l2_weight_q.min():6d}, {l2_weight_q.max():6d}]  (i8)")
    print(f"  Out weight  range: [{out_weight_q.min():6d}, {out_weight_q.max():6d}]  (i8)")
    print(f"\nUsage:  setoption name EvalFile value {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="train/model/best.pt",
                    help="PyTorch checkpoint (produced by 01_train.py)")
    ap.add_argument("--out",  default="train/model/checksmith.nnue",
                    help="Output .nnue path")
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
