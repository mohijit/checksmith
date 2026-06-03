"""
Train a HalfKP NNUE for Checksmith.

Architecture (matches src/nnue/):
    HalfKP features  → 40,960 sparse binary inputs (per side)
    Feature transformer  40,960 × 256  (i16 quantized, two halves concatenated)
    L1:              512 → 32   ReLU
    L2:              32  → 32   ReLU
    Output:           32 → 1    linear  (centipawns)

Training target:
    WDL label from game result (0.0 / 0.5 / 1.0 from side-to-move's perspective).
    Loss = MSE( sigmoid(output / CP_SCALE), target_wdl )

Usage:
    pip install torch chess numpy
    python train/01_train.py [--data train/data/positions.txt]
                             [--epochs 10] [--batch 4096] [--lr 0.001]
"""

import argparse
import math
import random
import struct
import sys
import time
from pathlib import Path

# ── Requirements ──────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    sys.exit("Missing PyTorch: pip install torch")

try:
    import chess
except ImportError:
    sys.exit("Missing python-chess: pip install chess")

import numpy as np

# ── Architecture constants (must match src/nnue/features.rs) ─────────────────

INPUT_FEATURES   = 40_960
HIDDEN_SIZE      = 512
L1_SIZE          = 32
L2_SIZE          = 32
QA               = 127
QB               = 127

# Win-probability sigmoid scale:  sigmoid(eval_cp / CP_SCALE) ≈ win_prob
CP_SCALE = 400.0

# ── HalfKP feature extraction ────────────────────────────────────────────────

PIECE_TYPE_INDEX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4,
}
FEATURES_PER_KING = 5 * 2 * 64  # 640


def halfkp_indices(board: chess.Board) -> tuple[list[int], list[int]]:
    """
    Compute active HalfKP feature indices for both perspectives.
    Returns (white_features, black_features).
    Matches the Rust implementation in src/nnue/features.rs.
    """
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return [], []

    bk_flip = bk ^ 56  # vertical flip: rank 1 ↔ rank 8

    wf, bf = [], []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue

        pt   = PIECE_TYPE_INDEX[piece.piece_type]
        sq_f = sq ^ 56

        # White perspective: friendly = color==WHITE → color_bit=0
        wc = 0 if piece.color == chess.WHITE else 1
        wf.append(wk * FEATURES_PER_KING + pt * 128 + wc * 64 + sq)

        # Black perspective: friendly = color==BLACK → color_bit=0; board flipped
        bc = 0 if piece.color == chess.BLACK else 1
        bf.append(bk_flip * FEATURES_PER_KING + pt * 128 + bc * 64 + sq_f)

    return wf, bf


# ── Dataset ──────────────────────────────────────────────────────────────────

class PositionDataset(Dataset):
    """
    Reads 'FEN WDL' text files produced by 00_extract.py.
    On each __getitem__ call, returns:
        white_indices  LongTensor of active feature indices for White's king
        black_indices  LongTensor of active feature indices for Black's king
        stm            0 = White to move, 1 = Black to move
        target         float WDL from side-to-move's perspective
    """

    def __init__(self, path: str, max_positions: int = 0):
        self.lines: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)
                    if max_positions and len(self.lines) >= max_positions:
                        break
        print(f"Loaded {len(self.lines):,} positions from {path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        # Last token is the WDL label; everything before is the FEN (6 fields).
        parts = line.rsplit(" ", 1)
        fen, wdl_str = parts[0], parts[1]
        wdl = float(wdl_str)

        board = chess.Board(fen)
        wf, bf = halfkp_indices(board)
        stm   = 0 if board.turn == chess.WHITE else 1

        return (
            torch.tensor(wf, dtype=torch.long),
            torch.tensor(bf, dtype=torch.long),
            torch.tensor(stm, dtype=torch.long),
            torch.tensor(wdl, dtype=torch.float32),
        )


def collate_fn(batch):
    """Variable-length feature index lists → padded tensors + length mask."""
    wf_list, bf_list, stms, targets = zip(*batch)
    B = len(batch)

    max_w = max(len(x) for x in wf_list)
    max_b = max(len(x) for x in bf_list)

    wf_pad = torch.zeros(B, max_w, dtype=torch.long)
    bf_pad = torch.zeros(B, max_b, dtype=torch.long)
    wf_len = torch.zeros(B, dtype=torch.long)
    bf_len = torch.zeros(B, dtype=torch.long)

    for i, (wf, bf) in enumerate(zip(wf_list, bf_list)):
        n, m = len(wf), len(bf)
        wf_pad[i, :n] = wf
        bf_pad[i, :m] = bf
        wf_len[i]     = n
        bf_len[i]      = m

    return (
        wf_pad, wf_len,
        bf_pad, bf_len,
        torch.stack(stms),
        torch.stack(targets),
    )


# ── Network ──────────────────────────────────────────────────────────────────

class NNUE(nn.Module):
    """
    HalfKP NNUE: 40,960 → 256 feature transformer, then 512 → 32 → 32 → 1.

    The feature transformer is computed manually (sparse embedding sum) so that
    both king perspectives are always maintained.  During inference the two
    halves are concatenated in STM-first order.
    """

    def __init__(self):
        super().__init__()

        # Feature transformer (one shared weight matrix, used twice)
        self.ft_bias   = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.ft_weight = nn.Embedding(INPUT_FEATURES, HIDDEN_SIZE)
        nn.init.uniform_(self.ft_weight.weight, -0.1 / HIDDEN_SIZE**0.5,
                                                 0.1 / HIDDEN_SIZE**0.5)

        # Fully-connected layers
        self.l1  = nn.Linear(HIDDEN_SIZE * 2, L1_SIZE)
        self.l2  = nn.Linear(L1_SIZE, L2_SIZE)
        self.out = nn.Linear(L2_SIZE, 1)

        # Initialise FC layers with small weights
        for layer in (self.l1, self.l2, self.out):
            nn.init.uniform_(layer.weight, -0.1, 0.1)
            nn.init.zeros_(layer.bias)

    def accumulate(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Sparse sum: for each item in the batch, sum the embedding rows given by
        `indices` (padded to the longest item in the batch).

        indices: (B, max_len) LongTensor
        lengths: (B,)         LongTensor
        returns: (B, HIDDEN_SIZE) float32
        """
        B, max_len = indices.shape
        device = indices.device

        # Embed all indices at once, then zero-out padding positions.
        emb = self.ft_weight(indices)            # (B, max_len, HIDDEN_SIZE)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        emb = emb * mask.unsqueeze(-1).float()   # zero padding
        acc = emb.sum(dim=1) + self.ft_bias      # (B, HIDDEN_SIZE)
        return acc

    def forward(
        self,
        wf_pad: torch.Tensor, wf_len: torch.Tensor,
        bf_pad: torch.Tensor, bf_len: torch.Tensor,
        stm:    torch.Tensor,
    ) -> torch.Tensor:
        """
        wf_pad / wf_len : White-perspective feature indices (padded) + lengths
        bf_pad / bf_len : Black-perspective feature indices (padded) + lengths
        stm             : (B,) side to move — 0=White, 1=Black
        returns         : (B,) eval in centipawns from side-to-move's perspective
        """
        acc_w = self.accumulate(wf_pad, wf_len)   # (B, HIDDEN_SIZE) — White king
        acc_b = self.accumulate(bf_pad, bf_len)   # (B, HIDDEN_SIZE) — Black king

        # ClampedReLU ∈ [0, 1]  (will be scaled to [0, QA] during export)
        acc_w = acc_w.clamp(0.0, 1.0)
        acc_b = acc_b.clamp(0.0, 1.0)

        # STM first: concatenate [STM_half, NSTM_half]
        # stm == 0 → White to move → [acc_w, acc_b]
        # stm == 1 → Black to move → [acc_b, acc_w]
        mask = (stm == 0).float().unsqueeze(-1)   # (B, 1)
        stm_half  = acc_w * mask + acc_b * (1 - mask)
        nstm_half = acc_b * mask + acc_w * (1 - mask)
        x = torch.cat([stm_half, nstm_half], dim=-1)  # (B, 512)

        x = self.l1(x).clamp(0.0, 1.0)   # ClampedReLU — matches Rust [0, QB]
        x = self.l2(x).clamp(0.0, 1.0)   # ClampedReLU — matches Rust [0, QB]
        x = self.out(x).squeeze(-1)       # (B,)  raw centipawns

        return x

    @staticmethod
    def sigmoid(cp: torch.Tensor) -> torch.Tensor:
        """Map centipawn eval to win probability ∈ (0, 1)."""
        return torch.sigmoid(cp / CP_SCALE)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for wf_pad, wf_len, bf_pad, bf_len, stm, target in loader:
        wf_pad = wf_pad.to(device);  wf_len = wf_len.to(device)
        bf_pad = bf_pad.to(device);  bf_len = bf_len.to(device)
        stm    = stm.to(device);     target = target.to(device)

        optimizer.zero_grad()
        pred = model(wf_pad, wf_len, bf_pad, bf_len, stm)
        pred_wdl = model.sigmoid(pred)
        loss = nn.functional.mse_loss(pred_wdl, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(target)
        n += len(target)

    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for wf_pad, wf_len, bf_pad, bf_len, stm, target in loader:
        wf_pad = wf_pad.to(device);  wf_len = wf_len.to(device)
        bf_pad = bf_pad.to(device);  bf_len = bf_len.to(device)
        stm    = stm.to(device);     target = target.to(device)
        pred     = model(wf_pad, wf_len, bf_pad, bf_len, stm)
        pred_wdl = model.sigmoid(pred)
        loss = nn.functional.mse_loss(pred_wdl, target)
        total_loss += loss.item() * len(target)
        n += len(target)
    return total_loss / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="train/data/positions.txt")
    ap.add_argument("--out-dir",   default="train/model")
    ap.add_argument("--epochs",    type=int,   default=10)
    ap.add_argument("--batch",     type=int,   default=4096)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--workers",   type=int,   default=4)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--max-pos",   type=int,   default=0,
                    help="Limit dataset size (0 = unlimited)")
    ap.add_argument("--resume",    default="",
                    help="Path to a checkpoint to resume from (e.g. .../best.pt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = PositionDataset(args.data, max_positions=args.max_pos)
    n_val   = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=(device != "cpu"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NNUE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    best_val_loss = float("inf")
    best_ckpt     = out_dir / "best.pt"
    start_epoch   = 1

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}  "
              f"(val_loss={best_val_loss:.6f})")
    else:
        if args.resume:
            print(f"Warning: checkpoint '{args.resume}' not found — starting fresh")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model, val_loader, device)
        scheduler.step()

        marker = " ★" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Save every epoch — includes optimizer and scheduler state so
        # training can be resumed exactly from any epoch.
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss":  val_loss,
        }, best_ckpt if val_loss <= best_val_loss else out_dir / "last.pt")

        # Always keep last.pt up to date for resume purposes.
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss":  val_loss,
        }, out_dir / "last.pt")

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={time.time()-t0:.0f}s{marker}")

    # ── Save final checkpoint ─────────────────────────────────────────────────
    print(f"\nBest checkpoint : {best_ckpt}  (val_loss={best_val_loss:.6f})")
    print(f"\nNext step: python train/02_export.py --ckpt {best_ckpt}")


if __name__ == "__main__":
    main()
