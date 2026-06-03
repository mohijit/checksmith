"""
Extract training positions from the angeluriot/Chess_games Hugging Face dataset.

Source: https://github.com/angeluriot/Chess_games
Format: Parquet files (14.2M games, moves in UCI notation)

Output: train/data/positions.txt — one position per line:
    <FEN> <WDL>
where WDL is 1.0 (White wins), 0.5 (draw), 0.0 (Black wins).

Usage:
    pip install datasets chess numpy
    python train/00_extract.py [--max-games 5000000] [--min-elo 1800]
"""

import argparse
import os
import sys
import random
from pathlib import Path

# ── Requirements check ────────────────────────────────────────────────────────

def require(pkg, install):
    try:
        __import__(pkg)
    except ImportError:
        sys.exit(f"Missing package: pip install {install}")

require("datasets", "datasets")
require("chess", "chess")

import chess
import chess.pgn
from datasets import load_dataset

# ── Constants ─────────────────────────────────────────────────────────────────

OPENING_PLIES    = 8    # skip the first N half-moves (opening book theory)
MAX_PLIES        = 200  # ignore games longer than this (adjudicated draws)
MAX_POS_PER_GAME = 8    # sample at most this many positions per game
FILTER_IN_CHECK  = True # skip positions where the side-to-move is in check

# Map the dataset's "winner" column to White's WDL score.
WINNER_TO_WDL = {
    "white": 1.0,
    "White": 1.0,
    "w":     1.0,
    "draw":  0.5,
    "Draw":  0.5,
    "d":     0.5,
    "":      0.5,  # no result / timeout
    "black": 0.0,
    "Black": 0.0,
    "b":     0.0,
}

# ── Feature extraction helpers ─────────────────────────────────────────────────

def get_wdl(row) -> float | None:
    winner = str(row.get("winner", row.get("Winner", ""))).strip().lower()
    return WINNER_TO_WDL.get(winner, None)


def elo_ok(row, min_elo: int) -> bool:
    try:
        w = int(row.get("white_elo", row.get("WhiteElo", 0)) or 0)
        b = int(row.get("black_elo", row.get("BlackElo", 0)) or 0)
        return w >= min_elo and b >= min_elo
    except (ValueError, TypeError):
        return True  # don't discard if ELO is missing


def uci_moves(row) -> list[str]:
    """Extract the UCI move list from the row (handles multiple column names)."""
    for col in ("uci_moves", "UCI", "moves_uci", "Moves"):
        val = row.get(col)
        if val:
            if isinstance(val, str):
                return val.split()
            if isinstance(val, list):
                return [str(m) for m in val]
    # Fall back: try to read from any move-like column
    for col in row.keys():
        if "uci" in col.lower() or "move" in col.lower():
            val = row[col]
            if val:
                if isinstance(val, str):
                    return val.split()
                if isinstance(val, list) and val:
                    return [str(m) for m in val]
    return []


def extract_positions(row, min_elo: int) -> list[tuple[str, float]]:
    """Replay one game and return (FEN, WDL) pairs for sampled positions."""
    wdl = get_wdl(row)
    if wdl is None:
        return []
    if not elo_ok(row, min_elo):
        return []

    moves = uci_moves(row)
    if not moves or len(moves) < OPENING_PLIES + 1:
        return []
    if len(moves) > MAX_PLIES:
        return []

    board = chess.Board()
    positions = []

    for ply, uci in enumerate(moves):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            break  # malformed move — skip rest of game

        if not board.is_legal(mv):
            break

        # Record positions from ply OPENING_PLIES onward.
        if ply >= OPENING_PLIES:
            if FILTER_IN_CHECK and board.is_check():
                board.push(mv)
                continue
            if board.is_game_over():
                break
            # Flip WDL to side-to-move perspective for consistent labeling.
            stm_wdl = wdl if board.turn == chess.WHITE else 1.0 - wdl
            positions.append((board.fen(), stm_wdl))

        board.push(mv)

    # Sample at most MAX_POS_PER_GAME positions.
    if len(positions) > MAX_POS_PER_GAME:
        positions = random.sample(positions, MAX_POS_PER_GAME)

    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=5_000_000,
                    help="Maximum number of games to process (default: 5M)")
    ap.add_argument("--min-elo", type=int, default=1800,
                    help="Minimum ELO rating for both players (default: 1800)")
    ap.add_argument("--out", default="train/data/positions.txt",
                    help="Output file path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from angeluriot/Chess_games …")
    # Streaming avoids downloading the full 7 GB up front.
    ds = load_dataset("angeluriot/Chess_games", split="train", streaming=True)

    games_seen = 0
    positions_written = 0

    with open(out_path, "w") as fout:
        for row in ds:
            if games_seen >= args.max_games:
                break

            pairs = extract_positions(row, args.min_elo)
            for fen, wdl in pairs:
                fout.write(f"{fen} {wdl:.1f}\n")
                positions_written += 1

            games_seen += 1
            if games_seen % 100_000 == 0:
                print(f"  games={games_seen:,}  positions={positions_written:,}")

    print(f"\nDone. {games_seen:,} games → {positions_written:,} positions")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
