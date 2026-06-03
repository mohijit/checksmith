"""
Blend Checksmith engine evaluations with game-result WDL labels.

Reads  : --input  (positions.txt from 00_extract.py)
           format: <FEN> <game_wdl>
Writes : --out    (scored.txt — use as --data input to 01_train.py)
           format: <FEN> <blended_wdl>

Blend formula:
    target = (1 − λ) × sigmoid(engine_cp / 400) + λ × game_wdl

where:
    λ (--lam)         weight given to the game result  [0 = pure eval, 1 = pure game]
    engine_cp         centipawns from side-to-move perspective at --depth

Why this helps:
    Game results are noisy labels (1800-ELO players blunder). Engine evaluations
    give a direct quality signal for each position, dramatically reducing label noise
    and improving training convergence.

Usage (on Kaggle / Linux after `cargo build --release`):
    python train/03_score.py \\
        --engine  /kaggle/working/target/release/checksmith \\
        --input   /kaggle/working/positions.txt \\
        --out     /kaggle/working/scored.txt \\
        --depth   5 \\
        --lam     0.5 \\
        --workers 4

Usage (Windows, from repo root):
    python train\\03_score.py ^
        --engine  target\\release\\checksmith.exe ^
        --input   train\\data\\positions.txt ^
        --out     train\\data\\scored.txt ^
        --depth   5 --lam 0.5 --workers 2

Resume:
    Re-run the same command after an interruption.  The script counts lines
    already in --out and skips that many lines from --input automatically.
"""

import argparse
import heapq
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

CP_SCALE = 400.0   # must match CP_SCALE in 01_train.py


# ── Math helpers ──────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def blend_wdl(eval_cp: int | None, game_wdl: float, lam: float) -> float:
    """
    Blend engine eval with game result.  Falls back to game_wdl when the
    engine returns no score (e.g. game-over position or subprocess error).
    """
    if eval_cp is None:
        return game_wdl
    eval_wdl = sigmoid(eval_cp / CP_SCALE)
    return (1.0 - lam) * eval_wdl + lam * game_wdl


# ── Engine subprocess wrapper ─────────────────────────────────────────────────

class EngineWorker:
    """
    Wraps a single Checksmith UCI process, kept alive for the session.
    One instance per worker thread.
    """

    MATE_CP = 30_000   # centipawn value used to represent forced mate

    def __init__(self, engine_path: str, hash_mb: int = 8):
        self._proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name OwnBook value false")
        self._send("isready")
        self._wait("readyok")

    def _send(self, cmd: str) -> None:
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait(self, token: str) -> str:
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("engine stdout closed unexpectedly")
            if token in line:
                return line.rstrip()

    def eval_fen(self, fen: str, depth: int) -> int | None:
        """
        Return centipawns from side-to-move's perspective at `depth`.
        Returns None if no score was reported (e.g. terminal position).
        Mate scores are mapped to ±MATE_CP.
        """
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        last_cp: int | None = None
        while True:
            line = self._proc.stdout.readline().rstrip()
            if not line:
                continue
            if line.startswith("bestmove"):
                break
            if "score cp" in line:
                try:
                    after = line.split("score cp")[1].split()
                    last_cp = int(after[0])
                except (IndexError, ValueError):
                    pass
            elif "score mate" in line:
                try:
                    after = line.split("score mate")[1].split()
                    m = int(after[0])
                    last_cp = self.MATE_CP if m > 0 else -self.MATE_CP
                except (IndexError, ValueError):
                    pass
        return last_cp

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()


# ── Worker thread ─────────────────────────────────────────────────────────────

def _worker(
    engine_path: str,
    depth: int,
    lam: float,
    work_q: "queue.Queue[tuple[int, str, float] | None]",
    result_q: "queue.Queue[tuple[int, str]]",
) -> None:
    engine = EngineWorker(engine_path)
    while True:
        item = work_q.get()
        if item is None:                     # shutdown sentinel
            work_q.task_done()
            break
        idx, fen, game_wdl = item
        try:
            cp     = engine.eval_fen(fen, depth)
            target = blend_wdl(cp, game_wdl, lam)
        except Exception:
            target = game_wdl               # fall back on any subprocess error
        result_q.put((idx, f"{fen} {target:.6f}"))
        work_q.task_done()
    engine.close()


# ── Utilities ─────────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Blend Checksmith engine evals with game-result WDL labels."
    )
    ap.add_argument("--engine",  required=True,
                    help="Path to compiled Checksmith binary")
    ap.add_argument("--input",   default="train/data/positions.txt",
                    help="Input file (FEN<SP>game_wdl)")
    ap.add_argument("--out",     default="train/data/scored.txt",
                    help="Output file (FEN<SP>blended_wdl)")
    ap.add_argument("--depth",   type=int,   default=5,
                    help="Search depth per position (default 5, ~1–5 ms each)")
    ap.add_argument("--lam",     type=float, default=0.5,
                    help="Lambda: weight for game result vs engine eval "
                         "(0=pure eval, 1=pure game result, default 0.5)")
    ap.add_argument("--workers", type=int,   default=2,
                    help="Parallel engine instances (default 2)")
    ap.add_argument("--max-pos", type=int,   default=0,
                    help="Cap positions to score (0 = all, default 0)")
    args = ap.parse_args()

    if not Path(args.engine).exists():
        sys.exit(f"Engine binary not found: {args.engine}\n"
                 f"Build with: cargo build --release")
    if not Path(args.input).exists():
        sys.exit(f"Input file not found: {args.input}")

    out_path   = Path(args.out)
    in_path    = Path(args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: lines already written become the skip count.
    resume_at = _count_lines(out_path)
    if resume_at:
        print(f"Resuming — {resume_at:,} positions already scored, skipping.")

    print("Counting input positions …")
    total_in = _count_lines(in_path)
    cap      = (resume_at + args.max_pos) if args.max_pos else total_in
    to_score = max(0, min(cap, total_in) - resume_at)
    print(f"Input total : {total_in:,}")
    print(f"Already done: {resume_at:,}")
    print(f"To score    : {to_score:,}")
    print(f"Depth       : {args.depth}   lambda={args.lam}   workers={args.workers}")

    if to_score == 0:
        print("Nothing to do.")
        return

    work_q:   "queue.Queue" = queue.Queue(maxsize=args.workers * 16)
    result_q: "queue.Queue" = queue.Queue()

    # Start worker threads, one engine subprocess each.
    threads = []
    for _ in range(args.workers):
        t = threading.Thread(
            target=_worker,
            args=(args.engine, args.depth, args.lam, work_q, result_q),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Writer thread: buffers out-of-order results and writes them in order.
    written   = 0
    t0        = time.time()
    write_err = [None]

    def writer_body() -> None:
        nonlocal written
        heap: list = []           # min-heap of (global_idx, line)
        next_idx   = resume_at    # the index we want to write next

        try:
            with open(out_path, "a") as fout:
                while written < to_score:
                    try:
                        idx, line = result_q.get(timeout=5.0)
                    except queue.Empty:
                        continue
                    heapq.heappush(heap, (idx, line))
                    while heap and heap[0][0] == next_idx:
                        _, out_line = heapq.heappop(heap)
                        fout.write(out_line + "\n")
                        next_idx += 1
                        written  += 1
                        if written % 10_000 == 0:
                            elapsed = time.time() - t0
                            rate    = written / elapsed if elapsed > 0 else 1.0
                            eta_h   = (to_score - written) / rate / 3600
                            print(f"  done={resume_at + written:,}  "
                                  f"rate={rate:.0f}/s  eta={eta_h:.1f}h",
                                  flush=True)
                        if written % 50_000 == 0:
                            fout.flush()
        except Exception as exc:
            write_err[0] = exc

    writer_t = threading.Thread(target=writer_body, daemon=True)
    writer_t.start()

    # Feed positions to workers, skipping already-done lines.
    fed = 0
    with open(in_path) as fin:
        for global_idx, raw in enumerate(fin):
            if global_idx < resume_at:
                continue
            if global_idx >= resume_at + to_score:
                break
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            fen, wdl_str = parts
            try:
                game_wdl = float(wdl_str)
            except ValueError:
                continue
            work_q.put((global_idx, fen, game_wdl))
            fed += 1

    # Signal all workers to stop, then wait.
    for _ in threads:
        work_q.put(None)
    work_q.join()

    # Wait for writer to drain.
    writer_t.join(timeout=60)

    if write_err[0]:
        sys.exit(f"Writer error: {write_err[0]}")

    elapsed = time.time() - t0
    print(f"\nDone. Scored {written:,} positions in {elapsed:.0f}s "
          f"({written/elapsed:.0f}/s)")
    print(f"Output: {out_path}")
    print(f"\nNext step:")
    print(f"  python train/01_train.py --data {out_path} --epochs 15")


if __name__ == "__main__":
    main()
