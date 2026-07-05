"""Poison-shard guard for the self-healing Phase-1 loops (run between passes).

A shard that keeps crashing the worker would otherwise be retried forever
(pending == all - done). After each worker pass, run_full.sh calls this guard:
every failure record under ``failed/phase1`` bumps that shard's attempt
counter; at >= ``--max-attempts`` the source tar is moved to ``quarantine/``
so the next pass no longer sees it. Prints the remaining pending count for
this worker's partition (the loop's exit signal).

Usage:
    phase1_guard.py --config CFG --num-workers N --worker-index I
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from emilia_pipeline.common.config import load_config
    from emilia_pipeline.common.io_utils import enumerate_shard_tasks, pending_tasks

    cfg = load_config(args.config)
    source = Path(cfg.paths.source)
    root = source.parent
    attempts_dir = root / "logs" / "attempts"
    quarantine = root / "quarantine"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    failed_dir = Path(cfg.paths.failed) / "phase1"
    done_dir = Path(cfg.paths.done)
    for rec in sorted(failed_dir.glob("*.json")) if failed_dir.exists() else []:
        shard = rec.stem
        if (done_dir / "phase1" / shard).exists():
            rec.unlink(missing_ok=True)  # succeeded on a later retry
            continue
        counter = attempts_dir / shard
        n = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(n))
        rec.unlink(missing_ok=True)  # consumed; a fresh failure re-creates it
        if n >= args.max_attempts and (source / f"{shard}.tar").exists():
            quarantine.mkdir(parents=True, exist_ok=True)
            (source / f"{shard}.tar").rename(quarantine / f"{shard}.tar")
            print(f"[guard] QUARANTINED {shard} after {n} failed attempts", flush=True)

    tasks = enumerate_shard_tasks(source)
    pending = pending_tasks(tasks, "phase1", cfg.paths.done)
    if args.num_workers > 1:
        pending = [
            s for s in pending
            if zlib.crc32(s.encode()) % args.num_workers == args.worker_index
        ]
    print(len(pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
