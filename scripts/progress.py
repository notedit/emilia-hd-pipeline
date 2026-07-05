"""One-line Phase-1 progress snapshot: done/pending/failed counts, rate, ETA.

Usage: progress.py --config configs/full.yaml [--expected 920]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--expected", type=int, default=None,
                    help="expected total shard count (default: tars present)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from emilia_pipeline.common.config import load_config

    cfg = load_config(args.config)
    source = Path(cfg.paths.source)
    root = source.parent
    n_src = len(list(source.glob("*.tar")))
    done_dir = Path(cfg.paths.done) / "phase1"
    markers = sorted(done_dir.glob("*"), key=lambda p: p.stat().st_mtime) if done_dir.exists() else []
    n_done = len(markers)
    n_quar = len(list((root / "quarantine").glob("*.tar"))) if (root / "quarantine").exists() else 0
    dl_done = (root / "DOWNLOAD_DONE").exists()
    total = args.expected or n_src

    # throughput over the last hour of done markers
    now = time.time()
    recent = [m for m in markers if now - m.stat().st_mtime < 3600]
    rate_h = len(recent)
    remaining = max(0, total - n_done - n_quar)
    eta = f"{remaining / rate_h:.1f}h" if rate_h else "n/a"

    print(
        f"[progress] downloaded={n_src}{'(+DONE)' if dl_done else ''} "
        f"done={n_done}/{total} quarantined={n_quar} "
        f"rate={rate_h}/h eta={eta}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
