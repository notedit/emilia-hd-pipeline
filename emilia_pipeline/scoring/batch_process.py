#!/usr/bin/env python
"""Streaming batch processor for Emilia-ZH under a small disk budget (e.g. 500 GB).

The full Emilia-ZH source is ~1.26 TB (920 tars), but the Phase-1 stage outputs
(parquet + fp16 embeddings) total only ~10 GB. So the strategy is simple:

    treat source tars as DISPOSABLE, stage outputs as KEEPERS.

For each batch of shards we: download -> run Phase-1 -> delete the source tars
and purge the HF cache copy -> keep only the tiny stage parquet/npy. Peak disk =
one batch of tars (+ their transient HF cache), never the whole dataset.

Everything is idempotent: a shard whose ``done/phase1/<shard>`` marker exists is
skipped (never re-downloaded, never re-processed), so the loop resumes cleanly
after any interruption. A disk-watermark guard refuses to start a batch that
would not fit, so a runaway download can't fill the disk.

This orchestrates the existing pieces (hf_hub_download + the Phase-1 worker CLI);
it adds no new pipeline logic. Real models run when ``runtime.use_mocks`` is
false and weights are configured.

Example:
    HF_TOKEN=hf_... HF_ENDPOINT=https://hf-mirror.com \\
    python -m emilia_pipeline.scoring.batch_process \\
        --config /data/emilia-100h/pilot_100h.yaml \\
        --repo amphion/Emilia-Dataset --prefix Emilia/ZH/ \\
        --shards ZH-B000000..ZH-B000019 \\
        --batch-size 5 --min-free-gb 60
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..common.config import Config, load_config
from ..common.io_utils import is_done

PHASE1_STAGE = "phase1"


# ---------------------------------------------------------------------------
# Shard-list expansion
# ---------------------------------------------------------------------------


def expand_shards(spec: str) -> list[str]:
    """Expand a shard spec into an ordered list of shard names.

    Accepts comma-separated names and ``A..B`` numeric ranges on a shared
    prefix, e.g. ``ZH-B000000..ZH-B000019`` or ``ZH-B000000,ZH-B000005``.
    Range endpoints must share a non-numeric prefix and have equal-width numeric
    suffixes (zero-padding preserved).
    """
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            lo, hi = part.split("..", 1)
            out.extend(_expand_range(lo.strip(), hi.strip()))
        else:
            out.append(part)
    return out


def _split_numeric_suffix(name: str) -> tuple[str, str]:
    """Split ``ZH-B000012`` -> ("ZH-B", "000012")."""
    i = len(name)
    while i > 0 and name[i - 1].isdigit():
        i -= 1
    return name[:i], name[i:]


def _expand_range(lo: str, hi: str) -> list[str]:
    plo, nlo = _split_numeric_suffix(lo)
    phi, nhi = _split_numeric_suffix(hi)
    if plo != phi or not nlo or not nhi:
        raise ValueError(f"bad range {lo}..{hi}: prefixes/suffixes mismatch")
    width = len(nlo)
    return [f"{plo}{i:0{width}d}" for i in range(int(nlo), int(nhi) + 1)]


# ---------------------------------------------------------------------------
# Disk + cache helpers
# ---------------------------------------------------------------------------


def free_gb(path: Path) -> float:
    """Return free space in GB on the filesystem holding ``path``."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def purge_hf_cache_for(repo: str) -> int:
    """Delete the HF cache blobs for a dataset repo; return bytes freed.

    hf_hub_download copies from ``~/.cache/huggingface`` into our source dir, so
    the cache holds a second copy of every downloaded tar. We purge it each
    batch to keep peak disk to a single copy.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    slug = "datasets--" + repo.replace("/", "--")
    cache_dir = Path(HF_HUB_CACHE) / slug
    if not cache_dir.exists():
        return 0
    freed = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    shutil.rmtree(cache_dir, ignore_errors=True)
    return freed


# ---------------------------------------------------------------------------
# Download one shard
# ---------------------------------------------------------------------------


def download_shard(repo: str, prefix: str, shard: str, dest_dir: Path, token: Optional[str]) -> Path:
    """Download one ``<prefix><shard>.tar`` into ``dest_dir`` (skip if present)."""
    from huggingface_hub import hf_hub_download

    dest = dest_dir / f"{shard}.tar"
    if dest.exists():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo, f"{prefix}{shard}.tar", repo_type="dataset", token=token
    )
    tmp = dest.with_suffix(".tar.tmp")
    shutil.copy(cached, tmp)
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# Run the Phase-1 worker on the currently-present source dir
# ---------------------------------------------------------------------------


def run_phase1(config_path: str, source_dir: Path, gpus: str, limit: Optional[int]) -> int:
    """Invoke the Phase-1 worker CLI over the shards currently in ``source_dir``.

    Idempotent by construction: the worker skips any shard with a done marker.
    Returns the worker's exit code.
    """
    env = dict(os.environ)
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    cmd = [
        sys.executable, "-m", "emilia_pipeline.phase1.worker",
        "--config", config_path, "--source", str(source_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return subprocess.run(cmd, env=env).returncode


# ---------------------------------------------------------------------------
# Main streaming loop
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Streaming batch Phase-1 under a disk budget")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", default="amphion/Emilia-Dataset")
    parser.add_argument("--prefix", default="Emilia/ZH/")
    parser.add_argument("--shards", required=True, help="e.g. ZH-B000000..ZH-B000019 or a,b,c")
    parser.add_argument("--batch-size", type=int, default=5, help="tars downloaded before each Phase-1 pass")
    parser.add_argument("--min-free-gb", type=float, default=60.0,
                        help="refuse a batch if free space would drop below this")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES for the worker")
    parser.add_argument("--keep-tars", action="store_true", help="do NOT delete source tars after processing")
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    source_dir = Path(config.paths.source)
    token = os.environ.get(args.token_env)
    est_tar_gb = 2.2  # conservative per-tar estimate for the watermark guard

    all_shards = expand_shards(args.shards)
    pending = [s for s in all_shards if not is_done(PHASE1_STAGE, s, config.paths.done)]
    done_already = len(all_shards) - len(pending)
    print(f"[batch] {len(all_shards)} shards total, {done_already} already done, {len(pending)} pending")
    print(f"[batch] disk free now: {free_gb(source_dir if source_dir.exists() else Path('/')):.0f} GB")

    processed = 0
    for i in range(0, len(pending), args.batch_size):
        batch = pending[i : i + args.batch_size]
        # Disk watermark guard: would this batch's downloads bust the budget?
        base = source_dir if source_dir.exists() else Path.home()
        projected = free_gb(base) - est_tar_gb * len(batch) * 2  # tar + transient cache
        if projected < args.min_free_gb:
            print(f"[batch] STOP: batch of {len(batch)} would drop free below "
                  f"{args.min_free_gb}GB (projected {projected:.0f}GB). "
                  f"Reduce --batch-size or free space.")
            return 2

        print(f"\n[batch] === downloading {len(batch)} tars: {batch[0]}..{batch[-1]} ===")
        got: list[Path] = []
        for shard in batch:
            if is_done(PHASE1_STAGE, shard, config.paths.done):
                continue
            try:
                got.append(download_shard(args.repo, args.prefix, shard, source_dir, token))
            except Exception as exc:  # noqa: BLE001 - log and continue the batch
                print(f"[batch] download FAILED {shard}: {exc}")

        print(f"[batch] === Phase-1 on {len(got)} tars ===")
        rc = run_phase1(args.config, source_dir, args.gpus, limit=None)
        if rc != 0:
            print(f"[batch] worker exited {rc}; stopping (done markers preserve progress)")
            return rc

        # Reclaim: delete processed source tars + purge the HF cache copy.
        if not args.keep_tars:
            for tar in got:
                shard = tar.stem
                if is_done(PHASE1_STAGE, shard, config.paths.done):
                    tar.unlink(missing_ok=True)
            freed = purge_hf_cache_for(args.repo)
            print(f"[batch] reclaimed source tars + {freed/1e9:.1f}GB HF cache; "
                  f"free now {free_gb(base):.0f}GB")
        processed += len(got)

    print(f"\n[batch] DONE. processed {processed} shards this run. "
          f"stage outputs kept under {config.paths.stage}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
