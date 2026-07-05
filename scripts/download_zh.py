"""Full Emilia-ZH downloader: every ZH tar -> paths.source, resumable.

Enumerates ``Emilia/ZH/*.tar`` in the source repo and downloads them with a
thread pool straight into the config's source dir. Files land via hf's
``local_dir`` direct-write (incomplete parts live under ``<staging>/.cache``)
and are atomically renamed into place, so:

  * a present ``*.tar`` in source == complete -> skipped on re-run (resume);
  * killing/re-running the script at any point is safe;
  * a ``DOWNLOAD_DONE`` marker is written next to source when everything is
    in place -- the run_full.sh worker loops use it as their exit condition.

Usage:
    .venv/bin/python scripts/download_zh.py --config configs/full.yaml \
        [--concurrency 8] [--limit N]   # HF_TOKEN must be set
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "amphion/Emilia-Dataset"
PREFIX = "Emilia/ZH/"


def list_zh_tars(token: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith(PREFIX) and f.endswith(".tar"))


def fetch_one(rel: str, source: Path, staging: Path, token: str) -> tuple[str, str]:
    """Download one repo file into ``source`` (atomic); returns (name, status)."""
    from huggingface_hub import hf_hub_download

    name = Path(rel).name
    dest = source / name
    if dest.exists():
        return name, "exists"
    got = hf_hub_download(
        REPO, rel, repo_type="dataset", token=token, local_dir=str(staging)
    )
    os.replace(got, dest)  # same filesystem: atomic
    return name, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from emilia_pipeline.common.config import load_config

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    source = Path(cfg.paths.source)
    source.mkdir(parents=True, exist_ok=True)
    staging = source.parent / "download_staging"
    staging.mkdir(parents=True, exist_ok=True)

    tars = list_zh_tars(token)
    if args.limit:
        tars = tars[: args.limit]
    todo = [t for t in tars if not (source / Path(t).name).exists()]
    print(f"[download] {len(tars)} ZH tars total, {len(todo)} to fetch", flush=True)

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(fetch_one, t, source, staging, token): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rel = futs[fut]
            try:
                name, status = fut.result()
                n_ok += 1
                if i % 10 == 0 or i == len(todo):
                    print(f"[download] {i}/{len(todo)} latest={name} ({status})", flush=True)
            except Exception as exc:  # noqa: BLE001 - log and keep going
                n_err += 1
                print(f"[download] ERROR {rel}: {type(exc).__name__}: {exc}", flush=True)

    missing = [t for t in tars if not (source / Path(t).name).exists()]
    if not missing and args.limit is None:
        (source.parent / "DOWNLOAD_DONE").touch()
        shutil.rmtree(staging, ignore_errors=True)
        print("[download] COMPLETE - marker written", flush=True)
    else:
        print(f"[download] finished pass: ok={n_ok} err={n_err} missing={len(missing)}"
              " (re-run to resume)", flush=True)
    return 0 if not missing or args.limit else (1 if n_err else 0)


if __name__ == "__main__":
    raise SystemExit(main())
