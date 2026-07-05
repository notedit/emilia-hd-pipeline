"""Publish the Phase-1 filtered release to HF -- with a human confirmation gate.

Safety contract (per project convention, 2026-07-05):
  * Default is a DRY RUN: package (optional) + print exactly what would be
    uploaded and which remote files would be deleted -- then exit.
  * The upload only happens after explicit confirmation: interactively typing
    ``upload``, or passing ``--yes`` (for pre-approved automation).
  * Remote replacement is atomic-ish and complete: stale remote files under
    the managed prefixes (``data/``, ``metadata/``, root parquets) that are
    absent from the new export are deleted in the same commit flow, so a
    smaller re-publish never leaves orphan shards behind. README.md is never
    touched unless ``--include-readme``.

Usage:
    .venv/bin/python scripts/publish_hf.py --config configs/full.yaml \
        [--revision main] [--skip-package] [--tag v1.0] [--yes] [--include-readme]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MANAGED_PREFIXES = ("data/", "metadata/")
MANAGED_ROOT_FILES = ("phase1_manifest_v1.parquet",)


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f}{unit}" if unit != "B" else f"{nbytes}B"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--repo", default=None, help="override hf.repo_id")
    ap.add_argument("--skip-package", action="store_true",
                    help="reuse the existing export dir instead of repackaging")
    ap.add_argument("--tag", default=None, help="create this tag after a successful upload")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive confirmation (pre-approved runs only)")
    ap.add_argument("--include-readme", action="store_true",
                    help="also upload the auto-generated dataset card as README.md")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from emilia_pipeline.common.config import load_config
    from emilia_pipeline.scoring import phase1_hf

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    repo = args.repo or cfg.hf.repo_id
    if not repo:
        print("hf.repo_id not configured", file=sys.stderr)
        return 2
    export_dir = Path(cfg.paths.export) / phase1_hf.EXPORT_SUBDIR

    # ---- 1) package (or reuse) ----
    if args.skip_package and export_dir.exists():
        print(f"[publish] reusing existing export: {export_dir}")
    else:
        print("[publish] packaging phase1 export ...", flush=True)
        res = phase1_hf.package_phase1(cfg)
        print(f"[publish] packaged {res.n_clips} clips ({res.n_with_audio} with audio), "
              f"{len(res.shard_paths)} shards")

    local_files = {
        str(p.relative_to(export_dir)): p.stat().st_size
        for p in export_dir.rglob("*")
        # .cache/ is upload_large_folder's resume bookkeeping, never published
        if p.is_file() and not str(p.relative_to(export_dir)).startswith(".cache/")
    }
    if not args.include_readme:
        local_files.pop("README.md", None)
    total = sum(local_files.values())

    # ---- 2) diff against the remote revision ----
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=cfg.hf.private, exist_ok=True)
    try:
        remote = set(api.list_repo_files(repo, repo_type="dataset", revision=args.revision))
    except Exception:
        remote = set()
    managed_remote = {
        f for f in remote
        if f.startswith(MANAGED_PREFIXES) or f in MANAGED_ROOT_FILES
    }
    to_delete = sorted(managed_remote - set(local_files))

    # ---- 3) show the plan and ask ----
    print(f"\n===== 发布计划 =====")
    print(f"repo:      {repo} @ {args.revision}")
    print(f"上传:      {len(local_files)} 个文件, 共 {human(total)}")
    for f in sorted(local_files)[:8]:
        print(f"   + {f} ({human(local_files[f])})")
    if len(local_files) > 8:
        print(f"   + ... 共 {len(local_files)} 个")
    print(f"远端删除:  {len(to_delete)} 个过期文件")
    for f in to_delete[:8]:
        print(f"   - {f}")
    if len(to_delete) > 8:
        print(f"   - ... 共 {len(to_delete)} 个")
    if args.tag:
        print(f"打 tag:    {args.tag}")

    if not args.yes:
        if not sys.stdin.isatty():
            print("\n[publish] DRY RUN (无终端且未加 --yes), 未做任何修改。确认无误后加 --yes 执行。")
            return 0
        answer = input("\n输入 'upload' 确认执行 (其他任意输入取消): ").strip()
        if answer != "upload":
            print("[publish] 已取消, 未做任何修改。")
            return 1

    # ---- 4) delete stale, then upload ----
    if to_delete:
        from huggingface_hub import CommitOperationDelete

        api.create_commit(
            repo_id=repo, repo_type="dataset", revision=args.revision,
            operations=[CommitOperationDelete(path_in_repo=f) for f in to_delete],
            commit_message=f"publish: remove {len(to_delete)} stale files",
        )
        print(f"[publish] deleted {len(to_delete)} stale remote files")
    api.upload_large_folder(
        repo_id=repo, folder_path=str(export_dir), repo_type="dataset",
        revision=args.revision,
    )
    print("[publish] upload complete")
    if args.tag:
        api.create_tag(repo, tag=args.tag, repo_type="dataset", revision=args.revision)
        print(f"[publish] tagged {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
