"""HuggingFace packaging and upload (design §2 收尾 / §6.4 tail).

Takes the S5-published subset (per-clip meta JSON + tiered selection) and emits
the release artifacts under ``export/``: shuffled WebDataset tar shards pairing
each clip's audio with its ``{clip_id}.json`` meta, plus a flat parquet manifest
of what went into each shard. When ``HF_TOKEN`` is present it uploads the folder
via ``huggingface_hub.upload_large_folder``; when absent it produces the folder
and skips the upload (project convention: network behind a graceful guard).

Audio bytes come from the Phase-2 ``repacked/`` WebDataset shards, resolved via
``repacked/repack_index.parquet`` (``clip_id -> (shard, member)``). When the
repacked audio is unavailable (e.g. unit tests, meta-only dry runs) packaging
still succeeds: shards are written with meta-only members and the manifest marks
``has_audio=False`` so the gap is explicit rather than silent.
"""

from __future__ import annotations

import io
import json
import os
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..common.config import Config
from ..common.io_utils import (
    atomic_write_parquet,
    parquet_glob,
    query_parquet,
    write_done_marker,
)

# HF token env var (skip upload gracefully when absent).
HF_TOKEN_ENV = "HF_TOKEN"
# Done-marker task id for the packaging step.
PACK_TASK_ID = "all"
# Default target uncompressed bytes per output tar shard (~1GB, design §4).
DEFAULT_SHARD_BYTES = 1_000_000_000


# ---------------------------------------------------------------------------
# Carriers
# ---------------------------------------------------------------------------


@dataclass
class PackEntry:
    """One clip destined for the release.

    Attributes:
        clip_id: Clip identifier.
        tier: Published tier string (``S``/``A``/``B``).
        meta_json: The published §7 meta dict (written as ``{clip_id}.json``).
        audio_bytes: Encoded audio (flac) if resolvable, else None.
        audio_ext: File extension for the audio member (default ``flac``).
    """

    clip_id: str
    tier: str
    meta_json: dict[str, Any]
    audio_bytes: Optional[bytes] = None
    audio_ext: str = "flac"

    @property
    def has_audio(self) -> bool:
        return self.audio_bytes is not None


@dataclass
class PackResult:
    """Outcome of a packaging (and optional upload) run."""

    export_dir: Path
    shard_paths: list[Path] = field(default_factory=list)
    manifest_path: Optional[Path] = None
    n_clips: int = 0
    n_with_audio: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    uploaded: bool = False
    upload_skipped_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Loading the published subset
# ---------------------------------------------------------------------------


def load_published_entries(
    config: Config, *, tiers: Optional[Iterable[str]] = None
) -> list[PackEntry]:
    """Load published clips from the S5 flat parquet + per-clip JSON.

    Reads the S5 flat meta parquet (``export/meta/part-*.parquet``) to find the
    tiered clips, then loads each ``export/meta_json/{clip_id}.json`` as the
    authoritative published record.

    Args:
        config: Pipeline config.
        tiers: Optional whitelist of tiers to include (default: all tiered).

    Returns:
        A list of :class:`PackEntry` without audio bytes yet (see
        :func:`attach_audio`). Empty if no S5 outputs exist.
    """
    from .s5_score import s5_flat_parquet_path, s5_json_dir

    flat_glob = parquet_glob(s5_flat_parquet_path(config).parent)
    if not _glob_has_files(flat_glob):
        return []

    tier_filter = set(tiers) if tiers is not None else None
    result = query_parquet(
        "SELECT clip_id, selection_tier FROM m WHERE selection_tier IS NOT NULL",
        m=flat_glob,
    )
    json_dir = s5_json_dir(config)
    entries: list[PackEntry] = []
    for clip_id, tier in result.fetchall():
        if tier_filter is not None and str(tier) not in tier_filter:
            continue
        jpath = json_dir / f"{clip_id}.json"
        if not jpath.exists():
            continue
        meta = json.loads(jpath.read_text(encoding="utf-8"))
        entries.append(PackEntry(clip_id=str(clip_id), tier=str(tier), meta_json=meta))
    return entries


# ---------------------------------------------------------------------------
# Audio resolution from the repacked shards
# ---------------------------------------------------------------------------


def load_repack_index(config: Config) -> dict[str, tuple[str, str]]:
    """Load ``repacked/repack_index.parquet`` as ``clip_id -> (shard, member)``.

    Returns an empty dict when the index is absent (audio then unresolved).
    The index is expected to carry columns ``clip_id`` and ``shard``; a
    ``member`` column is used verbatim when present, else the member name is
    derived as ``{clip_id}.flac``.
    """
    index_path = Path(config.paths.repacked) / "repack_index.parquet"
    if not index_path.exists():
        return {}
    result = query_parquet(
        "SELECT * FROM idx",
        idx=str(index_path),
    )
    cols = [d[0] for d in result.description]
    mapping: dict[str, tuple[str, str]] = {}
    for row in result.fetchall():
        rec = dict(zip(cols, row))
        clip_id = str(rec["clip_id"])
        shard = str(rec.get("shard", ""))
        member = str(rec.get("member") or f"{clip_id}.flac")
        mapping[clip_id] = (shard, member)
    return mapping


def attach_audio(entries: Sequence[PackEntry], config: Config) -> int:
    """Populate ``audio_bytes`` on entries from the repacked tars, in place.

    Groups clips by source shard so each tar is opened once and read
    sequentially. Missing shards / members leave ``audio_bytes=None`` (the entry
    is still packaged, meta-only).

    Args:
        entries: The entries to enrich.
        config: Pipeline config (supplies ``paths.repacked``).

    Returns:
        The number of entries for which audio was successfully attached.
    """
    index = load_repack_index(config)
    if not index:
        return 0
    by_shard: dict[str, list[PackEntry]] = {}
    for e in entries:
        loc = index.get(e.clip_id)
        if loc is None:
            continue
        by_shard.setdefault(loc[0], []).append(e)

    attached = 0
    repacked = Path(config.paths.repacked)
    for shard, shard_entries in by_shard.items():
        tar_path = repacked / (shard if shard.endswith(".tar") else f"{shard}.tar")
        if not tar_path.exists():
            continue
        wanted = {index[e.clip_id][1]: e for e in shard_entries}
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                e = wanted.get(member.name)
                if e is None:
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                e.audio_bytes = fh.read()
                e.audio_ext = member.name.rsplit(".", 1)[-1] if "." in member.name else "flac"
                attached += 1
    return attached


# ---------------------------------------------------------------------------
# Shuffle + shard writing
# ---------------------------------------------------------------------------


def shuffle_entries(entries: Sequence[PackEntry], *, seed: int = 1234) -> list[PackEntry]:
    """Return a deterministically shuffled copy of ``entries``.

    Shuffling decorrelates shard order from speaker / tier so streaming
    consumers see a well-mixed distribution (design §2 tail).
    """
    out = list(entries)
    random.Random(seed).shuffle(out)
    return out


def _entry_nbytes(entry: PackEntry) -> int:
    """Approximate on-disk size of an entry (audio + meta JSON)."""
    meta_size = len(json.dumps(entry.meta_json, ensure_ascii=False).encode("utf-8"))
    return meta_size + (len(entry.audio_bytes) if entry.audio_bytes else 0)


def write_shards(
    entries: Sequence[PackEntry],
    export_dir: Path,
    *,
    target_shard_bytes: int = DEFAULT_SHARD_BYTES,
    shard_prefix: str = "shard",
) -> list[Path]:
    """Write entries to WebDataset tar shards under ``export_dir/data``.

    Each clip contributes ``{clip_id}.{ext}`` (audio, when present) and
    ``{clip_id}.json`` (meta). Shards roll over when the accumulated size
    exceeds ``target_shard_bytes``. Written atomically via ``*.tmp`` + rename.

    Args:
        entries: The (already shuffled) entries to pack.
        export_dir: Root export directory.
        target_shard_bytes: Soft size cap per shard.
        shard_prefix: Tar file name prefix.

    Returns:
        The list of shard paths written, in order.
    """
    data_dir = export_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    shard_idx = 0
    cur: Optional[tarfile.TarFile] = None
    cur_tmp: Optional[Path] = None
    cur_final: Optional[Path] = None
    cur_bytes = 0

    def _open() -> None:
        nonlocal cur, cur_tmp, cur_final, cur_bytes
        final = data_dir / f"{shard_prefix}-{shard_idx:05d}.tar"
        tmp = final.with_name(final.name + ".tmp")
        cur = tarfile.open(tmp, "w")
        cur_tmp, cur_final, cur_bytes = tmp, final, 0

    def _close() -> None:
        nonlocal cur, cur_tmp, cur_final
        if cur is None:
            return
        cur.close()
        os.replace(cur_tmp, cur_final)
        shard_paths.append(cur_final)
        cur = None

    _open()
    for entry in entries:
        if cur is None:
            _open()
        if entry.audio_bytes is not None:
            _add_member(cur, f"{entry.clip_id}.{entry.audio_ext}", entry.audio_bytes)
        meta_bytes = json.dumps(entry.meta_json, ensure_ascii=False).encode("utf-8")
        _add_member(cur, f"{entry.clip_id}.json", meta_bytes)
        cur_bytes += _entry_nbytes(entry)
        if cur_bytes >= target_shard_bytes:
            _close()
            shard_idx += 1
            _open()
    _close()
    return shard_paths


def _add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Append an in-memory bytes member to an open tar."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest_rows(
    entries: Sequence[PackEntry], shard_assignment: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Build packing-manifest rows (one per clip)."""
    rows: list[dict[str, Any]] = []
    for e in entries:
        rows.append(
            {
                "clip_id": e.clip_id,
                "tier": e.tier,
                "shard": shard_assignment.get(e.clip_id, ""),
                "has_audio": e.has_audio,
                "audio_ext": e.audio_ext,
                "original_speaker": str(
                    e.meta_json.get("source", {}).get("original_speaker", "")
                ),
                "emotion_primary": str(
                    (e.meta_json.get("omni_labels") or {})
                    .get("emotion", {})
                    .get("primary", "")
                ),
                "selection_score": float(
                    (e.meta_json.get("selection") or {}).get("selection_score", 0.0)
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def package_export(
    config: Config,
    *,
    tiers: Optional[Iterable[str]] = None,
    seed: int = 1234,
    target_shard_bytes: Optional[int] = None,
    entries: Optional[Sequence[PackEntry]] = None,
    mark_done: bool = True,
) -> PackResult:
    """Produce the ``export/`` release folder (shards + manifest); no upload.

    Args:
        config: Pipeline config.
        tiers: Optional tier whitelist.
        seed: Shuffle seed (determinism).
        target_shard_bytes: Override the per-shard size cap.
        entries: Pre-built entries (skips loading from disk; used by tests).
        mark_done: Write the ``pack`` done marker after the manifest lands.

    Returns:
        A populated :class:`PackResult`.
    """
    export_dir = Path(config.paths.export)
    if entries is None:
        entries = load_published_entries(config, tiers=tiers)
        attach_audio(entries, config)
    entries = list(entries)

    shuffled = shuffle_entries(entries, seed=seed)
    cap = target_shard_bytes or config.repack.target_shard_bytes or DEFAULT_SHARD_BYTES
    shard_paths = write_shards(shuffled, export_dir, target_shard_bytes=cap)

    # Recompute shard assignment by replaying the same size-based rollover.
    shard_assignment = _replay_shard_assignment(shuffled, cap, shard_paths)
    manifest_rows = build_manifest_rows(shuffled, shard_assignment)
    manifest_path = export_dir / "packing_manifest_v1.parquet"
    atomic_write_parquet(manifest_rows or _empty_manifest_schema(), manifest_path)

    tier_counts: dict[str, int] = {}
    for e in shuffled:
        tier_counts[e.tier] = tier_counts.get(e.tier, 0) + 1

    result = PackResult(
        export_dir=export_dir,
        shard_paths=shard_paths,
        manifest_path=manifest_path,
        n_clips=len(shuffled),
        n_with_audio=sum(1 for e in shuffled if e.has_audio),
        tier_counts=tier_counts,
    )
    if mark_done:
        write_done_marker("pack", PACK_TASK_ID, config.paths.done)
    return result


def _replay_shard_assignment(
    entries: Sequence[PackEntry], cap: int, shard_paths: Sequence[Path]
) -> dict[str, str]:
    """Recompute which shard each clip landed in (mirrors write_shards rollover)."""
    assignment: dict[str, str] = {}
    if not shard_paths:
        return assignment
    idx = 0
    cur_bytes = 0
    names = [p.name for p in shard_paths]
    for e in entries:
        if idx >= len(names):
            idx = len(names) - 1
        assignment[e.clip_id] = names[idx]
        cur_bytes += _entry_nbytes(e)
        if cur_bytes >= cap:
            idx += 1
            cur_bytes = 0
    return assignment


def _empty_manifest_schema() -> list[dict[str, Any]]:
    """A single placeholder row so the empty manifest has a defined schema."""
    return [
        {
            "clip_id": "__none__",
            "tier": "",
            "shard": "",
            "has_audio": False,
            "audio_ext": "",
            "original_speaker": "",
            "emotion_primary": "",
            "selection_score": 0.0,
        }
    ]


def upload_to_hf(
    config: Config,
    repo_id: str,
    *,
    repo_type: str = "dataset",
    private: bool = True,
    token: Optional[str] = None,
) -> PackResult:
    """Package then upload the export folder to the Hub (skips if no token).

    Uses ``huggingface_hub.upload_large_folder`` (resumable, chunked). When no
    token is available (neither the ``token`` arg nor ``HF_TOKEN`` in env) the
    folder is still produced and the upload is skipped with a recorded reason,
    so a key-less CI run is a clean no-op.

    Args:
        config: Pipeline config.
        repo_id: Target Hub repo, e.g. ``"org/emilia-expressive-zh"``.
        repo_type: Hub repo type (default ``dataset``).
        private: Create/keep the repo private.
        token: Explicit token; falls back to ``HF_TOKEN`` env.

    Returns:
        The :class:`PackResult`, with ``uploaded`` / ``upload_skipped_reason`` set.
    """
    result = package_export(config)
    tok = token or os.environ.get(HF_TOKEN_ENV)
    if not tok:
        result.upload_skipped_reason = f"{HF_TOKEN_ENV} not set; produced export only"
        return result
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(result.export_dir),
            repo_type=repo_type,
        )
        result.uploaded = True
    except Exception as exc:  # pragma: no cover - network path
        result.upload_skipped_reason = f"upload failed: {exc}"
    return result


def _glob_has_files(glob: str) -> bool:
    """Return whether a parquet glob resolves to at least one file."""
    parent = Path(glob).parent
    pattern = Path(glob).name
    return parent.is_dir() and any(parent.glob(pattern))


__all__ = [
    "HF_TOKEN_ENV",
    "PACK_TASK_ID",
    "DEFAULT_SHARD_BYTES",
    "PackEntry",
    "PackResult",
    "load_published_entries",
    "load_repack_index",
    "attach_audio",
    "shuffle_entries",
    "write_shards",
    "build_manifest_rows",
    "package_export",
    "upload_to_hf",
]
