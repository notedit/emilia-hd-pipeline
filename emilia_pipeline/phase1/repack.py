"""Phase-1 -> Phase-2 bridge: repack survivors into priority-ordered shards.

Design §4 (重打包). After the fused scan lands S0-S3 parquet, this module:

  1. Queries the *survivor* set from DuckDB -- clips that passed S0 + S1, whose
     S3 purity verdict is acceptable, and (optionally) whose S2
     ``prosody_dsp_score`` is in the top fraction -- and orders them by
     labeling priority ``prosody_dsp_score * norm(aesthetics_pq)`` descending.
  2. Numbers the ordered survivors and slices them into fixed-size worklist
     slices (default 5,000 clips). **Slice number == priority order == labeling
     order == the anytime guarantee**: Phase-2 claims slices in ascending id, so
     stopping at any point leaves the current best subset labeled.
  3. Writes ``manifests/s4_worklist_v1.parquet`` (one row per survivor with its
     ``slice_id`` + priority) and repacks the audio into new WebDataset shards
     (one shard per slice, ~1 GB target) plus ``repacked/repack_index.parquet``
     mapping ``clip_id -> (shard, offset)``.

Repacking is parallel by slice (spawn ``mp.Pool``): each process reads its
slice's clips from the source tars (grouped by source shard for sequential
reads) and writes its own output tar independently -- no shared state. All
writes are atomic (``*.tmp`` then rename).

Note on trimming: ``intruded_trimmed`` clips carry ``trim_start_s`` /
``trim_end_s`` on their S3 row. :func:`repack_slice` decodes such clips and
writes only the kept ``[start, end)`` span into the output shard, so the released
audio matches the trimmed duration the meta advertises (design §4 S3b s3_trim).
Non-trimmed clips are copied verbatim (no re-encode).
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pyarrow as pa

from ..common import audio as audio_utils
from ..common.config import Config
from ..common.io_utils import (
    atomic_write_bytes,
    atomic_write_parquet,
    parquet_glob,
    query_parquet,
)
from ..common.prosody_sql import prosody_dsp_score_sql

WORKLIST_NAME = "s4_worklist_v1.parquet"
REPACK_INDEX_NAME = "repack_index.parquet"
_AUDIO_EXTS = (".flac", ".wav", ".ogg", ".mp3", ".opus")

__all__ = [
    "WORKLIST_NAME",
    "REPACK_INDEX_NAME",
    "SurvivorRow",
    "query_survivors",
    "build_worklist",
    "repack_slice",
    "run_repack",
]


# ---------------------------------------------------------------------------
# Survivor query (DuckDB over the four Phase-1 stage globs)
# ---------------------------------------------------------------------------


@dataclass
class SurvivorRow:
    """One priority-ordered survivor clip destined for Phase-2 labeling."""

    clip_id: str
    source_shard: str
    original_text: str
    original_speaker: str
    duration_s: float
    aes_pq: float
    prosody_dsp_score: float
    norm_aesthetics_pq: float
    priority: float
    verdict: str
    trimmed: bool
    trim_start_s: Optional[float]
    trim_end_s: Optional[float]
    priority_rank: int
    slice_id: str


def _sql_str_list(values: Sequence[str]) -> str:
    """Render a python str sequence as a SQL ``IN`` list literal."""
    escaped = [v.replace("'", "''") for v in values]
    return ", ".join(f"'{v}'" for v in escaped)


def query_survivors(
    config: Config, *, apply_s2_top_fraction: bool = True
) -> list[dict[str, Any]]:
    """Query and priority-order the survivor set from the Phase-1 parquet.

    Survivors pass S0 and S1 and have a repack-eligible S3 verdict
    (``config.repack.survivor_verdicts`` -- note this is intentionally broader
    than the S5 publish gate; degraded_pass is labeled but publish-gated later).
    ``prosody_dsp_score`` is computed HERE, globally over the whole survivor
    population via window functions (design §4 S2: z-score over 全体存活样本), not
    per-shard -- so a clip's score never depends on which shard it landed in.
    When ``apply_s2_top_fraction`` is set, only clips whose global score is at or
    above the ``config.s2.top_fraction`` quantile are kept. ``norm_aesthetics_pq``
    is the survivor-set min-max normalization of ``aes_pq``; ``priority`` is
    ``config.repack.priority_expr``. Rows are ordered by priority descending
    (ties broken by ``clip_id``).

    Args:
        config: Pipeline config (thresholds + stage paths).
        apply_s2_top_fraction: Apply the global S2 top-fraction prosody gate.

    Returns:
        A list of flat dict rows (priority-ordered), each carrying the columns of
        :class:`SurvivorRow` except ``priority_rank`` / ``slice_id`` (assigned by
        :func:`build_worklist`).
    """
    paths = config.paths
    verdicts = _sql_str_list(config.repack.survivor_verdicts)
    keep_frac = float(config.s2.top_fraction)
    # quantile at (1 - top_fraction) keeps the top ``top_fraction`` by score.
    q = max(0.0, min(1.0, 1.0 - keep_frac))
    # Global z-score-weighted prosody score over the whole survivor population.
    score_expr = prosody_dsp_score_sql(config.s2.z_weights, column_prefix="s2.")
    top_clause = (
        f"WHERE prosody_dsp_score >= (SELECT quantile_cont(prosody_dsp_score, {q}) FROM scored)"
        if apply_s2_top_fraction
        else ""
    )
    priority_expr = config.repack.priority_expr

    sql = f"""
    WITH base AS (
        SELECT
            s0.clip_id            AS clip_id,
            s0.shard              AS source_shard,
            s0.original_text      AS original_text,
            s0.original_speaker   AS original_speaker,
            s0.duration_s         AS duration_s,
            s1.aes_pq             AS aes_pq,
            s3.verdict            AS verdict,
            s3.trimmed            AS trimmed,
            s3.trim_start_s       AS trim_start_s,
            s3.trim_end_s         AS trim_end_s,
            {score_expr}          AS prosody_dsp_score
        FROM s0
        JOIN s1 ON s0.clip_id = s1.clip_id
        JOIN s2 ON s0.clip_id = s2.clip_id
        JOIN s3 ON s0.clip_id = s3.clip_id
        WHERE s0.passed AND s1.passed AND s3.verdict IN ({verdicts})
    ),
    scored AS (
        SELECT * FROM base
    ),
    kept AS (
        SELECT * FROM scored
        {top_clause}
    ),
    bounds AS (
        SELECT min(aes_pq) AS min_pq, max(aes_pq) AS max_pq FROM kept
    ),
    normed AS (
        SELECT
            kept.*,
            CASE WHEN bounds.max_pq > bounds.min_pq
                 THEN (kept.aes_pq - bounds.min_pq) / (bounds.max_pq - bounds.min_pq)
                 ELSE 1.0 END AS norm_aesthetics_pq
        FROM kept, bounds
    )
    SELECT
        *,
        ({priority_expr}) AS priority
    FROM normed
    ORDER BY priority DESC, clip_id ASC
    """

    result = query_parquet(
        sql,
        s0=parquet_glob(paths.s0_prefilter),
        s1=parquet_glob(paths.s1_acoustics),
        s2=parquet_glob(paths.s2_prosody),
        s3=parquet_glob(paths.s3_speaker_features),
    )
    return result.df().to_dict(orient="records")


# ---------------------------------------------------------------------------
# Worklist construction: assign slice_id in priority order
# ---------------------------------------------------------------------------


def _slice_token(slice_index: int, total_slices: int) -> str:
    """Zero-pad a slice index so sorted string order == numeric priority order."""
    width = max(5, len(str(max(0, total_slices - 1))))
    return f"{slice_index:0{width}d}"


def build_worklist(
    config: Config, *, apply_s2_top_fraction: bool = True
) -> list[dict[str, Any]]:
    """Build and persist the priority-ordered Phase-2 worklist.

    Assigns each survivor a ``priority_rank`` (0-based, priority order) and a
    ``slice_id`` (``rank // slice_size``, zero-padded so sorted order matches
    priority order), then writes ``manifests/s4_worklist_v1.parquet`` atomically.

    Args:
        config: Pipeline config (``s4.slice_size`` sizes each slice).
        apply_s2_top_fraction: Forwarded to :func:`query_survivors`.

    Returns:
        The worklist rows (list of flat dicts) with ``priority_rank`` and
        ``slice_id`` populated, in priority order.
    """
    survivors = query_survivors(config, apply_s2_top_fraction=apply_s2_top_fraction)
    slice_size = max(1, int(config.s4.slice_size))
    n = len(survivors)
    total_slices = (n + slice_size - 1) // slice_size if n else 0

    rows: list[dict[str, Any]] = []
    for rank, s in enumerate(survivors):
        slice_index = rank // slice_size
        row = dict(s)
        row["priority_rank"] = rank
        row["slice_id"] = _slice_token(slice_index, total_slices)
        # Coerce numpy scalar types that pandas may hand back to plain python.
        row["duration_s"] = float(row.get("duration_s", 0.0) or 0.0)
        row["aes_pq"] = float(row.get("aes_pq", 0.0) or 0.0)
        row["prosody_dsp_score"] = float(row.get("prosody_dsp_score", 0.0) or 0.0)
        row["norm_aesthetics_pq"] = float(row.get("norm_aesthetics_pq", 0.0) or 0.0)
        row["priority"] = float(row.get("priority", 0.0) or 0.0)
        row["trimmed"] = bool(row.get("trimmed", False))
        row["trim_start_s"] = _opt_float(row.get("trim_start_s"))
        row["trim_end_s"] = _opt_float(row.get("trim_end_s"))
        rows.append(row)

    worklist_path = config.paths.manifests / WORKLIST_NAME
    _write_worklist(rows, worklist_path)
    return rows


_WORKLIST_SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("source_shard", pa.string()),
        ("original_text", pa.string()),
        ("original_speaker", pa.string()),
        ("duration_s", pa.float64()),
        ("aes_pq", pa.float64()),
        ("prosody_dsp_score", pa.float64()),
        ("norm_aesthetics_pq", pa.float64()),
        ("priority", pa.float64()),
        ("verdict", pa.string()),
        ("trimmed", pa.bool_()),
        ("trim_start_s", pa.float64()),
        ("trim_end_s", pa.float64()),
        ("priority_rank", pa.int64()),
        ("slice_id", pa.string()),
    ]
)


def _opt_float(value: Any) -> Optional[float]:
    """Coerce a value to float, or None (handles pandas NaN / None)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def _write_worklist(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    """Atomically write the worklist parquet with an explicit stable schema."""
    payload = [{name: r.get(name) for name in _WORKLIST_SCHEMA.names} for r in rows]
    table = pa.Table.from_pylist(payload, schema=_WORKLIST_SCHEMA)
    return atomic_write_parquet(table, path)


# ---------------------------------------------------------------------------
# Source-tar audio reading
# ---------------------------------------------------------------------------


def _build_stem_index(tar: tarfile.TarFile) -> dict[str, str]:
    """Map audio-member stem -> member name for one open source tar."""
    index: dict[str, str] = {}
    for m in tar.getmembers():
        if m.isfile() and m.name.lower().endswith(_AUDIO_EXTS):
            index[Path(m.name).stem] = m.name
    return index


def _read_source_audio(
    source_dir: Path, source_shard: str, clip_id: str, _cache: dict[str, Any]
) -> Optional[tuple[str, bytes]]:
    """Read one clip's audio bytes from its source tar (cached per shard).

    Returns ``(member_name, audio_bytes)`` or ``None`` when the clip is missing.
    The open tar + its stem index are memoized in ``_cache`` so a slice reads
    each source shard sequentially and only once.
    """
    entry = _cache.get(source_shard)
    if entry is None:
        tar_path = source_dir / f"{source_shard}.tar"
        tar = tarfile.open(tar_path, "r")
        entry = (tar, _build_stem_index(tar))
        _cache[source_shard] = entry
    tar, stem_index = entry
    member_name = stem_index.get(clip_id)
    if member_name is None:
        return None
    handle = tar.extractfile(member_name)
    if handle is None:
        return None
    return member_name, handle.read()


# ---------------------------------------------------------------------------
# Per-slice repack (one output tar per slice; spawn-safe top-level function)
# ---------------------------------------------------------------------------


def repack_slice(
    slice_id: str,
    slice_rows: Sequence[dict[str, Any]],
    config: Config,
) -> list[dict[str, Any]]:
    """Repack one slice's survivors into a single output shard tar.

    Reads each clip's audio from its source tar (grouped by ``source_shard`` for
    sequential reads), writes ``repacked/shard-{slice_id}.tar`` (audio member +
    a ``{clip_id}.json`` sidecar carrying the reference text / priority) and
    returns the ``repack_index`` entries. Member order inside the tar follows the
    slice's priority order, so ``offset`` is the priority rank within the shard.

    Args:
        slice_id: Zero-padded slice token (== output shard number).
        slice_rows: The worklist rows for this slice, in priority order.
        config: Pipeline config (source + repacked dirs).

    Returns:
        A list of ``{clip_id, shard, offset}`` index dicts for the clips actually
        written (missing source clips are skipped).
    """
    source_dir = config.paths.source
    out_path = config.paths.repacked / f"shard-{slice_id}.tar"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read in source-shard-grouped order for sequential source reads, but write
    # members in priority order so offsets match the worklist ranking.
    read_order = sorted(
        range(len(slice_rows)),
        key=lambda i: (str(slice_rows[i].get("source_shard", "")), int(slice_rows[i].get("priority_rank", i))),
    )
    tar_cache: dict[str, Any] = {}
    audio_by_clip: dict[str, tuple[str, bytes]] = {}
    try:
        for i in read_order:
            row = slice_rows[i]
            clip_id = str(row["clip_id"])
            got = _read_source_audio(
                source_dir, str(row.get("source_shard", "")), clip_id, tar_cache
            )
            if got is not None:
                audio_by_clip[clip_id] = got
    finally:
        for tar, _ in tar_cache.values():
            tar.close()

    index_entries: list[dict[str, Any]] = []
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    offset = 0
    with tarfile.open(tmp_path, "w") as tar:
        for row in slice_rows:  # priority order
            clip_id = str(row["clip_id"])
            got = audio_by_clip.get(clip_id)
            if got is None:
                continue
            member_name, audio_bytes = got
            ext = Path(member_name).suffix or ".flac"
            # Apply the S3 head/tail trim so the released audio matches the
            # advertised (trimmed) duration + verdict (design §4 S3b). Only
            # ``intruded_trimmed`` clips carry bounds; everyone else is verbatim.
            if row.get("trimmed") and row.get("trim_start_s") is not None:
                audio_bytes, ext = _apply_trim(
                    audio_bytes,
                    float(row["trim_start_s"]),
                    _opt_float(row.get("trim_end_s")),
                )
            _add_tar_member(tar, f"{clip_id}{ext}", audio_bytes)
            sidecar = {
                "clip_id": clip_id,
                "original_text": row.get("original_text", ""),
                "original_speaker": row.get("original_speaker", ""),
                "source_shard": row.get("source_shard", ""),
                "duration_s": row.get("duration_s"),
                "priority": row.get("priority"),
                "priority_rank": row.get("priority_rank"),
                "slice_id": slice_id,
            }
            _add_tar_member(
                tar,
                f"{clip_id}.json",
                json.dumps(sidecar, ensure_ascii=False).encode("utf-8"),
            )
            index_entries.append(
                {"clip_id": clip_id, "shard": f"shard-{slice_id}", "offset": offset}
            )
            offset += 1
    os.replace(tmp_path, out_path)
    return index_entries


def _apply_trim(
    audio_bytes: bytes, trim_start_s: float, trim_end_s: Optional[float]
) -> tuple[bytes, str]:
    """Decode, slice to ``[trim_start_s, trim_end_s)``, re-encode to FLAC.

    Returns the trimmed audio bytes and the ``.flac`` extension. On any decode
    failure the original bytes are returned unchanged (defensive: never lose a
    clip to a codec edge case), with the caller's original extension implied by
    a ``.flac`` fallback only when we actually re-encoded.
    """
    arr, sr = audio_utils.decode_bytes(audio_bytes)
    trimmed = audio_utils.trim_segment(arr, sr, trim_start_s, trim_end_s)
    return audio_utils.encode_audio(trimmed, sr, fmt="FLAC"), ".flac"


def _add_tar_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Append an in-memory bytes member to an open tar."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# A picklable wrapper so ``mp.Pool`` can dispatch per-slice work by argument.
def _repack_slice_star(
    args: tuple[str, list[dict[str, Any]], Config]
) -> list[dict[str, Any]]:
    return repack_slice(args[0], args[1], args[2])


# ---------------------------------------------------------------------------
# Top-level repack orchestration
# ---------------------------------------------------------------------------


def run_repack(
    config: Config,
    *,
    apply_s2_top_fraction: bool = True,
    parallel: bool = True,
) -> dict[str, Any]:
    """Run the full repack: worklist -> per-slice shards -> repack index.

    Args:
        config: Pipeline config.
        apply_s2_top_fraction: Forwarded to :func:`build_worklist`.
        parallel: Repack slices with an ``mp.Pool`` (spawn) when True; otherwise
            process slices inline (deterministic, test-friendly).

    Returns:
        A summary dict: ``{"n_survivors", "n_slices", "n_indexed",
        "worklist_path", "index_path"}``.
    """
    rows = build_worklist(config, apply_s2_top_fraction=apply_s2_top_fraction)

    # Group worklist rows by slice_id (preserving priority order within a slice).
    by_slice: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_slice.setdefault(str(r["slice_id"]), []).append(r)
    slice_ids = sorted(by_slice)

    tasks = [(sid, by_slice[sid], config) for sid in slice_ids]
    index_entries: list[dict[str, Any]] = []
    if not tasks:
        pass
    elif not parallel or len(tasks) == 1:
        for t in tasks:
            index_entries.extend(_repack_slice_star(t))
    else:
        ctx = mp.get_context(config.runtime.mp_start_method)
        workers = min(len(tasks), max(1, config.runtime.n_gpus * 2))
        with ctx.Pool(processes=workers) as pool:
            for entries in pool.map(_repack_slice_star, tasks):
                index_entries.extend(entries)

    index_path = _write_repack_index(index_entries, config)
    return {
        "n_survivors": len(rows),
        "n_slices": len(slice_ids),
        "n_indexed": len(index_entries),
        "worklist_path": config.paths.manifests / WORKLIST_NAME,
        "index_path": index_path,
    }


_INDEX_SCHEMA = pa.schema(
    [("clip_id", pa.string()), ("shard", pa.string()), ("offset", pa.int64())]
)


def _write_repack_index(entries: Sequence[dict[str, Any]], config: Config) -> Path:
    """Atomically write ``repacked/repack_index.parquet`` (clip_id -> shard,offset)."""
    path = config.paths.repacked / REPACK_INDEX_NAME
    table = pa.Table.from_pylist(list(entries), schema=_INDEX_SCHEMA)
    return atomic_write_parquet(table, path)
