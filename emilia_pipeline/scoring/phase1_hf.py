"""Publish the Phase-1 *filtered* subset straight to a HuggingFace dataset.

This is the short-circuit release path: it packages the survivors of the S0-S3
fused scan (the same set :mod:`emilia_pipeline.phase1.repack` feeds into Phase-2)
into a WebDataset dataset **without** waiting for Phase-2 labeling or S5 scoring.
Its sibling :mod:`emilia_pipeline.scoring.hf_package` publishes the final,
S5-labeled subset; both upload to the *same* ``hf.repo_id`` on different git
revisions (design decision: one repo, two views), so a consumer can pin either
the raw-filtered Phase-1 view or the fully-labeled release.

Data sources (self-contained -- no dependency on ``run_repack``):

  * **Metadata** comes from the four Phase-1 stage parquet globs, joined and
    priority-ordered by :func:`query_phase1_survivors` (which computes the global
    ``prosody_dsp_score`` exactly as repack does, via
    :func:`~emilia_pipeline.common.prosody_sql.prosody_dsp_score_sql`).
  * **Audio** is read directly from the Emilia ``source/*.tar`` shards and the S3
    head/tail trim is applied inline (mirroring :func:`repack.repack_slice`), so
    the released audio matches the advertised trimmed duration / verdict.

Each clip contributes ``{clip_id}.flac`` (audio) + ``{clip_id}.json`` (a rich
meta record carrying the full S0-S3 metric block when ``hf.include_metrics``).
Entries are deterministically shuffled so shard order decorrelates from speaker.
Upload uses ``huggingface_hub.upload_large_folder`` and is a graceful no-op when
``HF_TOKEN`` is absent (project convention: network behind a guard).
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import pyarrow as pa

from ..common import audio as audio_utils
from ..common.config import Config
from ..common.io_utils import (
    atomic_write_parquet,
    parquet_glob,
    query_parquet,
    write_done_marker,
)
from ..common.prosody_sql import prosody_dsp_score_sql
from .hf_package import HF_TOKEN_ENV, _add_member, shuffle_entries

# Done-marker task id for the Phase-1 HF packaging step.
PACK_TASK_ID = "phase1"
# Export subdirectory for the Phase-1 filtered release (kept apart from the S5
# ``export/`` release so the two never collide on disk).
EXPORT_SUBDIR = "hf_phase1"
_AUDIO_EXTS = (".flac", ".wav", ".ogg", ".mp3", ".opus")

__all__ = [
    "PACK_TASK_ID",
    "EXPORT_SUBDIR",
    "Phase1Entry",
    "Phase1PackResult",
    "query_phase1_survivors",
    "build_meta_json",
    "load_phase1_entries",
    "attach_audio_from_source",
    "write_shards",
    "build_manifest_rows",
    "write_metrics_parquet",
    "upload_s4_labels",
    "package_phase1",
    "upload_phase1_to_hf",
]


# ---------------------------------------------------------------------------
# Carriers
# ---------------------------------------------------------------------------


@dataclass
class Phase1Entry:
    """One Phase-1 survivor clip destined for the filtered release."""

    clip_id: str
    source_shard: str
    meta_json: dict[str, Any]
    priority_rank: int
    trimmed: bool = False
    trim_start_s: Optional[float] = None
    trim_end_s: Optional[float] = None
    audio_bytes: Optional[bytes] = None
    audio_ext: str = "flac"
    # Flat survivor row (every queried metric column) for the top-level
    # ``metadata/phase1_metrics.parquet`` -- the Phase-2 join surface.
    metrics_row: Optional[dict[str, Any]] = None

    @property
    def has_audio(self) -> bool:
        return self.audio_bytes is not None


@dataclass
class Phase1PackResult:
    """Outcome of a Phase-1 packaging (and optional upload) run."""

    export_dir: Path
    shard_paths: list[Path] = field(default_factory=list)
    manifest_path: Optional[Path] = None
    n_clips: int = 0
    n_with_audio: int = 0
    uploaded: bool = False
    upload_skipped_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Survivor query (full metric set, priority-ordered)
# ---------------------------------------------------------------------------

# Metric columns carried through from each stage into the release meta. Kept as
# an explicit allow-list so the published schema is stable and reviewable.
_S1_METRICS = (
    "aes_pq", "aes_pc", "aes_ce", "aes_cu",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
    "snr_db", "clipping_ratio", "bandwidth_hz", "loudness_lufs",
)
_S2_METRICS = (
    "f0_mean_hz", "f0_std_st", "f0_range_st", "energy_std_db",
    "speech_rate_cps", "rate_var_cv", "pause_count", "pause_total_ms",
    "f0_tracker_confidence",
)
_S3_METRICS = ("mean_win_cos", "min_win_cos", "f0_stability", "n_windows", "gender_pred")


def _sql_str_list(values: Sequence[str]) -> str:
    """Render a python str sequence as a SQL ``IN`` list literal."""
    escaped = [str(v).replace("'", "''") for v in values]
    return ", ".join(f"'{v}'" for v in escaped)


def query_phase1_survivors(
    config: Config, *, apply_s2_top_fraction: bool = True
) -> list[dict[str, Any]]:
    """Query and priority-order the Phase-1 survivor set with full metrics.

    Same survivor definition and ordering as
    :func:`emilia_pipeline.phase1.repack.query_survivors` (passed S0 + S1, an
    eligible S3 verdict, global ``prosody_dsp_score`` top-fraction gate, ordered
    by ``config.repack.priority_expr`` descending) -- but this projection also
    carries every S1/S2/S3 metric column so the release meta can embed them.

    Args:
        config: Pipeline config (thresholds + stage paths).
        apply_s2_top_fraction: Apply the global S2 top-fraction prosody gate.

    Returns:
        Flat dict rows in priority order, each with ``priority_rank`` assigned.
    """
    paths = config.paths
    verdicts = _sql_str_list(config.repack.survivor_verdicts)
    keep_frac = float(config.s2.top_fraction)
    q = max(0.0, min(1.0, 1.0 - keep_frac))
    score_expr = prosody_dsp_score_sql(config.s2.z_weights, column_prefix="s2.")
    top_clause = (
        f"WHERE prosody_dsp_score >= (SELECT quantile_cont(prosody_dsp_score, {q}) FROM scored)"
        if apply_s2_top_fraction
        else ""
    )
    priority_expr = config.repack.priority_expr

    s1_cols = ", ".join(f"s1.{m} AS {m}" for m in _S1_METRICS)
    s2_cols = ", ".join(f"s2.{m} AS {m}" for m in _S2_METRICS)
    s3_cols = ", ".join(f"s3.{m} AS {m}" for m in _S3_METRICS)

    sql = f"""
    WITH base AS (
        SELECT
            s0.clip_id            AS clip_id,
            s0.shard              AS source_shard,
            s0.original_text      AS original_text,
            s0.original_speaker   AS original_speaker,
            s0.original_language  AS original_language,
            s0.duration_s         AS duration_s,
            s0.original_dnsmos    AS original_dnsmos,
            s3.verdict            AS verdict,
            s3.trimmed            AS trimmed,
            s3.trim_start_s       AS trim_start_s,
            s3.trim_end_s         AS trim_end_s,
            s3.trimmed_duration_s AS trimmed_duration_s,
            s3.intrusion_span_ms  AS intrusion_span_ms,
            {s1_cols},
            {s2_cols},
            {s3_cols},
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
    rows = result.df().to_dict(orient="records")
    for rank, row in enumerate(rows):
        row["priority_rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Meta JSON construction
# ---------------------------------------------------------------------------


def _f(value: Any) -> Optional[float]:
    """Coerce to float, dropping None / NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def build_meta_json(
    row: dict[str, Any], config: Config, *, include_metrics: bool = True
) -> dict[str, Any]:
    """Build one clip's published meta dict from a survivor row.

    The lean core (identity, text, speaker, duration, purity verdict, labeling
    priority, pipeline version) is always present. The full S0-S3 metric block
    (acoustics / prosody) is nested under ``metrics`` only when ``include_metrics``.
    """
    dur = _f(row.get("trimmed_duration_s")) or _f(row.get("duration_s"))
    meta: dict[str, Any] = {
        "clip_id": str(row["clip_id"]),
        "source_shard": str(row.get("source_shard", "")),
        "text": str(row.get("original_text", "") or ""),
        "speaker": str(row.get("original_speaker", "") or ""),
        "language": str(row.get("original_language", "") or ""),
        "duration_s": dur,
        "purity": {
            "verdict": str(row.get("verdict", "")),
            "trimmed": bool(row.get("trimmed", False)),
            "trim_start_s": _f(row.get("trim_start_s")),
            "trim_end_s": _f(row.get("trim_end_s")),
            "gender_pred": str(row.get("gender_pred", "unknown")),
        },
        "priority": _f(row.get("priority")),
        "priority_rank": int(row.get("priority_rank", 0)),
        "pipeline_version": config.version,
        "schema_version": config.schema_version,
        "stage": "phase1_filtered",
    }
    if include_metrics:
        meta["metrics"] = {
            "acoustics": {m: _f(row.get(m)) for m in _S1_METRICS},
            "prosody": {
                **{m: _f(row.get(m)) for m in _S2_METRICS},
                "prosody_dsp_score": _f(row.get("prosody_dsp_score")),
                "norm_aesthetics_pq": _f(row.get("norm_aesthetics_pq")),
            },
            "purity": {m: _f(row.get(m)) for m in ("mean_win_cos", "min_win_cos", "f0_stability")},
            "original_dnsmos": _f(row.get("original_dnsmos")),
        }
    return meta


def load_phase1_entries(
    config: Config, *, apply_s2_top_fraction: bool = True, include_metrics: bool = True
) -> list[Phase1Entry]:
    """Load Phase-1 survivors as :class:`Phase1Entry` (no audio bytes yet)."""
    rows = query_phase1_survivors(
        config, apply_s2_top_fraction=apply_s2_top_fraction
    )
    entries: list[Phase1Entry] = []
    for row in rows:
        entries.append(
            Phase1Entry(
                clip_id=str(row["clip_id"]),
                source_shard=str(row.get("source_shard", "")),
                meta_json=build_meta_json(row, config, include_metrics=include_metrics),
                priority_rank=int(row.get("priority_rank", 0)),
                trimmed=bool(row.get("trimmed", False)),
                trim_start_s=_f(row.get("trim_start_s")),
                trim_end_s=_f(row.get("trim_end_s")),
                metrics_row=row,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Audio resolution directly from the source tars (with S3 trim)
# ---------------------------------------------------------------------------


def _build_stem_index(tar: tarfile.TarFile) -> dict[str, str]:
    """Map audio-member stem -> member name for one open source tar."""
    index: dict[str, str] = {}
    for m in tar.getmembers():
        if m.isfile() and m.name.lower().endswith(_AUDIO_EXTS):
            index[Path(m.name).stem] = m.name
    return index


def attach_audio_from_source(entries: Sequence[Phase1Entry], config: Config) -> int:
    """Populate ``audio_bytes`` from the Emilia source tars, applying S3 trim.

    Groups clips by ``source_shard`` so each tar is opened once and read
    sequentially. ``intruded_trimmed`` clips are decoded, sliced to their kept
    ``[trim_start_s, trim_end_s)`` span, and re-encoded to FLAC (mirroring
    :func:`emilia_pipeline.phase1.repack.repack_slice`); everyone else is copied
    verbatim. Missing shards / members leave ``audio_bytes=None`` (still packaged,
    meta-only, so a gap is explicit rather than silent).

    Returns:
        The number of entries for which audio was successfully attached.
    """
    source_dir = Path(config.paths.source)
    by_shard: dict[str, list[Phase1Entry]] = {}
    for e in entries:
        by_shard.setdefault(e.source_shard, []).append(e)

    attached = 0
    for shard, shard_entries in by_shard.items():
        tar_path = source_dir / f"{shard}.tar"
        if not tar_path.exists():
            continue
        with tarfile.open(tar_path, "r") as tar:
            stem_index = _build_stem_index(tar)
            for e in shard_entries:
                member = stem_index.get(e.clip_id)
                if member is None:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                raw = handle.read()
                ext = Path(member).suffix.lstrip(".") or "flac"
                if e.trimmed and e.trim_start_s is not None:
                    raw, ext = _apply_trim(raw, e.trim_start_s, e.trim_end_s)
                e.audio_bytes = raw
                e.audio_ext = ext
                attached += 1
    return attached


def _apply_trim(
    audio_bytes: bytes, trim_start_s: float, trim_end_s: Optional[float]
) -> tuple[bytes, str]:
    """Decode, slice to ``[start, end)``, re-encode to FLAC (returns bytes, ext)."""
    arr, sr = audio_utils.decode_bytes(audio_bytes)
    trimmed = audio_utils.trim_segment(arr, sr, trim_start_s, trim_end_s)
    return audio_utils.encode_audio(trimmed, sr, fmt="FLAC"), "flac"


# ---------------------------------------------------------------------------
# Shard writing (priority order preserved; no re-shuffle unless asked)
# ---------------------------------------------------------------------------


def _entry_nbytes(entry: Phase1Entry) -> int:
    """Approximate on-disk size of an entry (audio + meta JSON)."""
    meta_size = len(json.dumps(entry.meta_json, ensure_ascii=False).encode("utf-8"))
    return meta_size + (len(entry.audio_bytes) if entry.audio_bytes else 0)


def write_shards(
    entries: Sequence[Phase1Entry],
    export_dir: Path,
    *,
    target_shard_bytes: int,
    shard_prefix: str = "shard",
) -> list[Path]:
    """Write entries to WebDataset tar shards under ``export_dir/data``.

    Each clip contributes ``{clip_id}.{ext}`` (audio, when present) and
    ``{clip_id}.json`` (meta). Shards roll over when the accumulated size exceeds
    ``target_shard_bytes``. Written atomically via ``*.tmp`` + rename.

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


# ---------------------------------------------------------------------------
# Manifest + dataset card
# ---------------------------------------------------------------------------

_MANIFEST_SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("source_shard", pa.string()),
        ("shard", pa.string()),
        ("has_audio", pa.bool_()),
        ("audio_ext", pa.string()),
        ("speaker", pa.string()),
        ("verdict", pa.string()),
        ("duration_s", pa.float64()),
        ("priority", pa.float64()),
        ("priority_rank", pa.int64()),
    ]
)


def build_manifest_rows(
    entries: Sequence[Phase1Entry], shard_assignment: dict[str, str]
) -> list[dict[str, Any]]:
    """Build packing-manifest rows (one per clip)."""
    rows: list[dict[str, Any]] = []
    for e in entries:
        purity = e.meta_json.get("purity", {})
        rows.append(
            {
                "clip_id": e.clip_id,
                "source_shard": e.source_shard,
                "shard": shard_assignment.get(e.clip_id, ""),
                "has_audio": e.has_audio,
                "audio_ext": e.audio_ext,
                "speaker": str(e.meta_json.get("speaker", "")),
                "verdict": str(purity.get("verdict", "")),
                "duration_s": _f(e.meta_json.get("duration_s")) or 0.0,
                "priority": _f(e.meta_json.get("priority")) or 0.0,
                "priority_rank": int(e.priority_rank),
            }
        )
    return rows


def _dataset_card(config: Config, result_n: int, n_audio: int) -> str:
    """Render a minimal dataset card (README.md) describing the filtered subset."""
    s0, s1 = config.s0, config.s1
    return f"""---
license: cc-by-nc-4.0
task_categories:
  - text-to-speech
  - audio-classification
language:
  - zh
tags:
  - emilia
  - expressive-speech
  - prosody
  - phase1-filtered
---

# Emilia Expressive — Phase-1 Filtered Subset

Auto-generated by `emilia_pipeline.scoring.phase1_hf`. This is the **Phase-1
filtered** view: clips that survived the S0-S3 acoustic / prosody / speaker-purity
funnel, **before** Phase-2 emotion labeling. The fully-labeled release lives in
the same repo on a different revision.

Derived from [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
(CC-BY-NC-4.0); the same license and usage restrictions apply.

- **Pipeline version:** `{config.version}` (schema `{config.schema_version}`)
- **Clips:** {result_n} ({n_audio} with audio)
- **Format:** WebDataset tar shards under `data/`; each clip is
  `{{clip_id}}.flac` + `{{clip_id}}.json`.
- **Metadata:** `metadata/phase1_metrics.parquet` -- one flat row per clip
  (full S0-S3 metrics, keyed by `clip_id`). Phase-2 emotion/prosody labels are
  published incrementally as `metadata/s4_labels.parquet` with the same key;
  audio tars are never rewritten.

## Filtering funnel

| Stage | What it does |
|-------|--------------|
| S0 | Metadata prefilter: {s0.min_duration_s}-{s0.max_duration_s}s, lang=`{s0.language}`, original DNSMOS ≥ {s0.min_original_dnsmos}, text ≥ {s0.min_text_chars} chars |
| S1 | Acoustic gate: aes_pq ≥ {s1.min_aes_pq}, aes_pc ≤ {s1.max_aes_pc}, SNR ≥ {s1.min_snr_db} dB, bandwidth ≥ {s1.min_bandwidth_hz} Hz |
| S2 | Prosody richness (top {int(config.s2.top_fraction * 100)}% by `prosody_dsp_score`) |
| S3 | Sliding-window speaker purity (verdicts: {", ".join(config.repack.survivor_verdicts)}) |

Clips are ordered by labeling priority (`{config.repack.priority_expr}`); the
`priority_rank` field preserves that order. `intruded_trimmed` clips ship
head/tail-trimmed audio matching their advertised duration.

## Per-clip JSON schema

Identity (`clip_id`, `source_shard`, `text`, `speaker`, `language`,
`duration_s`), `purity` (verdict + trim bounds + gender), and labeling `priority`
are always present. When packaged with `include_metrics`, a `metrics` block
carries the full S0-S3 acoustics + prosody numbers.
"""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def package_phase1(
    config: Config,
    *,
    apply_s2_top_fraction: Optional[bool] = None,
    include_metrics: Optional[bool] = None,
    target_shard_bytes: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 1234,
    entries: Optional[Sequence[Phase1Entry]] = None,
    mark_done: bool = True,
) -> Phase1PackResult:
    """Produce the Phase-1 filtered release folder (shards + manifest + card).

    Args:
        config: Pipeline config (``hf`` block supplies defaults).
        apply_s2_top_fraction: Override ``hf.apply_s2_top_fraction``.
        include_metrics: Override ``hf.include_metrics``.
        target_shard_bytes: Override ``hf.shard_bytes``.
        shuffle: Deterministically shuffle entries so shard order decorrelates
            from speaker (default True; set False to keep strict priority order).
        seed: Shuffle seed.
        entries: Pre-built entries (skips loading from disk; used by tests).
        mark_done: Write the ``pack/phase1`` done marker after the manifest lands.

    Returns:
        A populated :class:`Phase1PackResult`.
    """
    apply_top = (
        config.hf.apply_s2_top_fraction
        if apply_s2_top_fraction is None
        else apply_s2_top_fraction
    )
    inc_metrics = (
        config.hf.include_metrics if include_metrics is None else include_metrics
    )
    cap = target_shard_bytes or config.hf.shard_bytes

    export_dir = Path(config.paths.export) / EXPORT_SUBDIR
    if entries is None:
        entries = load_phase1_entries(
            config, apply_s2_top_fraction=apply_top, include_metrics=inc_metrics
        )
        attach_audio_from_source(entries, config)
    entries = list(entries)

    ordered = shuffle_entries(entries, seed=seed) if shuffle else entries
    shard_paths = write_shards(ordered, export_dir, target_shard_bytes=cap)

    shard_assignment = _replay_shard_assignment(ordered, cap, shard_paths)
    manifest_rows = build_manifest_rows(ordered, shard_assignment)
    manifest_path = export_dir / "phase1_manifest_v1.parquet"
    table = (
        pa.Table.from_pylist(manifest_rows, schema=_MANIFEST_SCHEMA)
        if manifest_rows
        else pa.Table.from_pylist([], schema=_MANIFEST_SCHEMA)
    )
    atomic_write_parquet(table, manifest_path)

    # Top-level flat metrics parquet: the Phase-2 join surface. Every S0-S3
    # metric column keyed by clip_id, so later labeling releases only append a
    # sibling parquet (same key) and never rewrite the audio tars.
    write_metrics_parquet(ordered, shard_assignment, config, export_dir)

    n_audio = sum(1 for e in ordered if e.has_audio)
    (export_dir / "README.md").write_text(
        _dataset_card(config, len(ordered), n_audio), encoding="utf-8"
    )

    result = Phase1PackResult(
        export_dir=export_dir,
        shard_paths=shard_paths,
        manifest_path=manifest_path,
        n_clips=len(ordered),
        n_with_audio=n_audio,
    )
    if mark_done:
        write_done_marker("pack", PACK_TASK_ID, config.paths.done)
    return result


def write_metrics_parquet(
    entries: Sequence[Phase1Entry],
    shard_assignment: dict[str, str],
    config: Config,
    export_dir: Path,
) -> Optional[Path]:
    """Write ``metadata/phase1_metrics.parquet``: one flat row per clip.

    Carries every column of :func:`query_phase1_survivors` plus the release
    ``shard`` each clip landed in and the pipeline/schema version. This is the
    stable, queryable metadata surface: Phase-2 labels ship as a *sibling*
    parquet keyed by the same ``clip_id`` (see :func:`upload_s4_labels`), so
    adding meta never touches ``data/*.tar``.
    """
    rows: list[dict[str, Any]] = []
    for e in entries:
        if e.metrics_row is None:
            continue
        row = dict(e.metrics_row)
        row["shard"] = shard_assignment.get(e.clip_id, "")
        row["has_audio"] = e.has_audio
        row["audio_ext"] = e.audio_ext
        row["pipeline_version"] = config.version
        row["schema_version"] = config.schema_version
        rows.append(row)
    if not rows:
        return None
    path = export_dir / "metadata" / "phase1_metrics.parquet"
    return atomic_write_parquet(pa.Table.from_pylist(rows), path)


def upload_s4_labels(
    config: Config,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str]]:
    """Incrementally publish Phase-2 S4 labels next to the Phase-1 release.

    Merges every ``stage/s4_labels/part-*.parquet`` into one
    ``metadata/s4_labels.parquet`` (keyed by ``clip_id``, joins 1:1 against
    ``metadata/phase1_metrics.parquet``) and uploads just that file to the
    Phase-1 revision. Anytime-friendly: re-run it as more slices finish -- the
    file is simply replaced with a superset. Audio tars are never rewritten.

    Returns:
        ``(local_parquet_path, upload_skipped_reason)``; reason is None when
        the upload succeeded, and the path is None when no labels exist yet.
    """
    glob = parquet_glob(config.paths.s4_labels)
    export_dir = Path(config.paths.export) / EXPORT_SUBDIR
    try:
        table = query_parquet("SELECT * FROM labels", labels=glob).arrow()
    except Exception:
        return None, "no s4 label parquet found yet"
    if table.num_rows == 0:
        return None, "s4 labels are empty"
    table = table.append_column(
        "schema_version", pa.array([config.schema_version] * table.num_rows)
    )
    path = atomic_write_parquet(table, export_dir / "metadata" / "s4_labels.parquet")

    target_repo = repo_id or config.hf.repo_id
    if not target_repo:
        return path, "hf.repo_id not set; produced parquet only"
    tok = token or os.environ.get(HF_TOKEN_ENV)
    if not tok:
        return path, f"{HF_TOKEN_ENV} not set; produced parquet only"
    rev = revision or config.hf.phase1_revision
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="metadata/s4_labels.parquet",
            repo_id=target_repo,
            repo_type="dataset",
            revision=rev,
        )
        return path, None
    except Exception as exc:  # pragma: no cover - network path
        return path, f"upload failed: {exc}"


def _replay_shard_assignment(
    entries: Sequence[Phase1Entry], cap: int, shard_paths: Sequence[Path]
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


def upload_phase1_to_hf(
    config: Config,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    private: Optional[bool] = None,
    token: Optional[str] = None,
) -> Phase1PackResult:
    """Package then upload the Phase-1 filtered folder to the Hub.

    Uploads to ``repo_id`` (default ``config.hf.repo_id``) on ``revision``
    (default ``config.hf.phase1_revision``) via
    ``huggingface_hub.upload_large_folder``. When no repo or no token is
    available the folder is still produced and the upload is skipped with a
    recorded reason -- a key-less / repo-less run is a clean no-op.

    Args:
        config: Pipeline config.
        repo_id: Override ``config.hf.repo_id``.
        revision: Override ``config.hf.phase1_revision``.
        private: Override ``config.hf.private``.
        token: Explicit token; falls back to ``HF_TOKEN`` env.

    Returns:
        The :class:`Phase1PackResult`, with ``uploaded`` / ``upload_skipped_reason``.
    """
    result = package_phase1(config)

    target_repo = repo_id or config.hf.repo_id
    if not target_repo:
        result.upload_skipped_reason = "hf.repo_id not set; produced folder only"
        return result

    tok = token or os.environ.get(HF_TOKEN_ENV)
    if not tok:
        result.upload_skipped_reason = f"{HF_TOKEN_ENV} not set; produced folder only"
        return result

    rev = revision or config.hf.phase1_revision
    is_private = config.hf.private if private is None else private
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        api.create_repo(
            target_repo, repo_type="dataset", private=is_private, exist_ok=True
        )
        api.create_branch(
            repo_id=target_repo, repo_type="dataset", branch=rev, exist_ok=True
        )
        api.upload_large_folder(
            repo_id=target_repo,
            folder_path=str(result.export_dir),
            repo_type="dataset",
            revision=rev,
        )
        result.uploaded = True
    except Exception as exc:  # pragma: no cover - network path
        result.upload_skipped_reason = f"upload failed: {exc}"
    return result
