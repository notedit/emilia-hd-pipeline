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
import soundfile as sf

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
    # Release tier: "prime" (S3-pass + S2 top-fraction), "extended" (S3-pass,
    # below the cut) or "s3rejected" (S1 survivor rejected by S3). Decides the
    # data/{tier}/ subdirectory the clip's shard lands in.
    tier: str = "prime"
    trimmed: bool = False
    trim_start_s: Optional[float] = None
    trim_end_s: Optional[float] = None
    audio_bytes: Optional[bytes] = None
    audio_ext: str = "flac"
    # Native sample rate of the published audio bytes (probed at attach time;
    # Emilia-ZH sources mix 24/32/44.1 kHz, audio is never resampled).
    sample_rate: Optional[int] = None
    # Sticky "audio was attached" flag: the streaming packager drops
    # ``audio_bytes`` right after writing each clip to its shard, so this must
    # survive the bytes themselves for the manifest / metrics parquet.
    has_audio: bool = False
    # Flat survivor row (every queried metric column) for the top-level
    # ``metadata/phase1_metrics.parquet`` -- the Phase-2 join surface.
    metrics_row: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.audio_bytes is not None:
            self.has_audio = True


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
    config: Config,
    *,
    apply_s2_top_fraction: bool = True,
    scope: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Query and priority-order the Phase-1 publish set with full metrics.

    Every row carries a ``tier`` column computed over ALL S1 survivors:
    ``prime`` (eligible S3 verdict AND ``prosody_dsp_score`` at/above the global
    top-fraction cut, taken among S3-eligible clips), ``extended`` (eligible
    verdict, below the cut) or ``s3rejected`` (S1 survivor with a rejecting S3
    verdict). ``scope`` decides which tiers are returned:

    * ``"prime"`` -- prime only (the legacy publish set)
    * ``"s3"``    -- prime + extended (every S3-pass clip)
    * ``"s1"``    -- everything (tiers partition the release)

    Note the z-score population for ``prosody_dsp_score`` is all S1 survivors
    (exactly the set S2 computed features for), so scores are scope-independent.

    Args:
        config: Pipeline config (thresholds + stage paths).
        apply_s2_top_fraction: Legacy switch, used only when ``scope`` is None:
            True -> "prime", False -> "s3".
        scope: Publish population ("prime" / "s3" / "s1"); overrides the legacy
            switch when given.

    Returns:
        Flat dict rows in priority order, each with ``priority_rank`` assigned.
    """
    if scope is None:
        scope = "prime" if apply_s2_top_fraction else "s3"
    if scope not in ("prime", "s3", "s1"):
        raise ValueError(f"unknown publish scope: {scope!r}")
    scope_clause = {
        "prime": "WHERE tier = 'prime'",
        "s3": "WHERE s3_eligible",
        "s1": "",
    }[scope]
    paths = config.paths
    verdicts = _sql_str_list(config.repack.survivor_verdicts)
    keep_frac = float(config.s2.top_fraction)
    q = max(0.0, min(1.0, 1.0 - keep_frac))
    score_expr = prosody_dsp_score_sql(config.s2.z_weights, column_prefix="s2.")
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
            (s3.verdict IN ({verdicts})) AS s3_eligible,
            {score_expr}          AS prosody_dsp_score
        FROM s0
        JOIN s1 ON s0.clip_id = s1.clip_id
        JOIN s2 ON s0.clip_id = s2.clip_id
        JOIN s3 ON s0.clip_id = s3.clip_id
        WHERE s0.passed AND s1.passed
    ),
    cut AS (
        SELECT quantile_cont(prosody_dsp_score, {q}) AS score_cut
        FROM base WHERE s3_eligible
    ),
    scored AS (
        SELECT
            base.*,
            CASE WHEN NOT base.s3_eligible THEN 's3rejected'
                 WHEN base.prosody_dsp_score >= cut.score_cut THEN 'prime'
                 ELSE 'extended' END AS tier
        FROM base, cut
    ),
    kept AS (
        SELECT * FROM scored
        {scope_clause}
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
        "tier": str(row.get("tier", "prime")),
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
    config: Config,
    *,
    apply_s2_top_fraction: bool = True,
    include_metrics: bool = True,
    scope: Optional[str] = None,
) -> list[Phase1Entry]:
    """Load Phase-1 survivors as :class:`Phase1Entry` (no audio bytes yet)."""
    rows = query_phase1_survivors(
        config, apply_s2_top_fraction=apply_s2_top_fraction, scope=scope
    )
    entries: list[Phase1Entry] = []
    for row in rows:
        tier = str(row.get("tier", "prime"))
        # S3-rejected clips ship verbatim: their S3 trim bounds are exactly the
        # judgement being second-guessed, so no destructive trim is applied.
        trimmed = bool(row.get("trimmed", False)) and tier != "s3rejected"
        entries.append(
            Phase1Entry(
                clip_id=str(row["clip_id"]),
                source_shard=str(row.get("source_shard", "")),
                meta_json=build_meta_json(row, config, include_metrics=include_metrics),
                priority_rank=int(row.get("priority_rank", 0)),
                tier=tier,
                trimmed=trimmed,
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
                e.has_audio = True
                e.sample_rate = _probe_sample_rate(raw)
                attached += 1
    return attached


def _probe_sample_rate(audio_bytes: bytes) -> Optional[int]:
    """Header-probe the native sample rate of encoded audio bytes."""
    try:
        return int(sf.info(io.BytesIO(audio_bytes)).samplerate)
    except Exception:
        return None


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


class _RollingShardWriter:
    """Rolling WebDataset tar writer for one ``data/{tier}/`` subdirectory.

    Streaming counterpart of the old whole-set ``write_shards``: clips are
    appended one at a time (so audio bytes never accumulate in memory), shards
    roll over past ``target_shard_bytes``, and each tar is written atomically
    via ``*.tmp`` + rename. ``add`` returns the release-shard name recorded in
    the manifest (``{tier}/{prefix}-NNNNN.tar``).
    """

    def __init__(
        self,
        export_dir: Path,
        tier: str,
        *,
        target_shard_bytes: int,
        shard_prefix: str = "shard",
    ) -> None:
        self.data_dir = export_dir / "data" / tier
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tier = tier
        self.cap = target_shard_bytes
        self.prefix = shard_prefix
        self.shard_paths: list[Path] = []
        self._idx = 0
        self._cur: Optional[tarfile.TarFile] = None
        self._tmp: Optional[Path] = None
        self._final: Optional[Path] = None
        self._bytes = 0

    def _open(self) -> None:
        self._final = self.data_dir / f"{self.prefix}-{self._idx:05d}.tar"
        self._tmp = self._final.with_name(self._final.name + ".tmp")
        self._cur = tarfile.open(self._tmp, "w")
        self._bytes = 0

    def _close_current(self) -> None:
        if self._cur is None:
            return
        self._cur.close()
        os.replace(self._tmp, self._final)
        self.shard_paths.append(self._final)
        self._cur = None

    def add(self, entry: Phase1Entry) -> str:
        if self._cur is None:
            self._open()
        if entry.audio_bytes is not None:
            _add_member(self._cur, f"{entry.clip_id}.{entry.audio_ext}", entry.audio_bytes)
        meta_bytes = json.dumps(entry.meta_json, ensure_ascii=False).encode("utf-8")
        _add_member(self._cur, f"{entry.clip_id}.json", meta_bytes)
        self._bytes += _entry_nbytes(entry)
        name = f"{self.tier}/{self._final.name}"
        if self._bytes >= self.cap:
            self._close_current()
            self._idx += 1
        return name

    def close(self) -> list[Path]:
        self._close_current()
        return self.shard_paths


# ---------------------------------------------------------------------------
# Manifest + dataset card
# ---------------------------------------------------------------------------

_MANIFEST_SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("source_shard", pa.string()),
        ("shard", pa.string()),
        ("tier", pa.string()),
        ("has_audio", pa.bool_()),
        ("audio_ext", pa.string()),
        ("sample_rate", pa.int32()),
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
                "tier": e.tier,
                "has_audio": e.has_audio,
                "audio_ext": e.audio_ext,
                "sample_rate": e.sample_rate,
                "speaker": str(e.meta_json.get("speaker", "")),
                "verdict": str(purity.get("verdict", "")),
                "duration_s": _f(e.meta_json.get("duration_s")) or 0.0,
                "priority": _f(e.meta_json.get("priority")) or 0.0,
                "priority_rank": int(e.priority_rank),
            }
        )
    return rows


def _dataset_card(
    config: Config,
    result_n: int,
    n_audio: int,
    tier_counts: Optional[dict[str, int]] = None,
) -> str:
    """Render a minimal dataset card (README.md) describing the filtered subset."""
    s0, s1 = config.s0, config.s1
    tc = tier_counts or {}
    keep_pct = int(config.s2.top_fraction * 100)
    verdicts = ", ".join(config.repack.survivor_verdicts)
    tier_rows = "\n".join(
        f"| `{t}` | {desc} | {tc.get(t, 0):,} |"
        for t, desc in (
            ("prime", f"S3 speaker-purity pass ({verdicts}) AND `prosody_dsp_score` in the global top {keep_pct}% of S3-pass clips — the curated expressive core"),
            ("extended", f"S3 pass, below the top-{keep_pct}% prosody cut — clean but prosodically flatter"),
            ("s3rejected", "S1-pass but rejected by S3 sliding-window purity (possible speaker intrusion / degradation); shipped verbatim, no trim, use at your own risk"),
        )
    )
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
filtered** view: every clip that survived the S0+S1 acoustic funnel, physically
partitioned into quality **tiers** so you can download exactly the strictness
level you want -- **before** Phase-2 emotion labeling.

Derived from [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
(CC-BY-NC-4.0); the same license and usage restrictions apply.

- **Pipeline version:** `{config.version}` (schema `{config.schema_version}`)
- **Clips:** {result_n:,} ({n_audio:,} with audio)
- **Format:** WebDataset tar shards under `data/{{tier}}/`; each clip is
  `{{clip_id}}.mp3|flac` + `{{clip_id}}.json`. Audio keeps its **original
  sample rate** (Emilia-ZH mixes 24/32/44.1 kHz -- see the `sample_rate`
  metadata column); it is never resampled by the pipeline.
- **Metadata:** `metadata/phase1_metrics.parquet` -- one flat row per clip
  (full S0-S3 metrics + `tier` + `sample_rate`, keyed by `clip_id`). Phase-2
  emotion/prosody labels are published incrementally as
  `metadata/s4_labels.parquet` with the same key; audio tars are never rewritten.

## Tiers: pick your filtering level

| Tier | Selection rule | Clips |
|------|----------------|-------|
{tier_rows}

Rules of thumb:

- **Just want the best expressive TTS data** -> download `data/prime/` only.
- **Want more hours, still clean** -> `data/prime/` + `data/extended/`, then
  optionally re-cut by `prosody_dsp_score` yourself.
- **Custom funnel** -> take all tiers and filter on
  `metadata/phase1_metrics.parquet`: every S0-S3 metric is a column, so any
  stricter (or looser) gate is a parquet query, no repacking needed.

## Filtering funnel (applied upstream of the tiers)

| Stage | What it does |
|-------|--------------|
| S0 | Metadata prefilter: {s0.min_duration_s}-{s0.max_duration_s}s, lang=`{s0.language}`, original DNSMOS ≥ {s0.min_original_dnsmos}, text ≥ {s0.min_text_chars} chars |
| S1 | Acoustic gate: aes_pq ≥ {s1.min_aes_pq}, aes_pc ≤ {s1.max_aes_pc}, aes_ce ≥ {s1.min_aes_ce}, SNR ≥ {s1.min_snr_db} dB, bandwidth ≥ {s1.min_bandwidth_hz} Hz |
| S2 | Prosody richness score (`prosody_dsp_score`, z-scored over all S1 survivors); the top {keep_pct}% cut among S3-pass clips defines `prime` |
| S3 | Sliding-window speaker purity; pass verdicts ({verdicts}) split `prime`/`extended`, the rest -> `s3rejected` |

Clips are ordered by labeling priority (`{config.repack.priority_expr}`); the
`priority_rank` field preserves that order. `intruded_trimmed` clips ship
head/tail-trimmed audio matching their advertised duration (`s3rejected` audio
is always verbatim).

## Per-clip JSON schema

Identity (`clip_id`, `source_shard`, `text`, `speaker`, `language`,
`duration_s`), `tier`, `purity` (verdict + trim bounds + gender), and labeling
`priority` are always present. When packaged with `include_metrics`, a
`metrics` block carries the full S0-S3 acoustics + prosody numbers.
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
    scope: Optional[str] = None,
) -> Phase1PackResult:
    """Produce the Phase-1 release folder (tiered shards + manifest + card).

    Shards are partitioned by tier under ``data/{tier}/``. Audio is attached
    and written *streaming*, one source shard at a time, so peak memory stays
    around a single source tar regardless of publish scope. The write order is
    a seeded shuffle of the entry list grouped by source shard (source-shard
    visiting order and within-shard order are both decorrelated; use a
    WebDataset shuffle buffer downstream for full mixing).

    Args:
        config: Pipeline config (``hf`` block supplies defaults).
        apply_s2_top_fraction: Override ``hf.apply_s2_top_fraction`` (legacy
            scope switch, used only when ``scope``/``hf.publish_scope`` unset).
        include_metrics: Override ``hf.include_metrics``.
        target_shard_bytes: Override ``hf.shard_bytes``.
        shuffle: Deterministically shuffle entries so shard order decorrelates
            from speaker (default True; set False to keep strict priority order).
        seed: Shuffle seed.
        entries: Pre-built entries with audio already attached (skips loading
            from disk; used by tests). Written directly, no streaming.
        mark_done: Write the ``pack/phase1`` done marker after the manifest lands.
        scope: Publish population ("prime" / "s3" / "s1"); default
            ``hf.publish_scope``, falling back to the legacy switch.

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
    publish_scope = scope or config.hf.publish_scope or ("prime" if apply_top else "s3")

    export_dir = Path(config.paths.export) / EXPORT_SUBDIR
    prefetched = entries is not None
    if entries is None:
        entries = load_phase1_entries(
            config, include_metrics=inc_metrics, scope=publish_scope
        )
    entries = list(entries)
    ordered = shuffle_entries(entries, seed=seed) if shuffle else entries

    writers: dict[str, _RollingShardWriter] = {}
    shard_assignment: dict[str, str] = {}

    def _writer(tier: str) -> _RollingShardWriter:
        if tier not in writers:
            writers[tier] = _RollingShardWriter(
                export_dir, tier, target_shard_bytes=cap
            )
        return writers[tier]

    if prefetched:
        for e in ordered:
            shard_assignment[e.clip_id] = _writer(e.tier).add(e)
    else:
        # Stream: group the (already shuffled) order by source shard, attach
        # one source tar's audio at a time, write, then drop the bytes.
        by_shard: dict[str, list[Phase1Entry]] = {}
        for e in ordered:
            by_shard.setdefault(e.source_shard, []).append(e)
        for i, (shard, shard_entries) in enumerate(by_shard.items(), 1):
            attach_audio_from_source(shard_entries, config)
            for e in shard_entries:
                shard_assignment[e.clip_id] = _writer(e.tier).add(e)
                e.audio_bytes = None  # has_audio / sample_rate persist
            if i % 25 == 0 or i == len(by_shard):
                print(
                    f"[package] source shards {i}/{len(by_shard)}, "
                    f"{len(shard_assignment)} clips written",
                    flush=True,
                )

    shard_paths: list[Path] = []
    for w in writers.values():
        shard_paths.extend(w.close())

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
    tier_counts: dict[str, int] = {}
    for e in ordered:
        tier_counts[e.tier] = tier_counts.get(e.tier, 0) + 1
    (export_dir / "README.md").write_text(
        _dataset_card(config, len(ordered), n_audio, tier_counts), encoding="utf-8"
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
        row["tier"] = e.tier
        row["has_audio"] = e.has_audio
        row["audio_ext"] = e.audio_ext
        row["sample_rate"] = e.sample_rate
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
