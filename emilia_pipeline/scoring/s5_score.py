"""Stage S5 - selection scoring, tiering and final-meta assembly (DuckDB SQL).

Design §7. This is the收尾 (wrap-up) stage: a one-shot pass that joins every
per-stage parquet (S0..S4) by ``clip_id``, assembles the published nested
:class:`~emilia_pipeline.common.contracts.MetaRecord` for each clip, computes
the selection score and Tier, and emits both the flat parquet and the §7
published per-clip JSON.

Everything the design mandates is honored here:
  * ``selection_score = 0.35*norm(pq) + 0.25*norm(prosody_dsp_score)
    + 0.30*norm(expressiveness*intensity) + 0.10*norm(ce)`` (weights from
    :class:`~emilia_pipeline.common.config.SelectionWeights`). Normalization is
    min-max over the candidate set (rows passing the hard constraints).
  * Hard constraints (applied as a query condition, never a stage-side drop):
    ``text_verdict != 'broken'`` AND no ``truncated_*`` defect AND
    ``verdict IN ('single','intruded_trimmed')``.
  * Tier-S: top ``tier_s_top_fraction`` by score AND emotion not in
    ``tier_s_exclude_emotions`` AND intensity >= ``tier_s_min_intensity``.
    Tier-A: quality-ok, full-emotion (non-flat). Tier-B: ok but flat (neutral).
  * Stratified sampling by ``original_speaker x emotion`` to suppress
    head-speaker dominance when ``max_clips_per_speaker`` is set. There is NO
    ``global_speaker_id`` -- speaker identity is Emilia's ``original_speaker``.

The heavy lifting (join) runs in DuckDB over append-only parquet; the scoring /
tiering / sampling logic is pure Python over the joined rows so it is unit
testable with synthetic rows and needs no GPU / API / real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..common.config import Config, SelectionWeights
from .. import PIPELINE_VERSION, SCHEMA_VERSION
from ..common.contracts import (
    AcousticsBlock,
    AestheticsBlock,
    AudioBlock,
    ContextLabel,
    DnsmosBlock,
    EmbeddingRef,
    EmotionLabel,
    LanguageLabel,
    MetaRecord,
    OmniLabelsBlock,
    PipelineBlock,
    ProsodyDspBlock,
    ProsodyLabel,
    PurityCheckBlock,
    SelectionBlock,
    SourceBlock,
    SpeakerBlock,
    Tier,
    flatten_meta,
)
from ..common.io_utils import (
    atomic_write_json,
    atomic_write_parquet,
    parquet_glob,
    query_parquet,
    write_done_marker,
)

# Reject-reason string constants (stored in SelectionBlock.reject_reason).
REASON_TEXT_BROKEN = "text_broken"
REASON_TRUNCATED = "truncated_defect"
REASON_VERDICT = "verdict_rejected"
REASON_NO_LABELS = "missing_omni_labels"
REASON_SPEAKER_QUOTA = "speaker_quota"

# Default published-audio sample rate / channels when not otherwise available.
_DEFAULT_CHANNELS = 1
_DEFAULT_LOUDNESS_LUFS = 0.0

# The S5 done-marker task id (single one-shot task over all clips).
S5_TASK_ID = "all"


# ---------------------------------------------------------------------------
# Result carriers
# ---------------------------------------------------------------------------


@dataclass
class ScoredRow:
    """A joined stage row plus its computed selection outcome.

    Attributes:
        joined: The flat dict produced by the DuckDB join (raw stage columns).
        selection_score: Weighted normalized score (0.0 for rejected rows).
        tier: Assigned Tier (``S``/``A``/``B``) or ``None`` when rejected/unsampled.
        reject_reason: Why the clip is not published, or ``None`` when kept.
    """

    joined: dict[str, Any]
    selection_score: float
    tier: Optional[Tier]
    reject_reason: Optional[str]


@dataclass
class S5Result:
    """Outcome of a full S5 pass."""

    scored: list[ScoredRow]
    meta_records: list[MetaRecord]
    flat_parquet_path: Optional[Path] = None
    json_dir: Optional[Path] = None
    n_candidates: int = 0
    n_kept: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DuckDB join over the stage parquets
# ---------------------------------------------------------------------------


def s5_join_sql(config: Optional[Config] = None) -> str:
    """Return the DuckDB SQL that joins S0..S4 by ``clip_id`` into flat columns.

    The spine is S3 (every clip that reached S5 has a speaker row + embedding);
    S0/S1/S2 are inner-joined (all survivors have them) and S4 is LEFT-joined
    (a clip may lack labels if S4 failed -- such rows fail the hard constraints
    downstream, they are not dropped here). ``prosody_dsp_score`` is materialized
    here globally (z-score over the full joined population) rather than read from
    S2, using the same expression as repack (design §4 S2).

    Args:
        config: Pipeline config supplying the prosody z-weights. Defaults to the
            built-in weights when omitted (used only by tests that inspect SQL).
    """
    from ..common.config import Config as _Cfg
    from ..common.prosody_sql import prosody_dsp_score_sql

    weights = (config or _Cfg()).s2.z_weights
    score_expr = prosody_dsp_score_sql(weights, column_prefix="s2.")
    return f"""
    SELECT
        s3.clip_id                         AS clip_id,
        s3.shard                           AS shard,
        s0.original_id                     AS original_id,
        s0.original_speaker                AS original_speaker,
        s0.original_text                   AS original_text,
        s0.original_language               AS original_language,
        s0.duration_s                      AS duration_s,
        s0.original_dnsmos                 AS original_dnsmos,
        s1.aes_pq, s1.aes_pc, s1.aes_ce, s1.aes_cu,
        s1.dnsmos_sig, s1.dnsmos_bak, s1.dnsmos_ovrl,
        s1.snr_db, s1.clipping_ratio, s1.bandwidth_hz, s1.loudness_lufs,
        s2.f0_mean_hz, s2.f0_std_st, s2.f0_range_st, s2.energy_std_db,
        s2.speech_rate_cps, s2.rate_var_cv, s2.pause_count, s2.pause_total_ms,
        s2.f0_tracker_confidence, {score_expr} AS prosody_dsp_score,
        s3.emb_file, s3.emb_row, s3.gender_pred,
        s3.n_windows, s3.mean_win_cos, s3.min_win_cos, s3.f0_stability,
        s3.verdict, s3.intrusion_span_ms, s3.trimmed, s3.trimmed_duration_s,
        s4.model                           AS s4_model,
        s4.prompt_version                  AS s4_prompt_version,
        s4.s4_status                       AS s4_status,
        s4.cer_vs_original                 AS cer_vs_original,
        s4.labels.text_verdict             AS text_verdict,
        s4.labels.text_fixed               AS text_fixed,
        s4.labels.text_punctuated          AS text_punctuated,
        s4.labels.emotion.primary          AS emotion_primary,
        s4.labels.emotion.secondary        AS emotion_secondary,
        s4.labels.emotion.intensity        AS emotion_intensity,
        s4.labels.emotion.confidence       AS emotion_confidence,
        s4.labels.prosody.expressiveness   AS prosody_expressiveness,
        s4.labels.prosody.speaking_style   AS prosody_speaking_style,
        s4.labels.prosody.rhythm           AS prosody_rhythm,
        s4.labels.prosody.prominent_stress AS prosody_prominent_stress,
        s4.labels.context.scenario         AS context_scenario,
        s4.labels.context."register"       AS context_register,
        s4.labels.context.summary          AS context_summary,
        s4.labels.language.primary         AS language_primary,
        s4.labels.language.code_switch     AS language_code_switch,
        s4.labels.language.accent          AS language_accent,
        s4.labels.paralinguistic           AS paralinguistic,
        s4.labels.defects                  AS defects,
        s4.labels.usable                   AS usable
    FROM s3
    JOIN s0 USING (clip_id)
    JOIN s1 USING (clip_id)
    JOIN s2 USING (clip_id)
    LEFT JOIN s4 USING (clip_id)
    """


def load_joined_rows(config: Config) -> list[dict[str, Any]]:
    """Join all stage parquets by ``clip_id`` and return flat dict rows.

    Args:
        config: Pipeline config (supplies the per-stage parquet directories).

    Returns:
        One dict per clip present across S0..S3 (S4 columns may be None).

    Raises:
        FileNotFoundError: If a required upstream stage directory has no part
            files (S0..S3). S4 is allowed to be empty (all rows unlabeled).
    """
    relations = {
        "s0": parquet_glob(config.paths.s0_prefilter),
        "s1": parquet_glob(config.paths.s1_acoustics),
        "s2": parquet_glob(config.paths.s2_prosody),
        "s3": parquet_glob(config.paths.s3_speaker_features),
        "s4": parquet_glob(config.paths.s4_labels),
    }
    for name in ("s0", "s1", "s2", "s3"):
        if not _glob_has_files(relations[name]):
            raise FileNotFoundError(
                f"S5 requires stage '{name}' parquet; none found at {relations[name]}"
            )
    if not _glob_has_files(relations["s4"]):
        # S4 optional: drop it from the query and synthesize NULL label columns.
        return _load_without_s4(config, relations)

    result = query_parquet(s5_join_sql(config), **relations)
    return [dict(zip(_result_columns(result), row)) for row in result.fetchall()]


def _load_without_s4(config: Config, relations: Mapping[str, str]) -> list[dict[str, Any]]:
    """Join S0..S3 only (no S4 parquet yet); label columns become None."""
    from ..common.prosody_sql import prosody_dsp_score_sql

    score_expr = prosody_dsp_score_sql(config.s2.z_weights, column_prefix="s2.")
    sql = f"""
    SELECT
        s3.clip_id AS clip_id, s3.shard AS shard,
        s0.original_id, s0.original_speaker, s0.original_text,
        s0.original_language, s0.duration_s, s0.original_dnsmos,
        s1.aes_pq, s1.aes_pc, s1.aes_ce, s1.aes_cu,
        s1.dnsmos_sig, s1.dnsmos_bak, s1.dnsmos_ovrl,
        s1.snr_db, s1.clipping_ratio, s1.bandwidth_hz, s1.loudness_lufs,
        s2.f0_mean_hz, s2.f0_std_st, s2.f0_range_st, s2.energy_std_db,
        s2.speech_rate_cps, s2.rate_var_cv, s2.pause_count, s2.pause_total_ms,
        s2.f0_tracker_confidence, {score_expr} AS prosody_dsp_score,
        s3.emb_file, s3.emb_row, s3.gender_pred,
        s3.n_windows, s3.mean_win_cos, s3.min_win_cos, s3.f0_stability,
        s3.verdict, s3.intrusion_span_ms, s3.trimmed, s3.trimmed_duration_s
    FROM s3
    JOIN s0 USING (clip_id)
    JOIN s1 USING (clip_id)
    JOIN s2 USING (clip_id)
    """
    sub = {k: relations[k] for k in ("s0", "s1", "s2", "s3")}
    result = query_parquet(sql, **sub)
    cols = _result_columns(result)
    rows = [dict(zip(cols, row)) for row in result.fetchall()]
    for r in rows:
        r.setdefault("s4_status", None)
        r.setdefault("text_verdict", None)
    return rows


def _result_columns(result: Any) -> list[str]:
    """Extract column names from a DuckDB result cursor."""
    return [d[0] for d in result.description]


def _glob_has_files(glob: str) -> bool:
    """Return whether a ``part-*.parquet`` glob resolves to at least one file."""
    parent = Path(glob).parent
    pattern = Path(glob).name
    return parent.is_dir() and any(parent.glob(pattern))


# ---------------------------------------------------------------------------
# Hard constraints (§7)
# ---------------------------------------------------------------------------


def hard_constraint_reason(row: Mapping[str, Any], config: Config) -> Optional[str]:
    """Return the first failing hard constraint, or ``None`` if the row is kept.

    Constraints (design §7):
        * S4 labels present and ``text_verdict`` in ``allowed_text_verdicts``
          (i.e. not ``broken``).
        * No ``truncated_*`` defect (``forbidden_defects``).
        * S3 ``verdict`` in ``allowed_verdicts`` (``single`` / ``intruded_trimmed``).
    """
    text_verdict = row.get("text_verdict")
    if text_verdict is None:
        return REASON_NO_LABELS
    if str(text_verdict) not in set(config.s5.allowed_text_verdicts):
        return REASON_TEXT_BROKEN
    defects = row.get("defects") or []
    forbidden = set(config.s5.forbidden_defects)
    if any(str(d) in forbidden for d in defects):
        return REASON_TRUNCATED
    verdict = row.get("verdict")
    if verdict is None or str(verdict) not in set(config.s5.allowed_verdicts):
        return REASON_VERDICT
    return None


# ---------------------------------------------------------------------------
# Selection score
# ---------------------------------------------------------------------------


def _norm(values: Sequence[float]) -> list[float]:
    """Min-max normalize to [0, 1]; a constant/empty batch maps to 0.0."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def _expr_intensity(row: Mapping[str, Any]) -> float:
    """expressiveness * intensity (0.0 when labels absent)."""
    expr = row.get("prosody_expressiveness")
    inten = row.get("emotion_intensity")
    if expr is None or inten is None:
        return 0.0
    return float(expr) * float(inten)


def compute_selection_scores(
    candidates: Sequence[Mapping[str, Any]], weights: SelectionWeights
) -> list[float]:
    """Compute selection scores for the candidate set (batch-relative norm).

    Args:
        candidates: Rows passing the hard constraints (norm baseline is this set).
        weights: The four selection weights (§7).

    Returns:
        One score per candidate, in input order.
    """
    if not candidates:
        return []
    pq = _norm([float(r.get("aes_pq") or 0.0) for r in candidates])
    prosody = _norm([float(r.get("prosody_dsp_score") or 0.0) for r in candidates])
    expr = _norm([_expr_intensity(r) for r in candidates])
    ce = _norm([float(r.get("aes_ce") or 0.0) for r in candidates])
    return [
        weights.pq * pq[i]
        + weights.prosody_dsp_score * prosody[i]
        + weights.expressiveness_intensity * expr[i]
        + weights.ce * ce[i]
        for i in range(len(candidates))
    ]


# ---------------------------------------------------------------------------
# Tiering (§7)
# ---------------------------------------------------------------------------


def assign_tiers(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    config: Config,
) -> list[Tier]:
    """Assign a Tier to every candidate given its selection score.

    Per design §7:
      * Tier-S: score in the top ``tier_s_top_fraction`` AND emotion not in
        ``tier_s_exclude_emotions`` (e.g. not neutral) AND intensity >=
        ``tier_s_min_intensity``.
      * Tier-B: "表现力一般" -- ``prosody.expressiveness`` at or below
        ``tier_b_max_expressiveness``.
      * Tier-A: everything else that is quality-ok (the full-emotion middle).

    The A/B split is driven by EXPRESSIVENESS, independent of the neutral-emotion
    check (which is only a Tier-S gate). A missing expressiveness label is
    treated as low (Tier-B) since it cannot be shown to be expressive.

    Args:
        candidates: Rows passing the hard constraints.
        scores: Their selection scores (same order).
        config: Pipeline config (S5 tier thresholds).

    Returns:
        One :class:`Tier` per candidate, in input order.
    """
    n = len(candidates)
    if n == 0:
        return []
    # Rank-based top cutoff: the top ``fraction`` by score qualify for Tier-S.
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_k = max(1, math.ceil(n * config.s5.tier_s_top_fraction))
    in_top = set(order[:top_k])
    exclude = set(config.s5.tier_s_exclude_emotions)
    min_intensity = config.s5.tier_s_min_intensity
    b_max_expr = config.s5.tier_b_max_expressiveness

    tiers: list[Tier] = []
    for i, row in enumerate(candidates):
        emotion = str(row.get("emotion_primary") or "")
        intensity = int(row.get("emotion_intensity") or 0)
        expr_raw = row.get("prosody_expressiveness")
        expressiveness = int(expr_raw) if expr_raw is not None else 0
        if i in in_top and emotion not in exclude and intensity >= min_intensity:
            tiers.append(Tier.S)
        elif expressiveness <= b_max_expr:
            # 表现力一般 -> Tier-B, regardless of emotion.
            tiers.append(Tier.B)
        else:
            tiers.append(Tier.A)
    return tiers


# ---------------------------------------------------------------------------
# Stratified sampling (suppress head-speaker dominance)
# ---------------------------------------------------------------------------


def stratified_sample(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    config: Config,
) -> list[bool]:
    """Return a keep-mask stratifying by ``original_speaker x emotion``.

    When ``config.s5.max_clips_per_speaker`` is None, everything is kept. Else,
    per speaker we bucket clips by emotion, sort each bucket by score desc, then
    round-robin across emotion buckets so no single emotion (or speaker) can
    dominate the published subset. NO global_speaker_id -- grouping key is the
    Emilia ``original_speaker`` string.

    Args:
        candidates: Rows passing the hard constraints.
        scores: Their selection scores (same order).
        config: Pipeline config.

    Returns:
        A boolean keep-mask aligned with ``candidates``.
    """
    cap = config.s5.max_clips_per_speaker
    n = len(candidates)
    keep = [True] * n
    if cap is None or n == 0:
        return keep

    # speaker -> emotion -> list of candidate indices.
    by_speaker: dict[str, dict[str, list[int]]] = {}
    for i, row in enumerate(candidates):
        spk = str(row.get("original_speaker") or "")
        emo = str(row.get("emotion_primary") or "")
        by_speaker.setdefault(spk, {}).setdefault(emo, []).append(i)

    for spk, buckets in by_speaker.items():
        total = sum(len(v) for v in buckets.values())
        if total <= cap:
            continue
        # Sort each emotion bucket by score desc (stable, deterministic).
        for emo in buckets:
            buckets[emo].sort(key=lambda idx: (scores[idx], -idx), reverse=True)
        # Round-robin pick across emotion buckets until the quota is filled.
        picked: set[int] = set()
        # Order emotions deterministically by their best score.
        emo_order = sorted(
            buckets.keys(),
            key=lambda e: (scores[buckets[e][0]], e),
            reverse=True,
        )
        cursors = {e: 0 for e in emo_order}
        while len(picked) < cap:
            progressed = False
            for emo in emo_order:
                if len(picked) >= cap:
                    break
                c = cursors[emo]
                if c < len(buckets[emo]):
                    picked.add(buckets[emo][c])
                    cursors[emo] = c + 1
                    progressed = True
            if not progressed:
                break
        for emo, idxs in buckets.items():
            for idx in idxs:
                if idx not in picked:
                    keep[idx] = False
    return keep


# ---------------------------------------------------------------------------
# Meta-record assembly (§7)
# ---------------------------------------------------------------------------


def _published_duration(row: Mapping[str, Any]) -> float:
    """Post-trim duration when trimmed, else the S0 duration."""
    if row.get("trimmed") and row.get("trimmed_duration_s") is not None:
        return float(row["trimmed_duration_s"])
    return float(row.get("duration_s") or 0.0)


def build_meta_record(
    row: Mapping[str, Any],
    config: Config,
    *,
    selection_score: float,
    tier: Optional[Tier],
    reject_reason: Optional[str],
) -> MetaRecord:
    """Assemble the published :class:`MetaRecord` for one joined clip row.

    ``omni_labels`` is populated only when the row carries S4 labels; a clip
    without labels still yields a valid record (omni_labels None, rejected).

    Args:
        row: A flat joined dict from :func:`load_joined_rows`.
        config: Pipeline config.
        selection_score: The computed score for this clip.
        tier: Assigned tier or None.
        reject_reason: None if kept, else the rejection reason.

    Returns:
        A fully validated :class:`MetaRecord`.
    """
    clip_id = str(row["clip_id"])
    duration = _published_duration(row)

    audio = AudioBlock(
        path=str(row.get("audio_path") or f"{clip_id}.flac"),
        duration_s=duration,
        sample_rate=int(config.runtime.audio_sample_rate),
        channels=_DEFAULT_CHANNELS,
        loudness_lufs=float(
            row["loudness_lufs"]
            if row.get("loudness_lufs") is not None
            else _DEFAULT_LOUDNESS_LUFS
        ),
    )
    source = SourceBlock(
        dataset="emilia",
        dataset_version="emilia-large-v1",
        original_id=str(row.get("original_id") or ""),
        original_text=str(row.get("original_text") or ""),
        original_speaker=str(row.get("original_speaker") or ""),
        original_dnsmos=_opt_float(row.get("original_dnsmos")),
        original_language=str(row.get("original_language") or ""),
    )
    acoustics = AcousticsBlock(
        aesthetics=AestheticsBlock(
            pq=float(row["aes_pq"]),
            pc=float(row["aes_pc"]),
            ce=float(row["aes_ce"]),
            cu=float(row["aes_cu"]),
        ),
        dnsmos_p835=(
            DnsmosBlock(
                sig=float(row["dnsmos_sig"]),
                bak=float(row["dnsmos_bak"]),
                ovrl=float(row["dnsmos_ovrl"]),
            )
            if _opt_float(row.get("dnsmos_ovrl")) is not None
            else None  # DNSMOS retired from S1; older rows may still carry it
        ),
        snr_db=float(row["snr_db"]),
        clipping_ratio=float(row["clipping_ratio"]),
        bandwidth_hz=float(row["bandwidth_hz"]),
    )
    prosody_dsp = ProsodyDspBlock(
        f0_mean_hz=float(row["f0_mean_hz"]),
        f0_std_st=float(row["f0_std_st"]),
        f0_range_st=float(row["f0_range_st"]),
        energy_std_db=float(row["energy_std_db"]),
        speech_rate_cps=float(row["speech_rate_cps"]),
        rate_var_cv=float(row["rate_var_cv"]),
        pause_count=int(row["pause_count"]),
        pause_total_ms=float(row["pause_total_ms"]),
        f0_tracker_confidence=float(row["f0_tracker_confidence"]),
        prosody_dsp_score=float(row["prosody_dsp_score"]),
    )
    speaker = SpeakerBlock(
        original_speaker=str(row.get("original_speaker") or ""),
        embedding_ref=EmbeddingRef(
            emb_file=str(row.get("emb_file") or ""),
            emb_row=int(row.get("emb_row") if row.get("emb_row") is not None else -1),
        ),
        gender_pred=str(row.get("gender_pred") or "unknown"),
        purity_check=PurityCheckBlock(
            n_windows=int(row["n_windows"]),
            mean_win_cos=float(row["mean_win_cos"]),
            min_win_cos=float(row["min_win_cos"]),
            f0_stability=float(row["f0_stability"]),
            verdict=str(row["verdict"]),
            intrusion_span_ms=_opt_float(row.get("intrusion_span_ms")),
            trimmed=bool(row.get("trimmed") or False),
        ),
    )

    omni_labels = _build_omni_labels(row)

    selection = SelectionBlock(
        selection_score=float(selection_score),
        tier=tier,
        reject_reason=reject_reason,
    )
    stages_passed = ["s0", "s1", "s2", "s3"]
    if omni_labels is not None:
        stages_passed.append("s4")
    pipeline = PipelineBlock(
        version=PIPELINE_VERSION,
        stages_passed=stages_passed,
        processed_at=_now_iso(),
    )

    return MetaRecord(
        clip_id=clip_id,
        schema_version=SCHEMA_VERSION,
        audio=audio,
        source=source,
        acoustics=acoustics,
        prosody_dsp=prosody_dsp,
        speaker=speaker,
        omni_labels=omni_labels,
        selection=selection,
        pipeline=pipeline,
    )


def _build_omni_labels(row: Mapping[str, Any]) -> Optional[OmniLabelsBlock]:
    """Reconstruct the OmniLabelsBlock from flat S4 columns (None if unlabeled)."""
    if row.get("text_verdict") is None or row.get("emotion_primary") is None:
        return None
    return OmniLabelsBlock(
        model=str(row.get("s4_model") or ""),
        prompt_version=str(row.get("s4_prompt_version") or ""),
        text_verdict=str(row["text_verdict"]),
        text_fixed=str(row.get("text_fixed") or ""),
        text_punctuated=str(row.get("text_punctuated") or ""),
        cer_vs_original=_opt_float(row.get("cer_vs_original")),
        emotion=EmotionLabel(
            primary=str(row["emotion_primary"]),
            secondary=(
                str(row["emotion_secondary"])
                if row.get("emotion_secondary") is not None
                else None
            ),
            intensity=int(row["emotion_intensity"]),
            confidence=float(row["emotion_confidence"]),
        ),
        prosody=ProsodyLabel(
            expressiveness=int(row["prosody_expressiveness"]),
            speaking_style=str(row["prosody_speaking_style"]),
            rhythm=str(row["prosody_rhythm"]),
            prominent_stress=bool(row["prosody_prominent_stress"]),
        ),
        context=ContextLabel(
            scenario=str(row["context_scenario"]),
            register=str(row["context_register"]),
            summary=str(row.get("context_summary") or ""),
        ),
        language=LanguageLabel(
            primary=str(row.get("language_primary") or "zh"),
            code_switch=bool(row.get("language_code_switch") or False),
            accent=str(row.get("language_accent") or "standard"),
        ),
        paralinguistic=[str(p) for p in (row.get("paralinguistic") or [])],
        defects=[str(d) for d in (row.get("defects") or [])],
        usable=bool(row.get("usable") or False),
    )


def _opt_float(value: Any) -> Optional[float]:
    """Coerce to float or return None."""
    return None if value is None else float(value)


def _now_iso() -> str:
    """Current UTC time in the §7 ISO-8601 ``...Z`` form."""
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Orchestration: score + tier + assemble
# ---------------------------------------------------------------------------


def score_rows(rows: Sequence[Mapping[str, Any]], config: Config) -> list[ScoredRow]:
    """Score, tier and sample joined rows (pure; no IO).

    Rows failing the hard constraints keep ``selection_score=0.0``, ``tier=None``
    and a ``reject_reason``. Passing rows are scored, tiered, and (when a speaker
    cap is set) stratified-sampled; unsampled survivors get ``tier=None`` with
    ``reject_reason='speaker_quota'``.

    Args:
        rows: Joined flat dict rows (from :func:`load_joined_rows` or synthetic).
        config: Pipeline config.

    Returns:
        A :class:`ScoredRow` per input row, in the original order.
    """
    reasons = [hard_constraint_reason(r, config) for r in rows]
    cand_idx = [i for i, reason in enumerate(reasons) if reason is None]
    candidates = [rows[i] for i in cand_idx]

    scores = compute_selection_scores(candidates, config.s5.weights)
    tiers = assign_tiers(candidates, scores, config)
    keep = stratified_sample(candidates, scores, config)

    # Map candidate-local results back to global positions.
    score_by_i: dict[int, float] = {}
    tier_by_i: dict[int, Optional[Tier]] = {}
    reason_by_i: dict[int, Optional[str]] = {}
    for local, gi in enumerate(cand_idx):
        score_by_i[gi] = scores[local]
        if keep[local]:
            tier_by_i[gi] = tiers[local]
            reason_by_i[gi] = None
        else:
            tier_by_i[gi] = None
            reason_by_i[gi] = REASON_SPEAKER_QUOTA

    scored: list[ScoredRow] = []
    for i, row in enumerate(rows):
        if reasons[i] is not None:
            scored.append(ScoredRow(dict(row), 0.0, None, reasons[i]))
        else:
            scored.append(
                ScoredRow(
                    dict(row),
                    score_by_i[i],
                    tier_by_i[i],
                    reason_by_i[i],
                )
            )
    return scored


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def s5_flat_parquet_path(config: Config) -> Path:
    """Return the S5 flat-meta parquet path (``{export}/meta/part-all.parquet``)."""
    return Path(config.paths.export) / "meta" / "part-all.parquet"


def s5_json_dir(config: Config) -> Path:
    """Return the directory for published per-clip JSON (``{export}/meta_json``)."""
    return Path(config.paths.export) / "meta_json"


def write_s5_outputs(
    result: S5Result,
    config: Config,
    *,
    mark_done: bool = True,
    published_only: bool = True,
) -> S5Result:
    """Emit the flat parquet (all clips) and per-clip published JSON.

    The flat parquet contains every joined clip (kept + rejected) so downstream
    threshold changes are pure SQL. The per-clip JSON is the published artifact:
    by default only tiered (kept) clips get a JSON file.

    Args:
        result: A scored :class:`S5Result` (from :func:`run_s5` with write off).
        config: Pipeline config.
        mark_done: Write the ``s5`` done marker after the parquet lands.
        published_only: When True, only kept (tier != None) clips get JSON.

    Returns:
        The same result with ``flat_parquet_path`` / ``json_dir`` filled in.
    """
    flat_rows = [flatten_meta(rec) for rec in result.meta_records]
    flat_path = s5_flat_parquet_path(config)
    atomic_write_parquet(flat_rows, flat_path)
    result.flat_parquet_path = flat_path

    json_dir = s5_json_dir(config)
    for rec in result.meta_records:
        if published_only and (rec.selection is None or rec.selection.tier is None):
            continue
        atomic_write_json(rec.model_dump(mode="json"), json_dir / f"{rec.clip_id}.json")
    result.json_dir = json_dir

    if mark_done:
        write_done_marker("s5", S5_TASK_ID, config.paths.done)
    return result


def run_s5(
    config: Config,
    *,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    write: bool = True,
    mark_done: bool = True,
    published_only: bool = True,
) -> S5Result:
    """Run the full S5 pass: join -> score -> tier -> assemble -> (persist).

    Args:
        config: Pipeline config.
        rows: Pre-joined rows (skips the DuckDB join; used by tests). When None
            the rows are loaded from the stage parquets via :func:`load_joined_rows`.
        write: Persist the flat parquet + per-clip JSON when True.
        mark_done: Write the ``s5`` done marker (only when ``write``).
        published_only: Only tiered clips get a JSON file.

    Returns:
        A populated :class:`S5Result`.
    """
    joined = list(rows) if rows is not None else load_joined_rows(config)
    scored = score_rows(joined, config)

    meta_records = [
        build_meta_record(
            sr.joined,
            config,
            selection_score=sr.selection_score,
            tier=sr.tier,
            reject_reason=sr.reject_reason,
        )
        for sr in scored
    ]

    tier_counts: dict[str, int] = {}
    n_kept = 0
    for sr in scored:
        if sr.tier is not None:
            tier_counts[sr.tier.value] = tier_counts.get(sr.tier.value, 0) + 1
            n_kept += 1

    result = S5Result(
        scored=scored,
        meta_records=meta_records,
        n_candidates=sum(
            1 for sr in scored if sr.reject_reason in (None, REASON_SPEAKER_QUOTA)
        ),
        n_kept=n_kept,
        tier_counts=tier_counts,
    )
    if write:
        write_s5_outputs(
            result, config, mark_done=mark_done, published_only=published_only
        )
    return result


__all__ = [
    "REASON_TEXT_BROKEN",
    "REASON_TRUNCATED",
    "REASON_VERDICT",
    "REASON_NO_LABELS",
    "REASON_SPEAKER_QUOTA",
    "S5_TASK_ID",
    "ScoredRow",
    "S5Result",
    "s5_join_sql",
    "load_joined_rows",
    "hard_constraint_reason",
    "compute_selection_scores",
    "assign_tiers",
    "stratified_sample",
    "build_meta_record",
    "score_rows",
    "s5_flat_parquet_path",
    "s5_json_dir",
    "write_s5_outputs",
    "run_s5",
]
