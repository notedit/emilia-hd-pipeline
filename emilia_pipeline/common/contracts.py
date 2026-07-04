"""Typed row schemas and closed vocabularies for every pipeline stage.

This module is the single source of truth for row shapes. Downstream stage
code MUST import these dataclasses / pydantic models and enums rather than
inventing its own field names.

Design constraints honored here (design doc v1.1):
  * NO global speaker clustering. Speaker identity is Emilia's ``original_speaker``.
    There is no ``global_speaker_id``, ``spk_confidence``, ``max_centroid_cos`` or
    ``cluster_size`` anywhere.
  * All numeric metrics are stored, never used to hard-drop rows in the stage.
    Pass/reject is a downstream query condition.
  * The published nested JSON (§7) and the flat parquet meta row share one
    definition; :func:`flatten_meta` / :func:`unflatten_meta` bridge them.

Two families of models coexist:
  * Per-stage rows (``S0PrefilterRow`` ... ``S4LabelRow``) -- what each stage
    writes to ``stage/<stage>/part-*.parquet``.
  * The final published meta (``MetaRecord`` and its nested sub-models) -- the
    per-clip ``{clip_id}.json`` in the released tar and the flattened parquet.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Closed vocabularies (enums). str-mixin so values serialize as plain strings.
# ---------------------------------------------------------------------------


class Language(str, enum.Enum):
    """Language codes handled this cycle (only ``zh`` passes S0)."""

    ZH = "zh"
    EN = "en"
    OTHER = "other"


class RejectStage(str, enum.Enum):
    """Which stage first rejected a clip (bookkeeping; not a hard drop)."""

    S0 = "s0"
    S1 = "s1"
    S2 = "s2"
    S3 = "s3"
    S4 = "s4"
    S5 = "s5"
    NONE = "none"


class SpeakerVerdict(str, enum.Enum):
    """S3 purity-check verdict vocabulary (design §4 S3b)."""

    SINGLE = "single"
    INTRUDED_TRIMMED = "intruded_trimmed"
    INTRUDED_REJECTED = "intruded_rejected"
    OVERLAP_REJECTED = "overlap_rejected"
    DEGRADED_PASS = "degraded_pass"


class GenderPred(str, enum.Enum):
    """Coarse gender prediction from the speaker model."""

    FEMALE = "female"
    MALE = "male"
    UNKNOWN = "unknown"


class TextVerdict(str, enum.Enum):
    """S4 text-vs-audio verdict."""

    MATCH = "match"
    FIXABLE = "fixable"
    BROKEN = "broken"


class EmotionPrimary(str, enum.Enum):
    """Closed emotion vocabulary (S4 schema §5.3)."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    AFFECTIONATE = "affectionate"
    SERIOUS = "serious"


class SpeakingStyle(str, enum.Enum):
    """Closed speaking-style vocabulary."""

    NARRATION = "narration"
    CONVERSATIONAL = "conversational"
    STORYTELLING = "storytelling"
    SPEECH = "speech"
    BROADCAST = "broadcast"
    ACTING = "acting"
    VLOG = "vlog"
    INTERVIEW = "interview"


class Rhythm(str, enum.Enum):
    """Prosody rhythm vocabulary."""

    STEADY = "steady"
    VARIED = "varied"
    DRAMATIC = "dramatic"


class Scenario(str, enum.Enum):
    """Context scenario vocabulary."""

    PODCAST = "podcast"
    AUDIOBOOK = "audiobook"
    DRAMA = "drama"
    INTERVIEW = "interview"
    LECTURE = "lecture"
    VLOG = "vlog"
    CUSTOMER_SERVICE = "customer_service"
    OTHER = "other"


class Register(str, enum.Enum):
    """Context register vocabulary."""

    FORMAL = "formal"
    CASUAL = "casual"
    INTIMATE = "intimate"


class Accent(str, enum.Enum):
    """Accent vocabulary."""

    STANDARD = "standard"
    ACCENTED = "accented"
    DIALECT = "dialect"


class Paralinguistic(str, enum.Enum):
    """Closed paralinguistic-event vocabulary (multi-label list)."""

    LAUGHTER = "laughter"
    SIGH = "sigh"
    CRYING = "crying"
    BREATH_PROMINENT = "breath_prominent"
    FILLER_HEAVY = "filler_heavy"
    DISFLUENT = "disfluent"


class Defect(str, enum.Enum):
    """Closed defect vocabulary (multi-label list)."""

    TRUNCATED_HEAD = "truncated_head"
    TRUNCATED_TAIL = "truncated_tail"
    ARTIFACT = "artifact"
    OTHER = "other"


class Tier(str, enum.Enum):
    """Published quality tier."""

    S = "S"
    A = "A"
    B = "B"


class S4Status(str, enum.Enum):
    """S4 row status: OK or the API/parse failed (row kept, not dropped)."""

    OK = "ok"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Base model config: forbid extras so typos surface, allow enum values on dump.
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Per-stage row schemas (what each stage writes to parquet)
# ---------------------------------------------------------------------------


class S0PrefilterRow(_Base):
    """S0 metadata-prefilter output row (no audio decode).

    All boolean gate fields are recorded; ``passed`` is their AND but the row is
    written regardless of outcome (numeric-in, judgment-out principle).
    """

    clip_id: str
    shard: str
    original_id: str
    original_speaker: str
    original_text: str
    original_language: str
    duration_s: float
    original_dnsmos: Optional[float] = None
    # Individual gate outcomes (design §4 S0 table).
    dur_ok: bool
    lang_ok: bool
    dnsmos_ok: bool
    text_ok: bool
    passed: bool
    reject_reason: Optional[str] = None


class S1AcousticsRow(_Base):
    """S1 strict acoustic filter output row. All metrics stored; pass is a query."""

    clip_id: str
    shard: str
    # Audiobox-Aesthetics.
    aes_pq: float
    aes_pc: float
    aes_ce: float
    aes_cu: float
    # DNSMOS P.835.
    dnsmos_sig: float
    dnsmos_bak: float
    dnsmos_ovrl: float
    # CPU metrics.
    snr_db: float
    clipping_ratio: float
    bandwidth_hz: float
    # Integrated loudness (LUFS), computed once at decode (design §7 audio block).
    loudness_lufs: float
    # Convenience pass flag (recomputed downstream from thresholds; advisory).
    passed: bool
    reject_reason: Optional[str] = None


class S2ProsodyRow(_Base):
    """S2 prosody-richness DSP output row (CPU).

    Only the raw richness metrics are stored. ``prosody_dsp_score`` is a
    z-score-weighted sum that MUST be normalized over the whole surviving
    population, so it is deliberately NOT persisted per-shard here (that would
    make a clip's score depend on which shard it landed in). It is computed
    globally in DuckDB at repack / S5 time from these raw columns
    (design §4 S2: "取全体存活样本 top 40%"; principle: 数值入库、判定后置).
    """

    clip_id: str
    shard: str
    f0_mean_hz: float
    f0_std_st: float
    f0_range_st: float
    energy_std_db: float
    speech_rate_cps: float
    rate_var_cv: float
    pause_count: int
    pause_total_ms: float
    f0_tracker_confidence: float


class S3SpeakerRow(_Base):
    """S3 sliding-window purity-check output row.

    NO clustering fields. Speaker identity is ``original_speaker`` only. The
    clip-level mean embedding lives in an ``emb-{shard}.npy`` referenced by
    ``emb_file`` + ``emb_row`` (embeddings never enter parquet).
    """

    clip_id: str
    shard: str
    original_speaker: str
    # Embedding pointer into the sidecar npy.
    emb_file: str
    emb_row: int
    gender_pred: GenderPred = GenderPred.UNKNOWN
    # Purity-check window statistics.
    n_windows: int
    mean_win_cos: float
    min_win_cos: float
    f0_stability: float
    verdict: SpeakerVerdict
    intrusion_span_ms: Optional[float] = None
    trimmed: bool = False
    # If trimmed, the post-trim duration (design: "被修剪的 clip 更新时长").
    trimmed_duration_s: Optional[float] = None
    # If trimmed, the kept [start, end) span in seconds so repack can actually
    # apply the trim to the released audio (design §4 S3b s3_trim bookkeeping).
    # Without these, published audio would be untrimmed while its duration/verdict
    # claim otherwise. None when not trimmed.
    trim_start_s: Optional[float] = None
    trim_end_s: Optional[float] = None


# ----- S4 nested sub-schemas mirroring the guided-JSON schema (§5.3) --------


class EmotionLabel(_Base):
    """Emotion block of the S4 guided-JSON schema."""

    primary: EmotionPrimary
    secondary: Optional[EmotionPrimary] = None
    intensity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)


class ProsodyLabel(_Base):
    """Prosody block of the S4 guided-JSON schema."""

    expressiveness: int = Field(ge=1, le=5)
    speaking_style: SpeakingStyle
    rhythm: Rhythm
    prominent_stress: bool


class ContextLabel(_Base):
    """Context block of the S4 guided-JSON schema."""

    scenario: Scenario
    register: Register
    summary: str


class LanguageLabel(_Base):
    """Language block of the S4 guided-JSON schema."""

    primary: str
    code_switch: bool
    accent: Accent


class S4GuidedJSON(_Base):
    """Exact mirror of the S4 guided-JSON output schema (design §5.3).

    This is what the omni API (or its mock) must return; the client validates
    the raw response against this model before persisting.
    """

    text_verdict: TextVerdict
    text_fixed: str
    text_punctuated: str
    emotion: EmotionLabel
    prosody: ProsodyLabel
    context: ContextLabel
    language: LanguageLabel
    paralinguistic: List[Paralinguistic] = Field(default_factory=list)
    defects: List[Defect] = Field(default_factory=list)
    usable: bool


class S4LabelRow(_Base):
    """S4 label output row = guided JSON + bookkeeping written to parquet.

    Failures are NOT dropped: ``s4_status=failed`` with ``error`` populated and
    ``labels`` left None.
    """

    clip_id: str
    slice_id: str
    model: str
    prompt_version: str
    s4_status: S4Status = S4Status.OK
    error: Optional[str] = None
    # CER of text_fixed vs original_text (computed client-side, stored not gated).
    cer_vs_original: Optional[float] = None
    labels: Optional[S4GuidedJSON] = None


# ---------------------------------------------------------------------------
# Final published meta record (§7). Nested for JSON, flattenable for parquet.
# ---------------------------------------------------------------------------


class AestheticsBlock(_Base):
    pq: float
    pc: float
    ce: float
    cu: float


class DnsmosBlock(_Base):
    sig: float
    bak: float
    ovrl: float


class AcousticsBlock(_Base):
    aesthetics: AestheticsBlock
    dnsmos_p835: DnsmosBlock
    snr_db: float
    clipping_ratio: float
    bandwidth_hz: float


class AudioBlock(_Base):
    path: str
    duration_s: float
    sample_rate: int
    channels: int
    loudness_lufs: float


class SourceBlock(_Base):
    dataset: str
    dataset_version: str
    original_id: str
    original_text: str
    original_speaker: str
    original_dnsmos: Optional[float] = None
    original_language: str


class ProsodyDspBlock(_Base):
    f0_mean_hz: float
    f0_std_st: float
    f0_range_st: float
    energy_std_db: float
    speech_rate_cps: float
    rate_var_cv: float
    pause_count: int
    pause_total_ms: float
    f0_tracker_confidence: float
    prosody_dsp_score: float


class EmbeddingRef(_Base):
    emb_file: str
    emb_row: int


class PurityCheckBlock(_Base):
    n_windows: int
    mean_win_cos: float
    min_win_cos: float
    f0_stability: float
    verdict: SpeakerVerdict
    intrusion_span_ms: Optional[float] = None
    trimmed: bool = False


class SpeakerBlock(_Base):
    """Speaker block: original_speaker only, NO global cluster fields."""

    original_speaker: str
    embedding_ref: EmbeddingRef
    gender_pred: GenderPred = GenderPred.UNKNOWN
    purity_check: PurityCheckBlock


class OmniLabelsBlock(_Base):
    """Published omni-labels block (§7) = guided JSON fields + model/version/cer."""

    model: str
    prompt_version: str
    text_verdict: TextVerdict
    text_fixed: str
    text_punctuated: str
    cer_vs_original: Optional[float] = None
    emotion: EmotionLabel
    prosody: ProsodyLabel
    context: ContextLabel
    language: LanguageLabel
    paralinguistic: List[Paralinguistic] = Field(default_factory=list)
    defects: List[Defect] = Field(default_factory=list)
    usable: bool


class SelectionBlock(_Base):
    selection_score: float
    tier: Optional[Tier] = None
    reject_reason: Optional[str] = None


class PipelineBlock(_Base):
    version: str
    stages_passed: List[str] = Field(default_factory=list)
    processed_at: str


class MetaRecord(_Base):
    """The complete per-clip published record (design §7).

    Serializes to the released tar's ``{clip_id}.json`` and, via
    :func:`flatten_meta`, to a flat parquet row. ``omni_labels`` and
    ``selection`` are Optional because a clip may exist pre-S4/pre-S5.
    """

    clip_id: str
    schema_version: str
    audio: AudioBlock
    source: SourceBlock
    acoustics: AcousticsBlock
    prosody_dsp: ProsodyDspBlock
    speaker: SpeakerBlock
    omni_labels: Optional[OmniLabelsBlock] = None
    selection: Optional[SelectionBlock] = None
    pipeline: PipelineBlock


# ---------------------------------------------------------------------------
# Flatten / unflatten helpers (nested JSON <-> flat parquet columns)
# ---------------------------------------------------------------------------

_FLATTEN_SEP = "_"
# Keys whose value is a list; stored as-is (parquet supports list columns).
_LIST_LEAVES = {"stages_passed", "paralinguistic", "defects"}


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    """Recursively flatten a nested dict into ``out`` with ``_``-joined keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            leaf = key if not prefix else f"{prefix}{_FLATTEN_SEP}{key}"
            if key in _LIST_LEAVES or not isinstance(value, dict):
                out[leaf] = value
            else:
                _flatten(leaf, value, out)
    else:
        out[prefix] = obj


def flatten_meta(record: "MetaRecord | dict") -> Dict[str, Any]:
    """Flatten a :class:`MetaRecord` (or its dict) to flat parquet columns.

    Nested paths are ``_``-joined, e.g. ``omni_labels.emotion.primary`` becomes
    ``omni_labels_emotion_primary``. List leaves are preserved as list columns.

    Args:
        record: A :class:`MetaRecord` instance or an already-dumped dict.

    Returns:
        A single-level dict suitable for a parquet row.
    """
    data = record.model_dump(mode="json") if isinstance(record, MetaRecord) else record
    out: Dict[str, Any] = {}
    _flatten("", data, out)
    return out


def unflatten_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`flatten_meta`: rebuild the nested dict from flat columns.

    Reconstruction is driven by an explicit prefix -> nested-path table so that
    block names containing underscores (``dnsmos_p835``, ``prosody_dsp``,
    ``omni_labels`` ...) are not mis-split. Unknown flat keys land at top level.
    The result can be fed to ``MetaRecord.model_validate``.

    Args:
        row: Flat column dict as produced by :func:`flatten_meta`.

    Returns:
        A nested dict matching the :class:`MetaRecord` shape.
    """
    nested: Dict[str, Any] = {}
    for flat_key, value in row.items():
        parts = _split_flat_key(flat_key)
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


# Explicit prefix -> nested path table. Longest-prefix wins, so deeper blocks
# must precede their parents. Each value is the real nested key path (block
# names with underscores stay intact).
_PREFIX_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("acoustics_aesthetics", ("acoustics", "aesthetics")),
    ("acoustics_dnsmos_p835", ("acoustics", "dnsmos_p835")),
    ("acoustics", ("acoustics",)),
    ("audio", ("audio",)),
    ("source", ("source",)),
    ("prosody_dsp", ("prosody_dsp",)),
    ("speaker_embedding_ref", ("speaker", "embedding_ref")),
    ("speaker_purity_check", ("speaker", "purity_check")),
    ("speaker", ("speaker",)),
    ("omni_labels_emotion", ("omni_labels", "emotion")),
    ("omni_labels_prosody", ("omni_labels", "prosody")),
    ("omni_labels_context", ("omni_labels", "context")),
    ("omni_labels_language", ("omni_labels", "language")),
    ("omni_labels", ("omni_labels",)),
    ("selection", ("selection",)),
    ("pipeline", ("pipeline",)),
)
# Sort by descending prefix length for correct longest-match resolution.
_PREFIX_PATHS_SORTED = tuple(
    sorted(_PREFIX_PATHS, key=lambda kv: len(kv[0]), reverse=True)
)


def _split_flat_key(flat_key: str) -> List[str]:
    """Split a flattened key into its nested path using the prefix->path table.

    The final leaf name (which may itself contain underscores, e.g.
    ``rate_var_cv``) is appended verbatim, never re-split.
    """
    for prefix, path in _PREFIX_PATHS_SORTED:
        if flat_key.startswith(prefix + _FLATTEN_SEP):
            leaf = flat_key[len(prefix) + 1 :]
            return list(path) + [leaf]
    return [flat_key]


__all__ = [
    # enums
    "Language",
    "RejectStage",
    "SpeakerVerdict",
    "GenderPred",
    "TextVerdict",
    "EmotionPrimary",
    "SpeakingStyle",
    "Rhythm",
    "Scenario",
    "Register",
    "Accent",
    "Paralinguistic",
    "Defect",
    "Tier",
    "S4Status",
    # per-stage rows
    "S0PrefilterRow",
    "S1AcousticsRow",
    "S2ProsodyRow",
    "S3SpeakerRow",
    "S4GuidedJSON",
    "S4LabelRow",
    # nested label blocks
    "EmotionLabel",
    "ProsodyLabel",
    "ContextLabel",
    "LanguageLabel",
    # meta record + blocks
    "AestheticsBlock",
    "DnsmosBlock",
    "AcousticsBlock",
    "AudioBlock",
    "SourceBlock",
    "ProsodyDspBlock",
    "EmbeddingRef",
    "PurityCheckBlock",
    "SpeakerBlock",
    "OmniLabelsBlock",
    "SelectionBlock",
    "PipelineBlock",
    "MetaRecord",
    # helpers
    "flatten_meta",
    "unflatten_meta",
]
