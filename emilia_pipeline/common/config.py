"""Pydantic config models mirroring the design doc's ``pipeline_v1.yaml``.

Every threshold, weight, path and API setting the pipeline needs lives here as
a validated pydantic model. :func:`load_config` reads the YAML and returns a
fully-typed :class:`Config`. Thresholds are stored, not applied, by stages --
the values here are only consulted by downstream SQL / advisory pass flags.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Disk layout (§3)
# ---------------------------------------------------------------------------


class PathsConfig(_Base):
    """Filesystem layout per design §3. All are directories unless noted."""

    root: Path = Path("/data/emilia-expressive")
    source: Path = Path("/data/emilia-expressive/source")
    stage: Path = Path("/data/emilia-expressive/stage")
    s0_prefilter: Path = Path("/data/emilia-expressive/stage/s0_prefilter")
    s1_acoustics: Path = Path("/data/emilia-expressive/stage/s1_acoustics")
    s2_prosody: Path = Path("/data/emilia-expressive/stage/s2_prosody")
    s3_speaker_features: Path = Path("/data/emilia-expressive/stage/s3_speaker/features")
    s3_speaker_embeddings: Path = Path(
        "/data/emilia-expressive/stage/s3_speaker/embeddings"
    )
    s4_labels: Path = Path("/data/emilia-expressive/stage/s4_labels")
    repacked: Path = Path("/data/emilia-expressive/repacked")
    manifests: Path = Path("/data/emilia-expressive/manifests")
    done: Path = Path("/data/emilia-expressive/done")
    failed: Path = Path("/data/emilia-expressive/failed")
    export: Path = Path("/data/emilia-expressive/export")


# ---------------------------------------------------------------------------
# S0 - metadata prefilter thresholds (§4 S0)
# ---------------------------------------------------------------------------


class S0Config(_Base):
    """Metadata-prefilter gates."""

    min_duration_s: float = 3.0
    max_duration_s: float = 20.0
    language: str = "zh"
    min_original_dnsmos: float = 3.2
    min_text_chars: int = 4


# ---------------------------------------------------------------------------
# S1 - strict acoustic thresholds (§4 S1)
# ---------------------------------------------------------------------------


class S1Config(_Base):
    """Strict acoustic gate thresholds (initial values, §4/§10)."""

    min_aes_pq: float = 7.0  # main gate
    max_aes_pc: float = 2.5  # low complexity = clean single-speaker
    # aesthetics CE / CU: no hard threshold, stored for S5.
    # DNSMOS was retired from S1 (CPU-serial, ~2% marginal rejection on top of
    # aes_pq; the S0 metadata gate keeps Emilia's own dnsmos >= 3.2).
    min_snr_db: float = 20.0
    max_clipping_ratio: float = 0.001
    min_bandwidth_hz: float = 8000.0


# ---------------------------------------------------------------------------
# S2 - prosody richness weights + selection quantile (§4 S2)
# ---------------------------------------------------------------------------


class ProsodyZWeights(_Base):
    """z-score weights for ``prosody_dsp_score`` (§4 S2).

    ``prosody_dsp_score`` = sum(weight_i * zscore(metric_i)). Weights should be
    re-normalized by consumers if they do not sum to 1.
    """

    f0_std_st: float = 0.30
    f0_range_st: float = 0.20
    energy_std_db: float = 0.15
    speech_rate_cps: float = 0.10
    rate_var_cv: float = 0.15
    pause_count: float = 0.10


class S2Config(_Base):
    """Prosody stage config."""

    z_weights: ProsodyZWeights = Field(default_factory=ProsodyZWeights)
    # top fraction kept by prosody_dsp_score (computed downstream in DuckDB).
    top_fraction: float = 0.40
    # silero VAD params.
    vad_threshold: float = 0.5
    vad_min_silence_ms: float = 100.0
    # pyworld F0 floor/ceil (Hz).
    f0_floor_hz: float = 60.0
    f0_ceil_hz: float = 600.0


# ---------------------------------------------------------------------------
# S3 - sliding-window purity thresholds (§4 S3)
# ---------------------------------------------------------------------------


class S3Config(_Base):
    """Sliding-window speaker purity thresholds. Bias toward not over-killing."""

    window_s: float = 1.5
    window_overlap: float = 0.5  # 50%
    embedding_dim: int = 192
    # cosine below which a window is considered a candidate intrusion.
    min_win_cos_threshold: float = 0.70
    # mean cosine below which the clip is "uniformly depressed".
    mean_win_cos_threshold: float = 0.80
    # f0_tracker_confidence below which "uniformly depressed" -> overlap_rejected.
    f0_confidence_poor: float = 0.60
    # minimum residual duration after head/tail trimming (design: 剩余>=3s).
    min_trim_residual_s: float = 3.0


# ---------------------------------------------------------------------------
# S4 - cloud omni API settings (§5.1)
# ---------------------------------------------------------------------------


class S4RetryConfig(_Base):
    """Single-request retry with exponential backoff (covers 429/5xx)."""

    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    backoff_multiplier: float = 2.0


class S4Config(_Base):
    """Qwen3-Omni cloud API settings (design §5).

    Defaults target the internal **Venus LLM proxy** (OpenAI-compatible endpoint
    with a proprietary ``venus_multimodal_url`` audio content type). Set
    ``provider="openai"`` to use the standard OpenAI ``input_audio`` content type
    instead (e.g. DashScope compatible-mode).
    """

    # "venus" -> venus_multimodal_url content; "openai" -> standard input_audio.
    provider: str = "venus"
    model: str = "server:272349"
    base_url: str = "http://v2.open.venus.oa.com/llmproxy"
    # Optional env override for the base URL (the Venus example reads OPENAI_BASE_URL).
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    prompt_version: str = "v3.1"
    temperature: float = 0.0  # greedy
    # cap on generated tokens; full-label JSON needs headroom (triage is tiny).
    max_tokens: int = 1024
    # in-flight concurrency ceiling (semaphore) -- the only real tuning knob.
    max_concurrency: int = 8
    request_timeout_s: float = 120.0
    retry: S4RetryConfig = Field(default_factory=S4RetryConfig)
    # API-required input sample rate for audio.
    sample_rate: int = 16000
    # worklist slice size (clips per Phase-2 task).
    slice_size: int = 5000
    # two-pass triage toggle (§5.2). Off by default (single-pass full labeling).
    two_pass_triage: bool = False
    triage_pass_rate_switch: float = 0.60
    # Structured/guided JSON via response_format. The Venus proxy path relies on
    # prompt-enforced JSON + client-side pydantic validation instead, so this is
    # off by default; flip on only for endpoints that support json_schema.
    use_guided_json: bool = False

    def resolved_base_url(self) -> str:
        """Return the base URL, letting ``base_url_env`` override the config value."""
        return os.environ.get(self.base_url_env) or self.base_url


# ---------------------------------------------------------------------------
# S5 - selection scoring + tiering (§7)
# ---------------------------------------------------------------------------


class SelectionWeights(_Base):
    """selection_score weights (§7). Sum should be 1.0."""

    pq: float = 0.35
    prosody_dsp_score: float = 0.25
    expressiveness_intensity: float = 0.30
    ce: float = 0.10


class S5Config(_Base):
    """S5 scoring, hard constraints and tiering."""

    weights: SelectionWeights = Field(default_factory=SelectionWeights)
    # Hard constraints (§7): applied as SQL filters, not stage-side drops.
    allowed_text_verdicts: list[str] = Field(
        default_factory=lambda: ["match", "fixable"]
    )
    forbidden_defects: list[str] = Field(
        default_factory=lambda: ["truncated_head", "truncated_tail"]
    )
    allowed_verdicts: list[str] = Field(
        default_factory=lambda: ["single", "intruded_trimmed"]
    )
    # Tier-S extra conditions.
    tier_s_top_fraction: float = 0.20
    tier_s_min_intensity: int = 3
    tier_s_exclude_emotions: list[str] = Field(default_factory=lambda: ["neutral"])
    # Tier A/B split (§7): B = "表现力一般" (expressiveness at/below this), A = rest.
    # Decoupled from the neutral-emotion check, which is only a Tier-S gate.
    tier_b_max_expressiveness: int = 2
    # sampling: stratify by original_speaker x emotion to suppress head speakers.
    max_clips_per_speaker: Optional[int] = None


# ---------------------------------------------------------------------------
# Repack (Phase 1 -> Phase 2 bridge)
# ---------------------------------------------------------------------------


class RepackConfig(_Base):
    """WebDataset repack settings (§4 repack)."""

    target_shard_bytes: int = 1_000_000_000  # ~1GB/shard
    # priority = prosody_dsp_score * norm(aesthetics_pq)
    priority_expr: str = "prosody_dsp_score * norm_aesthetics_pq"
    # S3 verdicts eligible to enter the Phase-2 labeling pool. This is SEPARATE
    # from ``s5.allowed_verdicts`` (the publish-time hard constraint): degraded_pass
    # is a "放行" verdict (design §4 S3b) and IS labeled + scored, but it is not
    # part of the S5 keep set unless also added there. Keeping the two keys
    # distinct means labeling coverage and publish gating are tuned independently.
    survivor_verdicts: list[str] = Field(
        default_factory=lambda: ["single", "intruded_trimmed", "degraded_pass"]
    )


# ---------------------------------------------------------------------------
# HuggingFace release (dataset publish; both the Phase-1 filtered subset and
# the final S5-labeled subset upload to the same repo, distinguished by revision)
# ---------------------------------------------------------------------------


class HFConfig(_Base):
    """HuggingFace dataset publish settings (design §2 收尾 tail).

    The Phase-1 filtered subset and the final S5-labeled subset publish to the
    *same* ``repo_id`` on different git ``revision``s (a branch), so consumers can
    pin either view. The token is never stored here -- it is read from the
    ``HF_TOKEN`` env at upload time, and upload is a graceful no-op when unset.
    """

    # Target Hub dataset repo, e.g. "org/emilia-expressive-zh". None -> upload
    # is skipped (folder still produced), so a repo-less config is a clean no-op.
    repo_id: Optional[str] = None
    # Keep the created repo private.
    private: bool = True
    # Git revision (branch) the Phase-1 filtered subset publishes to. The final
    # labeled release uses ``main`` (or its own revision), keeping the two views
    # in one repo per the "same repo, different revision" decision.
    phase1_revision: str = "phase1-filtered"
    # Target uncompressed bytes per output tar shard (default ~1GB).
    shard_bytes: int = 1_000_000_000
    # Embed the full S0-S3 metric block in each clip's meta JSON (research view).
    # When False, only the lean identity + verdict + priority fields are written.
    include_metrics: bool = True
    # Apply the S2 global top-fraction prosody gate when selecting survivors.
    apply_s2_top_fraction: bool = True


# ---------------------------------------------------------------------------
# Runtime (parallelism, mock toggles)
# ---------------------------------------------------------------------------


class RuntimeConfig(_Base):
    """Execution / parallelism knobs and the global mock switch."""

    n_gpus: int = 2
    total_cores: int = 176
    # cpu pool size per GPU worker = total_cores / n_gpus (design §6.2).
    cpu_per_gpu: Optional[int] = None
    audio_sample_rate: int = 24000  # Emilia native.
    # Global mock switch: when True, all GPU models and API clients use mocks.
    use_mocks: bool = False
    # Redis for dispatch (optional; multiprocessing.Queue when absent).
    redis_url: Optional[str] = None
    mp_start_method: str = "spawn"

    def resolved_cpu_per_gpu(self) -> int:
        """Return the CPU-pool size per GPU worker, deriving it if unset."""
        if self.cpu_per_gpu is not None:
            return self.cpu_per_gpu
        return max(1, self.total_cores // max(1, self.n_gpus))


# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------


class ModelsConfig(_Base):
    """GPU model identifiers / weight locations (consulted by the factory)."""

    aesthetics_model: str = "audiobox-aesthetics"
    aesthetics_weights: Optional[str] = None
    dnsmos_onnx: Optional[str] = None
    campplus_model: str = "cam++"
    campplus_weights: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class Config(_Base):
    """Top-level pipeline config (mirrors ``pipeline_v1.yaml``)."""

    version: str = "voxsift-emilia-v1.3"
    schema_version: str = "1.3"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    s0: S0Config = Field(default_factory=S0Config)
    s1: S1Config = Field(default_factory=S1Config)
    s2: S2Config = Field(default_factory=S2Config)
    s3: S3Config = Field(default_factory=S3Config)
    s4: S4Config = Field(default_factory=S4Config)
    s5: S5Config = Field(default_factory=S5Config)
    repack: RepackConfig = Field(default_factory=RepackConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def api_key(self) -> Optional[str]:
        """Read the S4 API key from the configured env var; None if unset."""
        return os.environ.get(self.s4.api_key_env)


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load and validate a pipeline config from a YAML file.

    Args:
        path: Path to a ``pipeline_v1.yaml``-style file.

    Returns:
        A fully validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the file violates the schema.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.model_validate(data)


__all__ = [
    "PathsConfig",
    "S0Config",
    "S1Config",
    "ProsodyZWeights",
    "S2Config",
    "S3Config",
    "S4RetryConfig",
    "S4Config",
    "SelectionWeights",
    "S5Config",
    "RepackConfig",
    "HFConfig",
    "RuntimeConfig",
    "ModelsConfig",
    "Config",
    "load_config",
]
