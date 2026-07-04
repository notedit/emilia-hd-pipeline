"""Stage S2 - prosody-richness coarse filter (CPU DSP).

Design §4 S2. Goal: cheaply judge "有没有戏" (is there prosodic drama) and cut
flat readings, deliberately loose -- fine judgment is deferred to S4. All
numbers are stored, never used to hard-drop rows here; the top-40% percentile
gate is computed downstream in DuckDB (design §1, principle 5).

Metrics produced per clip (mirrors :class:`S2ProsodyRow`):
  * ``f0_mean_hz``            -- mean voiced F0 (Hz), from pyworld.
  * ``f0_std_st``            -- std of log-F0 in semitones (cross-gender comparable).
  * ``f0_range_st``          -- P5..P95 span of log-F0 in semitones.
  * ``energy_std_db``        -- std of frame RMS in dB.
  * ``speech_rate_cps``      -- characters / speech-second (char count from Emilia text).
  * ``rate_var_cv``          -- coefficient of variation of a windowed speaking-rate proxy.
  * ``pause_count`` / ``pause_total_ms`` -- silero-VAD silence-gap statistics.
  * ``f0_tracker_confidence`` -- voiced-frame ratio combined with F0 jump rate.
    **Passed through to S3** for the overlap verdict (design §4 S2/S3b).

``prosody_dsp_score`` is a z-score-weighted sum of the six richness metrics that
MUST be normalized over the whole surviving population, so it is deliberately NOT
stored on :class:`S2ProsodyRow` (that would make a clip's score depend on which
shard it landed in). It is materialized globally in DuckDB at repack / S5 time
via :func:`emilia_pipeline.common.prosody_sql.prosody_dsp_score_sql`, from the
raw metric columns above. :func:`compute_prosody_dsp_scores` remains as the
Python reference implementation (used by tests + calibration), but the pipeline
does not persist its output.

Heavy/optional deps (pyworld F0, silero VAD) sit behind lazy factories with
deterministic MOCK fallbacks (:func:`get_f0_tracker`, :func:`get_vad`), so the
unit tests run with zero GPU / model download / real data (project convention).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..common.audio import duration_s as _duration_s
from ..common.audio import resample
from ..common.config import Config, ProsodyZWeights
from ..common.contracts import S2ProsodyRow
from ..common.io_utils import atomic_write_parquet, write_done_marker
from ..common.models import _seeded_rng, content_hash

# --- DSP constants (module-level so tests / callers can reference them) -------

# pyworld analysis frame period (ms).
_F0_FRAME_PERIOD_MS = 5.0
# semitone reference pitch (cancels out of std/range; only fixes f0_mean scaling).
_SEMITONE_REF_HZ = 55.0
# consecutive-voiced-frame semitone delta above which we count a tracking "jump".
_JUMP_SEMITONE_THRESHOLD = 6.0
# energy framing for RMS / envelope.
_ENERGY_FRAME_S = 0.025
_ENERGY_HOP_S = 0.010
# silence floor (relative to peak RMS) below which a frame is treated as silence.
_ENERGY_SILENCE_REL = 1e-3
# sliding window for the rate-variation proxy.
_RATE_WINDOW_S = 1.0
# minimum separation between pseudo-syllable envelope peaks (s) ~ max 6.7 syll/s.
_SYLLABLE_MIN_SEP_S = 0.15
# silero requires 16k or 8k; VAD runs on a 16k resample of the native signal.
_VAD_SAMPLE_RATE = 16000

# The six richness metrics that feed prosody_dsp_score, in weight-field order.
_SCORE_METRICS = (
    "f0_std_st",
    "f0_range_st",
    "energy_std_db",
    "speech_rate_cps",
    "rate_var_cv",
    "pause_count",
)


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass
class F0Track:
    """Result of an F0 tracker: per-frame F0 in Hz (0 == unvoiced) + frame step."""

    f0_hz: np.ndarray
    frame_period_ms: float = _F0_FRAME_PERIOD_MS


@dataclass
class S2ClipInput:
    """One decoded clip handed to the S2 stage.

    ``audio`` is float32 mono in ``[-1, 1]`` at ``sr`` (Emilia native 24k). The
    stage never decodes tars itself; the fused Phase-1 worker supplies decoded
    audio (design §6.2).
    """

    clip_id: str
    audio: np.ndarray
    sr: int
    text: str


@dataclass
class ProsodyFeatures:
    """Raw per-clip prosody metrics (everything except prosody_dsp_score).

    ``prosody_dsp_score`` is a batch-relative z-score sum and so is computed in a
    second pass by :func:`compute_prosody_dsp_scores`, not here.
    """

    f0_mean_hz: float
    f0_std_st: float
    f0_range_st: float
    energy_std_db: float
    speech_rate_cps: float
    rate_var_cv: float
    pause_count: int
    pause_total_ms: float
    f0_tracker_confidence: float


# ---------------------------------------------------------------------------
# F0 tracker: interface + real (pyworld) + mock
# ---------------------------------------------------------------------------


class BaseF0Tracker(abc.ABC):
    """Interface for an F0 tracker. ``track`` returns a :class:`F0Track`."""

    is_mock: bool = False

    @abc.abstractmethod
    def track(self, audio: np.ndarray, sr: int) -> F0Track:
        """Return per-frame F0 (Hz, 0 for unvoiced) for a mono float32 signal."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release native resources. Default no-op."""


class PyworldF0Tracker(BaseF0Tracker):
    """Real F0 tracker backed by pyworld dio + stonemask on a 16 kHz resample.

    dio@16k is ~24x faster than harvest at native rate (pilot-measured on real
    Emilia clips: 0.008x vs 0.192x realtime) at the cost of systematically lower
    voiced-confidence estimates -- ``s3.f0_confidence_poor`` is calibrated for
    dio's distribution, not harvest's. S2 is deliberately a *loose* richness
    filter and the std/range metrics are population-z-scored downstream, so
    dio's absolute bias washes out of the score.
    """

    is_mock = False

    # dio cost scales with sample rate (unlike harvest); f0_ceil 600 Hz sits far
    # below the 8 kHz Nyquist, so tracking on a 16 kHz resample loses nothing.
    _TRACK_SR = 16000

    def __init__(self, f0_floor_hz: float, f0_ceil_hz: float) -> None:
        self.f0_floor_hz = float(f0_floor_hz)
        self.f0_ceil_hz = float(f0_ceil_hz)

    def track(self, audio: np.ndarray, sr: int) -> F0Track:
        import pyworld  # lazy; heavy native extension

        wav = np.asarray(audio, dtype=np.float32)
        if wav.size == 0:
            return F0Track(f0_hz=np.zeros(0, dtype=np.float64))
        if sr != self._TRACK_SR:
            wav = resample(wav, sr, self._TRACK_SR)
        x = np.ascontiguousarray(wav, dtype=np.float64)
        f0, t = pyworld.dio(
            x,
            self._TRACK_SR,
            f0_floor=self.f0_floor_hz,
            f0_ceil=self.f0_ceil_hz,
            frame_period=_F0_FRAME_PERIOD_MS,
        )
        # stonemask refines the dio estimate.
        f0 = pyworld.stonemask(x, f0, t, self._TRACK_SR)
        return F0Track(f0_hz=np.asarray(f0, dtype=np.float64), frame_period_ms=_F0_FRAME_PERIOD_MS)


class MockF0Tracker(BaseF0Tracker):
    """Deterministic F0 tracker: hash-seeded voiced contour, no native deps."""

    is_mock = True

    def __init__(self, f0_floor_hz: float, f0_ceil_hz: float) -> None:
        self.f0_floor_hz = float(f0_floor_hz)
        self.f0_ceil_hz = float(f0_ceil_hz)

    def track(self, audio: np.ndarray, sr: int) -> F0Track:
        dur = _duration_s(audio, sr)
        n = int(round(dur * 1000.0 / _F0_FRAME_PERIOD_MS))
        if n <= 0:
            return F0Track(f0_hz=np.zeros(0, dtype=np.float64))
        rng = _seeded_rng("s2_f0", content_hash(audio))
        base = float(rng.uniform(120.0, 240.0))
        glide = float(rng.uniform(2.0, 40.0))  # semitone-ish movement scale
        t = np.arange(n, dtype=np.float64)
        contour = base * np.exp(
            (glide / 12.0)
            * np.log(2.0)
            * np.sin(2 * np.pi * t / max(4.0, n / 3.0))
        )
        f0 = np.clip(contour, self.f0_floor_hz, self.f0_ceil_hz)
        # Devoice a deterministic fraction of frames (silence / unvoiced).
        voiced_ratio = float(rng.uniform(0.55, 0.9))
        mask = rng.random(n) < voiced_ratio
        f0 = np.where(mask, f0, 0.0)
        return F0Track(f0_hz=f0.astype(np.float64), frame_period_ms=_F0_FRAME_PERIOD_MS)


def get_f0_tracker(config: Config) -> BaseF0Tracker:
    """Return a real (pyworld) or mock F0 tracker.

    Falls back to the deterministic mock when ``config.runtime.use_mocks`` is set
    or pyworld cannot be imported, so tests run without the native extension.

    Args:
        config: Pipeline config (reads ``s2.f0_floor_hz`` / ``s2.f0_ceil_hz``).

    Returns:
        A :class:`BaseF0Tracker`.
    """
    floor, ceil = config.s2.f0_floor_hz, config.s2.f0_ceil_hz
    if config.runtime.use_mocks:
        return MockF0Tracker(floor, ceil)
    try:
        import pyworld  # noqa: F401
    except Exception:
        return MockF0Tracker(floor, ceil)
    return PyworldF0Tracker(floor, ceil)


# ---------------------------------------------------------------------------
# Voice-activity detector: interface + real (silero) + mock
# ---------------------------------------------------------------------------


class BaseVAD(abc.ABC):
    """Interface for a VAD. ``detect`` returns speech spans in seconds."""

    is_mock: bool = False

    @abc.abstractmethod
    def detect(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Return sorted, non-overlapping ``(start_s, end_s)`` speech segments."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release model resources. Default no-op."""


class SileroVAD(BaseVAD):
    """Real VAD backed by silero-vad (torch). Audio is resampled to 16k first."""

    is_mock = False

    def __init__(self, threshold: float, min_silence_ms: float) -> None:
        self.threshold = float(threshold)
        self.min_silence_ms = float(min_silence_ms)
        self._model = None
        self._get_ts = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._model = load_silero_vad()
            self._get_ts = get_speech_timestamps

    def detect(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        import torch

        self._ensure_loaded()
        wav = resample(np.asarray(audio, dtype=np.float32), sr, _VAD_SAMPLE_RATE)
        if wav.size == 0:
            return []
        tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        spans = self._get_ts(
            tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=_VAD_SAMPLE_RATE,
            min_silence_duration_ms=int(self.min_silence_ms),
            return_seconds=True,
        )
        return [(float(s["start"]), float(s["end"])) for s in spans]


class MockVAD(BaseVAD):
    """Deterministic VAD: hash-seeded 1-3 speech spans with small gaps."""

    is_mock = True

    def __init__(self, threshold: float, min_silence_ms: float) -> None:
        self.threshold = float(threshold)
        self.min_silence_ms = float(min_silence_ms)

    def detect(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        dur = _duration_s(audio, sr)
        if dur <= 0.0:
            return []
        rng = _seeded_rng("s2_vad", content_hash(audio))
        n_seg = int(rng.integers(1, 4))
        # Carve the clip into n_seg speech spans separated by silent gaps.
        gap = float(rng.uniform(0.12, 0.4))
        usable = max(0.0, dur - gap * (n_seg - 1))
        seg_len = usable / n_seg if n_seg else usable
        spans: list[tuple[float, float]] = []
        cursor = 0.0
        for _ in range(n_seg):
            start = cursor
            end = min(dur, start + seg_len)
            spans.append((round(start, 3), round(end, 3)))
            cursor = end + gap
            if cursor >= dur:
                break
        return spans


def get_vad(config: Config) -> BaseVAD:
    """Return a real (silero) or mock VAD.

    Falls back to the deterministic mock when ``config.runtime.use_mocks`` is set
    or silero/torch are unavailable.

    Args:
        config: Pipeline config (reads ``s2.vad_threshold`` / ``s2.vad_min_silence_ms``).

    Returns:
        A :class:`BaseVAD`.
    """
    thr, min_sil = config.s2.vad_threshold, config.s2.vad_min_silence_ms
    if config.runtime.use_mocks:
        return MockVAD(thr, min_sil)
    try:
        import silero_vad  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return MockVAD(thr, min_sil)
    return SileroVAD(thr, min_sil)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _hz_to_semitones(f0_hz: np.ndarray, ref_hz: float = _SEMITONE_REF_HZ) -> np.ndarray:
    """Convert positive F0 values (Hz) to semitones relative to ``ref_hz``."""
    return 12.0 * np.log2(np.maximum(f0_hz, 1e-9) / ref_hz)


def _f0_metrics(track: F0Track) -> tuple[float, float, float, float]:
    """Return ``(f0_mean_hz, f0_std_st, f0_range_st, f0_tracker_confidence)``.

    Std / range are computed in the semitone (log-F0) domain so they are
    comparable across speakers of different pitch (design §4 S2). Confidence
    combines the voiced-frame ratio with a penalty for large frame-to-frame
    F0 jumps (tracker instability), and is passed through to S3.
    """
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    n = f0.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    voiced = f0 > 0.0
    n_voiced = int(voiced.sum())
    voiced_ratio = n_voiced / n
    if n_voiced == 0:
        return 0.0, 0.0, 0.0, 0.0

    voiced_hz = f0[voiced]
    f0_mean_hz = float(np.mean(voiced_hz))
    st = _hz_to_semitones(voiced_hz)
    f0_std_st = float(np.std(st)) if n_voiced > 1 else 0.0
    if n_voiced >= 2:
        p5, p95 = np.percentile(st, [5.0, 95.0])
        f0_range_st = float(p95 - p5)
    else:
        f0_range_st = 0.0

    # Jump rate: fraction of consecutive-voiced frame pairs with a large delta.
    both_voiced = voiced[:-1] & voiced[1:]
    n_pairs = int(both_voiced.sum())
    if n_pairs > 0:
        st_full = _hz_to_semitones(np.where(voiced, f0, _SEMITONE_REF_HZ))
        deltas = np.abs(np.diff(st_full))[both_voiced]
        jump_rate = float(np.mean(deltas > _JUMP_SEMITONE_THRESHOLD))
    else:
        jump_rate = 0.0
    confidence = float(np.clip(voiced_ratio * (1.0 - jump_rate), 0.0, 1.0))
    return f0_mean_hz, f0_std_st, f0_range_st, confidence


def _frame_rms(audio: np.ndarray, sr: int) -> np.ndarray:
    """Framed RMS of a mono signal (frame/hop = 25/10 ms)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    frame = max(1, int(round(_ENERGY_FRAME_S * sr)))
    hop = max(1, int(round(_ENERGY_HOP_S * sr)))
    if x.size < frame:
        return np.array([np.sqrt(np.mean(x * x) + 1e-12)], dtype=np.float64)
    n_frames = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx]
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def _energy_std_db(rms: np.ndarray) -> float:
    """Std (dB) of frame RMS over non-silent frames."""
    if rms.size == 0:
        return 0.0
    peak = float(rms.max())
    if peak <= 0.0:
        return 0.0
    keep = rms >= peak * _ENERGY_SILENCE_REL
    kept = rms[keep]
    if kept.size < 2:
        return 0.0
    db = 20.0 * np.log10(kept + 1e-12)
    return float(np.std(db))


def _count_chars(text: str) -> int:
    """Count non-whitespace characters in the reference text (syllable proxy)."""
    if not text:
        return 0
    return sum(1 for ch in text if not ch.isspace())


def _pause_stats(
    segments: Sequence[tuple[float, float]], min_silence_ms: float
) -> tuple[int, float, float]:
    """Return ``(pause_count, pause_total_ms, speech_seconds)`` from VAD spans.

    Pauses are internal silence gaps between consecutive speech segments whose
    duration is at least ``min_silence_ms`` (leading/trailing silence excluded).
    """
    if not segments:
        return 0, 0.0, 0.0
    ordered = sorted(segments, key=lambda s: s[0])
    speech_seconds = float(sum(max(0.0, e - s) for s, e in ordered))
    pause_count = 0
    pause_total_ms = 0.0
    for (_, prev_end), (next_start, _) in zip(ordered[:-1], ordered[1:]):
        gap_ms = max(0.0, (next_start - prev_end) * 1000.0)
        if gap_ms >= min_silence_ms:
            pause_count += 1
            pause_total_ms += gap_ms
    return pause_count, pause_total_ms, speech_seconds


def _pick_peaks(env: np.ndarray, min_sep: int, threshold: float) -> int:
    """Count local maxima in ``env`` above ``threshold``, spaced >= ``min_sep``.

    A tiny numpy peak picker (avoids a hard scipy dependency) used to estimate
    pseudo-syllable nuclei for the speaking-rate variation proxy.
    """
    n = env.size
    if n == 0:
        return 0
    count = 0
    last = -min_sep - 1
    for i in range(n):
        if env[i] < threshold:
            continue
        left = env[i - 1] if i > 0 else -np.inf
        right = env[i + 1] if i < n - 1 else -np.inf
        if env[i] >= left and env[i] >= right and (i - last) >= min_sep:
            count += 1
            last = i
    return count


def _rate_var_cv(rms: np.ndarray) -> float:
    """Coefficient of variation of a windowed pseudo-syllable rate.

    Splits the RMS envelope into ~1 s non-overlapping windows, counts envelope
    peaks (pseudo-syllable nuclei) per window to get a local rate, and returns
    ``std / mean`` across windows. Returns 0 when fewer than two windows exist.
    """
    if rms.size == 0:
        return 0.0
    frames_per_s = 1.0 / _ENERGY_HOP_S
    win = max(1, int(round(_RATE_WINDOW_S * frames_per_s)))
    min_sep = max(1, int(round(_SYLLABLE_MIN_SEP_S * frames_per_s)))
    peak_thr = float(np.median(rms)) if rms.size else 0.0
    n_win = rms.size // win
    if n_win < 2:
        return 0.0
    rates: list[float] = []
    for w in range(n_win):
        seg = rms[w * win : (w + 1) * win]
        rates.append(_pick_peaks(seg, min_sep, peak_thr) / _RATE_WINDOW_S)
    arr = np.asarray(rates, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 0.0:
        return 0.0
    return float(arr.std() / mean)


# ---------------------------------------------------------------------------
# Per-clip feature extraction
# ---------------------------------------------------------------------------


def extract_prosody_features(
    audio: np.ndarray,
    sr: int,
    text: str,
    config: Config,
    *,
    f0_tracker: Optional[BaseF0Tracker] = None,
    vad: Optional[BaseVAD] = None,
) -> ProsodyFeatures:
    """Compute all raw S2 prosody metrics for a single clip.

    Everything except ``prosody_dsp_score`` (which is batch-relative) is filled.
    Trackers are injected for reuse across a shard; when omitted they are built
    from ``config`` via the mock-aware factories.

    Args:
        audio: float32 mono samples in ``[-1, 1]`` at ``sr``.
        sr: Native sample rate (Emilia 24k).
        text: Emilia reference text (character count drives speech rate).
        config: Pipeline config.
        f0_tracker: Optional pre-built F0 tracker; built from config if None.
        vad: Optional pre-built VAD; built from config if None.

    Returns:
        A :class:`ProsodyFeatures` with the nine raw metrics.
    """
    f0_tracker = f0_tracker if f0_tracker is not None else get_f0_tracker(config)
    vad = vad if vad is not None else get_vad(config)

    track = f0_tracker.track(audio, sr)
    f0_mean_hz, f0_std_st, f0_range_st, confidence = _f0_metrics(track)

    rms = _frame_rms(audio, sr)
    energy_std_db = _energy_std_db(rms)
    rate_var_cv = _rate_var_cv(rms)

    segments = vad.detect(audio, sr)
    pause_count, pause_total_ms, speech_seconds = _pause_stats(
        segments, config.s2.vad_min_silence_ms
    )

    n_chars = _count_chars(text)
    # Prefer VAD speech time; fall back to total duration when VAD found nothing.
    denom = speech_seconds if speech_seconds > 0.0 else _duration_s(audio, sr)
    speech_rate_cps = float(n_chars / denom) if denom > 0.0 else 0.0

    return ProsodyFeatures(
        f0_mean_hz=f0_mean_hz,
        f0_std_st=f0_std_st,
        f0_range_st=f0_range_st,
        energy_std_db=energy_std_db,
        speech_rate_cps=speech_rate_cps,
        rate_var_cv=rate_var_cv,
        pause_count=pause_count,
        pause_total_ms=pause_total_ms,
        f0_tracker_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Batch prosody_dsp_score (z-score weighted sum)
# ---------------------------------------------------------------------------


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standardize a 1-D array; returns zeros when the std is zero (or n<2)."""
    if values.size < 2:
        return np.zeros_like(values)
    std = float(values.std())
    if std <= 0.0:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def compute_prosody_dsp_scores(
    features: Sequence[ProsodyFeatures], weights: ProsodyZWeights
) -> list[float]:
    """Compute the batch-relative ``prosody_dsp_score`` for a set of clips.

    ``prosody_dsp_score = sum_i w_i * zscore_i`` over the six richness metrics,
    z-scored across the supplied batch (a shard's surviving clips). The absolute
    top-40% percentile gate is applied later in DuckDB (design §4 S2); this only
    produces the raw comparable score.

    Args:
        features: Per-clip :class:`ProsodyFeatures` (batch order preserved).
        weights: z-score weights (:class:`ProsodyZWeights`).

    Returns:
        One float score per input clip, in order.
    """
    n = len(features)
    if n == 0:
        return []
    weight_map = weights.model_dump()
    score = np.zeros(n, dtype=np.float64)
    for metric in _SCORE_METRICS:
        col = np.asarray(
            [float(getattr(f, metric)) for f in features], dtype=np.float64
        )
        score += float(weight_map[metric]) * _zscore(col)
    return [float(s) for s in score]


# ---------------------------------------------------------------------------
# Stage orchestration: features -> rows -> parquet + done marker
# ---------------------------------------------------------------------------


def run_s2_stage(
    clips: Sequence[S2ClipInput],
    shard: str,
    config: Config,
    *,
    f0_tracker: Optional[BaseF0Tracker] = None,
    vad: Optional[BaseVAD] = None,
) -> list[S2ProsodyRow]:
    """Run S2 over a batch of decoded clips and return typed rows.

    Trackers are built once and reused across the batch. ``prosody_dsp_score``
    is computed batch-relative after all per-clip features are extracted.

    Args:
        clips: Decoded clips (typically one source shard's S1-surviving clips).
        shard: Shard token stored on every row.
        config: Pipeline config.
        f0_tracker: Optional shared F0 tracker (built from config if None).
        vad: Optional shared VAD (built from config if None).

    Returns:
        A list of :class:`S2ProsodyRow`, one per input clip, in order.
    """
    f0_tracker = f0_tracker if f0_tracker is not None else get_f0_tracker(config)
    vad = vad if vad is not None else get_vad(config)

    feats = [
        extract_prosody_features(
            c.audio, c.sr, c.text, config, f0_tracker=f0_tracker, vad=vad
        )
        for c in clips
    ]

    rows: list[S2ProsodyRow] = []
    for clip, feat in zip(clips, feats):
        rows.append(
            S2ProsodyRow(
                clip_id=clip.clip_id,
                shard=shard,
                f0_mean_hz=feat.f0_mean_hz,
                f0_std_st=feat.f0_std_st,
                f0_range_st=feat.f0_range_st,
                energy_std_db=feat.energy_std_db,
                speech_rate_cps=feat.speech_rate_cps,
                rate_var_cv=feat.rate_var_cv,
                pause_count=feat.pause_count,
                pause_total_ms=feat.pause_total_ms,
                f0_tracker_confidence=feat.f0_tracker_confidence,
            )
        )
    return rows


def write_s2_shard(
    rows: Sequence[S2ProsodyRow],
    shard: str,
    config: Config,
    *,
    mark_done: bool = True,
) -> Any:
    """Atomically write S2 rows to ``s2_prosody/part-{shard}.parquet``.

    Follows the global write discipline (design §3): parquet is written via
    ``*.tmp`` then renamed by :func:`atomic_write_parquet`; the done marker is
    created only after that rename succeeds.

    Args:
        rows: S2 rows for one shard.
        shard: Shard token (used in the file name and done-marker task id).
        config: Pipeline config (supplies output / done directories).
        mark_done: When True, write the done marker after the parquet lands.

    Returns:
        The path to the written parquet file.
    """
    out_path = config.paths.s2_prosody / f"part-{shard}.parquet"
    payload = [r.model_dump() for r in rows]
    written = atomic_write_parquet(payload, out_path)
    if mark_done:
        write_done_marker("s2", shard, config.paths.done)
    return written


def process_shard_s2(
    clips: Sequence[S2ClipInput],
    shard: str,
    config: Config,
    *,
    f0_tracker: Optional[BaseF0Tracker] = None,
    vad: Optional[BaseVAD] = None,
    mark_done: bool = True,
) -> list[S2ProsodyRow]:
    """Convenience end-to-end: compute S2 rows and persist them for one shard.

    Args:
        clips: Decoded S1-surviving clips for the shard.
        shard: Shard token.
        config: Pipeline config.
        f0_tracker: Optional shared F0 tracker.
        vad: Optional shared VAD.
        mark_done: Whether to write the done marker after the parquet lands.

    Returns:
        The computed :class:`S2ProsodyRow` list (also written to disk).
    """
    rows = run_s2_stage(clips, shard, config, f0_tracker=f0_tracker, vad=vad)
    write_s2_shard(rows, shard, config, mark_done=mark_done)
    return rows


__all__ = [
    # data carriers
    "F0Track",
    "S2ClipInput",
    "ProsodyFeatures",
    # tracker / vad factories + impls
    "BaseF0Tracker",
    "PyworldF0Tracker",
    "MockF0Tracker",
    "get_f0_tracker",
    "BaseVAD",
    "SileroVAD",
    "MockVAD",
    "get_vad",
    # feature / score / stage functions
    "extract_prosody_features",
    "compute_prosody_dsp_scores",
    "run_s2_stage",
    "write_s2_shard",
    "process_shard_s2",
]
