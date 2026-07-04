"""S1 - strict acoustic filtering (design doc §4 S1).

This stage combines one GPU model (Audiobox-Aesthetics -> pq/pc/ce/cu) with
CPU DSP metrics (SNR, clipping ratio, effective bandwidth via spectral
roll-off, loudness) into one :class:`S1AcousticsRow` per clip. DNSMOS was
retired from the stage (onnx CPU-serial and slow; pilot-measured marginal
rejection ~2% on top of aes_pq>=7.0) -- S0's metadata gate keeps Emilia's own
dnsmos >= 3.2, and the ``dnsmos_*`` row columns remain (NULL) for schema
stability with data produced by earlier runs.

Design conventions honored here:
  * The GPU models are obtained through :func:`emilia_pipeline.common.models.get_model`
    so unit tests run with deterministic mocks (zero GPU). CPU DSP is pure numpy.
  * Metrics are stored, never used to hard-drop a row. Pass/reject is a *query
    condition* -- exposed as the standalone :func:`s1_pass` predicate (and
    mirrored into the advisory ``passed`` / ``reject_reason`` columns for
    convenience). Threshold changes = re-run :func:`s1_pass` over stored
    parquet, not the pipeline.
  * Short-circuit rule: the *caller* (the fusion worker) skips S2/S3 when
    :func:`s1_pass` is False. S1's own metrics are computed in full for ROC
    calibration.

Public surface:
  * :func:`compute_cpu_metrics` -- SNR / clipping / bandwidth for one clip.
  * :func:`get_s1_models` / :class:`S1ModelBundle` -- lazy model handle.
  * :func:`compute_s1_rows` -- batch: list of decoded clips + handle -> rows.
  * :func:`s1_pass` / :func:`s1_reject_reason` -- the pass predicate.
  * :func:`write_s1_rows` -- atomic parquet write for one shard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ..common import audio as audio_utils
from ..common.config import Config
from ..common.contracts import S1AcousticsRow
from ..common.io_utils import atomic_write_parquet, parquet_glob
from ..common.models import MODEL_AESTHETICS, BaseAudioModel, get_model

# ---------------------------------------------------------------------------
# CPU DSP tuning constants (frame geometry for SNR / bandwidth estimation).
# ---------------------------------------------------------------------------

# STFT frame for the Welch-style power spectrum used by both SNR and bandwidth.
_FRAME_LEN = 1024
_FRAME_HOP = 512
_EPS = 1e-10

# Fraction of cumulative spectral energy that defines the effective bandwidth
# (spectral roll-off). A high value is deliberate: it locates where real signal
# energy actually stops, exposing fake "24k" clips upsampled from a lower rate
# (their spectrum is empty above the true Nyquist). Design §4: 防上采样假 24k.
DEFAULT_ROLLOFF_PERCENT = 0.99

# Percentiles of per-frame power used by the blind energy SNR estimator.
_SNR_NOISE_PCTILE = 10.0
_SNR_SPEECH_PCTILE = 95.0

# Clip type accepted by the batch entry point: (clip_id, samples, sample_rate).
Clip = tuple[str, np.ndarray, int]
# A row-like object the pass predicate accepts.
RowLike = Union[S1AcousticsRow, Mapping[str, Any]]


# ---------------------------------------------------------------------------
# CPU metrics
# ---------------------------------------------------------------------------


def _power_spectrum(arr: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs_hz, mean_power)`` via a Hann-windowed Welch average.

    Frames of :data:`_FRAME_LEN` with hop :data:`_FRAME_HOP` are Hann-windowed,
    rFFT'd, squared and averaged. Signals shorter than one frame use a single
    zero-padded frame so the estimator degrades gracefully rather than raising.

    Args:
        arr: float32 mono samples.
        sr: Sample rate in Hz.

    Returns:
        ``(freqs, power)`` where ``freqs`` are the rFFT bin centers in Hz and
        ``power`` is the (real, non-negative) mean power per bin.
    """
    x = np.asarray(arr, dtype=np.float64)
    n = x.size
    nfft = min(_FRAME_LEN, n) if n >= 1 else _FRAME_LEN
    nfft = max(nfft, 16)
    if n < nfft:
        x = np.pad(x, (0, nfft - n))
        n = x.size
    window = np.hanning(nfft)
    win_norm = np.sum(window**2) + _EPS

    starts = range(0, max(1, n - nfft + 1), _FRAME_HOP)
    acc = None
    count = 0
    for start in starts:
        frame = x[start : start + nfft]
        if frame.size < nfft:
            frame = np.pad(frame, (0, nfft - frame.size))
        spec = np.fft.rfft(frame * window)
        power = (np.abs(spec) ** 2) / win_norm
        acc = power if acc is None else acc + power
        count += 1
    mean_power = (acc / count) if acc is not None else np.zeros(nfft // 2 + 1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(sr))
    return freqs, mean_power


def bandwidth_hz(
    arr: np.ndarray, sr: int, roll_percent: float = DEFAULT_ROLLOFF_PERCENT
) -> float:
    """Effective bandwidth via spectral roll-off (design §4).

    The roll-off frequency is the lowest frequency below which ``roll_percent``
    of the total spectral energy is contained. A genuine 24 kHz recording keeps
    energy well above 8 kHz; a clip upsampled from 8/16 kHz rolls off early,
    which this metric surfaces.

    Args:
        arr: float32 mono samples.
        sr: Sample rate in Hz.
        roll_percent: Cumulative-energy fraction defining the cutoff.

    Returns:
        Roll-off frequency in Hz (0.0 for silent/empty input).
    """
    if arr is None or np.asarray(arr).size == 0:
        return 0.0
    freqs, power = _power_spectrum(arr, sr)
    total = float(np.sum(power))
    if total <= _EPS:
        return 0.0
    cumulative = np.cumsum(power)
    threshold = roll_percent * total
    idx = int(np.searchsorted(cumulative, threshold))
    idx = min(idx, freqs.size - 1)
    return float(freqs[idx])


def snr_db(arr: np.ndarray, sr: int) -> float:
    """Blind SNR estimate in dB (energy-percentile method).

    A framewise power estimate is split into a noise floor (low percentile) and
    an active-speech level (high percentile); SNR is their log-ratio. This is
    the cheap CPU-side alternative to WADA-SNR named in design §4 and needs no
    VAD or clean-reference. Robust to the synthetic voiced/silent test signals.

    Args:
        arr: float32 mono samples.
        sr: Sample rate in Hz (used only for framing consistency).

    Returns:
        Estimated SNR in dB. Returns 0.0 for empty input and is clamped to a
        finite range for degenerate (pure tone / pure silence) signals.
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.size == 0:
        return 0.0
    nfft = min(_FRAME_LEN, x.size)
    nfft = max(nfft, 16)
    hop = max(1, nfft // 2)
    powers = []
    for start in range(0, max(1, x.size - nfft + 1), hop):
        frame = x[start : start + nfft]
        if frame.size < nfft:
            frame = np.pad(frame, (0, nfft - frame.size))
        powers.append(float(np.mean(frame**2)))
    if not powers:
        powers = [float(np.mean(x**2))]
    parr = np.asarray(powers, dtype=np.float64)
    noise_p = float(np.percentile(parr, _SNR_NOISE_PCTILE))
    speech_p = float(np.percentile(parr, _SNR_SPEECH_PCTILE))
    noise_p = max(noise_p, _EPS)
    signal_p = max(speech_p - noise_p, _EPS)
    value = 10.0 * np.log10(signal_p / noise_p)
    # Clamp to a sane finite range (silence / pure tone can blow up the ratio).
    return float(np.clip(value, -20.0, 100.0))


def compute_cpu_metrics(arr: np.ndarray, sr: int) -> dict[str, float]:
    """Compute the CPU-side S1 metrics for one clip.

    Args:
        arr: float32 mono samples at native sample rate.
        sr: Sample rate in Hz.

    Returns:
        Dict with ``snr_db``, ``clipping_ratio``, ``bandwidth_hz`` and
        ``loudness_lufs``. Loudness is computed here (once, at decode) so it can
        be threaded through to the published §7 audio block without a second
        decode pass.
    """
    return {
        "snr_db": snr_db(arr, sr),
        "clipping_ratio": float(audio_utils.clipping_ratio(arr)),
        "bandwidth_hz": bandwidth_hz(arr, sr),
        "loudness_lufs": float(audio_utils.loudness_lufs(arr, sr)),
    }


# ---------------------------------------------------------------------------
# Model handle (lazy; mock-aware via the shared factory)
# ---------------------------------------------------------------------------


@dataclass
class S1ModelBundle:
    """The GPU model S1 needs, obtained via the mock-aware factory.

    Attributes:
        aesthetics: Audiobox-Aesthetics model (pq / pc / ce / cu).
    """

    aesthetics: BaseAudioModel

    @property
    def is_mock(self) -> bool:
        """True when the underlying model is a mock (test / key-less runs)."""
        return bool(self.aesthetics.is_mock)

    def close(self) -> None:
        """Release model resources."""
        self.aesthetics.close()


def get_s1_models(config: Config) -> S1ModelBundle:
    """Build the S1 model bundle (real or mock per ``config``/env).

    Args:
        config: Pipeline config. ``config.runtime.use_mocks`` (or absent
            GPU/weights) forces deterministic mocks.

    Returns:
        A :class:`S1ModelBundle`.
    """
    return S1ModelBundle(aesthetics=get_model(MODEL_AESTHETICS, config))


# ---------------------------------------------------------------------------
# Pass predicate (the "judgment-out" query condition)
# ---------------------------------------------------------------------------


def _get(row: RowLike, field: str) -> Any:
    """Read ``field`` from an :class:`S1AcousticsRow` or a plain mapping."""
    if isinstance(row, S1AcousticsRow):
        return getattr(row, field)
    return row[field]


def s1_reject_reason(row: RowLike, config: Config) -> str | None:
    """Return a ``;``-joined list of failed S1 gates, or None if the row passes.

    Gates (design §4 S1, thresholds from ``config.s1``):
      * ``aes_pq  >= min_aes_pq``      (main gate; source-separation artifacts)
      * ``aes_pc  <= max_aes_pc``      (low complexity = clean single speaker)
      * ``snr_db  >= min_snr_db``
      * ``clipping_ratio <= max_clipping_ratio``
      * ``bandwidth_hz   >= min_bandwidth_hz``

    Args:
        row: An S1 row (model or mapping) carrying the stored metrics.
        config: Pipeline config supplying ``s1`` thresholds.

    Returns:
        None when every gate passes; otherwise a compact reason string, e.g.
        ``"aes_pq<7.0;bandwidth_hz<8000.0"``.
    """
    s1 = config.s1
    reasons: list[str] = []
    if float(_get(row, "aes_pq")) < s1.min_aes_pq:
        reasons.append(f"aes_pq<{s1.min_aes_pq}")
    if float(_get(row, "aes_pc")) > s1.max_aes_pc:
        reasons.append(f"aes_pc>{s1.max_aes_pc}")
    if float(_get(row, "snr_db")) < s1.min_snr_db:
        reasons.append(f"snr_db<{s1.min_snr_db}")
    if float(_get(row, "clipping_ratio")) > s1.max_clipping_ratio:
        reasons.append(f"clipping_ratio>{s1.max_clipping_ratio}")
    if float(_get(row, "bandwidth_hz")) < s1.min_bandwidth_hz:
        reasons.append(f"bandwidth_hz<{s1.min_bandwidth_hz}")
    return ";".join(reasons) if reasons else None


def s1_pass(row: RowLike, config: Config) -> bool:
    """Return whether a clip passes every strict S1 acoustic gate.

    Args:
        row: An S1 row (model or mapping) carrying the stored metrics.
        config: Pipeline config supplying ``s1`` thresholds.

    Returns:
        True iff no gate is violated.
    """
    return s1_reject_reason(row, config) is None


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def compute_s1_rows(
    clips: Sequence[Clip],
    models: S1ModelBundle,
    config: Config,
    *,
    shard: str,
    cpu_metrics: Optional[Sequence[dict[str, float]]] = None,
) -> list[S1AcousticsRow]:
    """Compute S1 rows for a batch of decoded clips.

    Aesthetics runs chunked over the whole batch; CPU metrics are computed (or
    threaded through) per clip. The ``dnsmos_*`` columns always store ``None``
    (stage retired); every metric is stored on the row and the advisory
    ``passed`` / ``reject_reason`` columns are filled from :func:`s1_pass`.
    No row is dropped here.

    Args:
        clips: Sequence of ``(clip_id, samples, sample_rate)`` tuples. Samples
            are float32 mono at their native rate.
        models: The :class:`S1ModelBundle` handle.
        config: Pipeline config (thresholds + runtime).
        shard: Source shard token, stored on every row and used for the part
            filename by :func:`write_s1_rows`.
        cpu_metrics: Optional pre-computed CPU metrics per clip (aligned with
            ``clips``), as returned by :func:`compute_cpu_metrics`. The fusion
            worker's CPU pool already computes these during decode, so passing
            them here avoids recomputing SNR / bandwidth / loudness a second
            time. When ``None`` they are computed inline.

    Returns:
        One :class:`S1AcousticsRow` per input clip, in input order.

    Raises:
        ValueError: If ``cpu_metrics`` is given but its length != ``len(clips)``.
    """
    if not clips:
        return []
    if cpu_metrics is not None and len(cpu_metrics) != len(clips):
        raise ValueError("cpu_metrics must align 1:1 with clips")

    batch: list[tuple[np.ndarray, int]] = [(arr, sr) for _, arr, sr in clips]
    aes_out = models.aesthetics.predict(batch)

    rows: list[S1AcousticsRow] = []
    for i, ((clip_id, arr, sr), aes) in enumerate(zip(clips, aes_out)):
        cpu = cpu_metrics[i] if cpu_metrics is not None else compute_cpu_metrics(arr, sr)
        metrics: dict[str, Any] = {
            "clip_id": clip_id,
            "shard": shard,
            "aes_pq": float(aes["aes_pq"]),
            "aes_pc": float(aes["aes_pc"]),
            "aes_ce": float(aes["aes_ce"]),
            "aes_cu": float(aes["aes_cu"]),
            "dnsmos_sig": None,
            "dnsmos_bak": None,
            "dnsmos_ovrl": None,
            "snr_db": cpu["snr_db"],
            "clipping_ratio": cpu["clipping_ratio"],
            "bandwidth_hz": cpu["bandwidth_hz"],
            "loudness_lufs": float(cpu.get("loudness_lufs", 0.0)),
        }
        reason = s1_reject_reason(metrics, config)
        rows.append(
            S1AcousticsRow(**metrics, passed=(reason is None), reject_reason=reason)
        )
    return rows


# ---------------------------------------------------------------------------
# Atomic shard write
# ---------------------------------------------------------------------------


def s1_part_path(config: Config, shard: str) -> Path:
    """Return the canonical S1 parquet path for ``shard`` (``part-{shard}.parquet``)."""
    return Path(config.paths.s1_acoustics) / f"part-{shard}.parquet"


def write_s1_rows(
    rows: Sequence[S1AcousticsRow], config: Config, shard: str
) -> Path:
    """Atomically write S1 rows to ``stage/s1_acoustics/part-{shard}.parquet``.

    Uses the shared atomic writer (``*.tmp`` then rename). The caller creates
    the done marker only after this returns (design §3 write discipline).

    Args:
        rows: The S1 rows for one shard.
        config: Pipeline config (for the output path).
        shard: Source shard token.

    Returns:
        The parquet path written.
    """
    payload = [r.model_dump() for r in rows]
    return atomic_write_parquet(payload, s1_part_path(config, shard))


def s1_parquet_glob(config: Config) -> str:
    """Return the DuckDB glob over all S1 part files (excludes ``*.tmp``)."""
    return parquet_glob(config.paths.s1_acoustics)


__all__ = [
    "DEFAULT_ROLLOFF_PERCENT",
    "Clip",
    "RowLike",
    "bandwidth_hz",
    "snr_db",
    "compute_cpu_metrics",
    "S1ModelBundle",
    "get_s1_models",
    "s1_reject_reason",
    "s1_pass",
    "compute_s1_rows",
    "s1_part_path",
    "write_s1_rows",
    "s1_parquet_glob",
]
