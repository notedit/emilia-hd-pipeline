"""Audio decode / resample / loudness helpers.

Keeps ``soundfile`` and ``librosa`` behind functions so importing this module is
cheap and stage code depends on a stable surface. All decoders return float32
mono in ``[-1, 1]`` plus the native sample rate.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _to_mono_float32(arr: np.ndarray) -> np.ndarray:
    """Collapse to mono and cast to float32 (soundfile returns float64/int)."""
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes (flac/wav/ogg) to ``(float32 mono, sr)``.

    Args:
        data: Raw encoded audio bytes.

    Returns:
        ``(samples, sample_rate)`` with samples float32 mono in ``[-1, 1]``.
    """
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(data)) as f:
        sr = f.samplerate
        arr = f.read(dtype="float32", always_2d=False)
    return _to_mono_float32(arr), sr


def decode_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode an audio file from disk to ``(float32 mono, sr)``."""
    import soundfile as sf

    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return _to_mono_float32(arr), sr


def decode_tar_member(
    tar: tarfile.TarFile, member_name: str
) -> tuple[np.ndarray, int]:
    """Decode one audio member from an open WebDataset/tar archive.

    Args:
        tar: An open :class:`tarfile.TarFile`.
        member_name: The member (e.g. ``"clip123.flac"``) to extract.

    Returns:
        ``(float32 mono samples, sample_rate)``.

    Raises:
        KeyError: If the member is absent from the archive.
    """
    handle = tar.extractfile(member_name)
    if handle is None:
        raise KeyError(f"tar member not found or not a file: {member_name}")
    return decode_bytes(handle.read())


def resample(arr: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a mono float32 signal from ``sr_in`` to ``sr_out``.

    No-op (returns the input) when the rates match. Uses librosa's high-quality
    resampler; kept behind this function so callers never import librosa.
    """
    if sr_in == sr_out:
        return np.ascontiguousarray(arr, dtype=np.float32)
    import librosa

    out = librosa.resample(
        np.asarray(arr, dtype=np.float32), orig_sr=sr_in, target_sr=sr_out
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def loudness_lufs(arr: np.ndarray, sr: int) -> float:
    """Estimate integrated loudness in LUFS.

    Uses ``pyloudnorm`` when available (ITU-R BS.1770); otherwise falls back to
    a dBFS-RMS approximation so this never hard-fails in a mock/test env.

    Args:
        arr: float32 mono samples.
        sr: Sample rate in Hz.

    Returns:
        Integrated loudness estimate in LUFS (float).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return float("-inf")
    try:
        import pyloudnorm as pyln  # optional dependency

        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(arr.astype(np.float64)))
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float64)))) + 1e-12)
        return 20.0 * np.log10(rms)


def peak_dbfs(arr: np.ndarray) -> float:
    """Return the peak level in dBFS (0 dBFS = full scale)."""
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    return 20.0 * np.log10(peak + 1e-12)


def clipping_ratio(arr: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at/above ``threshold`` absolute amplitude."""
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr) >= threshold))


def duration_s(arr: np.ndarray, sr: int) -> float:
    """Duration of a signal in seconds."""
    return float(len(arr)) / float(sr) if sr else 0.0


def trim_segment(
    arr: np.ndarray, sr: int, start_s: float, end_s: Optional[float] = None
) -> np.ndarray:
    """Return the ``[start_s, end_s)`` slice of a signal (used by S3 trimming)."""
    start = max(0, int(round(start_s * sr)))
    stop = len(arr) if end_s is None else min(len(arr), int(round(end_s * sr)))
    return np.ascontiguousarray(arr[start:stop], dtype=np.float32)


def encode_audio(arr: np.ndarray, sr: int, fmt: str = "FLAC", **sf_kwargs: Any) -> bytes:
    """Encode a float32 mono signal to compressed audio bytes.

    Used by repack to re-emit trimmed clips (``intruded_trimmed``) whose audio
    was sliced to its kept span. Non-trimmed clips are copied verbatim and never
    hit this path (no needless re-encode).

    Args:
        arr: float32 mono samples in ``[-1, 1]``.
        sr: Sample rate in Hz.
        fmt: soundfile format string (default ``"FLAC"``).
        **sf_kwargs: Extra ``soundfile.write`` options (e.g. ``bitrate_mode`` /
            ``compression_level`` for MP3 quality).

    Returns:
        Encoded audio bytes.
    """
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(arr, dtype=np.float32), sr, format=fmt, **sf_kwargs)
    return buf.getvalue()


__all__ = [
    "decode_bytes",
    "decode_file",
    "decode_tar_member",
    "resample",
    "loudness_lufs",
    "peak_dbfs",
    "clipping_ratio",
    "duration_s",
    "trim_segment",
    "encode_audio",
]
