"""Synthetic audio + fixture generation (backbone of the mock test tier).

Produces controllable test signals (sine / noise / multi-speaker-ish mixes with
tunable f0, energy, pauses, duration) and assembles a tiny Emilia-style
WebDataset tar shard with sidecar JSON metadata. No real data, GPU or API keys
required, so the whole test suite can run offline.

Emilia member convention per clip inside the tar:
  * ``{key}.flac`` -- audio
  * ``{key}.json`` -- Emilia-style metadata (id, text, speaker, language,
    duration, dnsmos)
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_SR = 24000


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------


def sine_tone(
    freq_hz: float,
    duration_s: float,
    sr: int = DEFAULT_SR,
    amplitude: float = 0.3,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a sine tone (a crude voiced-frame stand-in with fixed f0)."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return sig.astype(np.float32)


def noise(
    duration_s: float,
    sr: int = DEFAULT_SR,
    amplitude: float = 0.05,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Generate white noise (background / unvoiced stand-in)."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sr))
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def _apply_energy_contour(sig: np.ndarray, energy_var: float, seed: int) -> np.ndarray:
    """Multiply by a slow random gain contour to inject energy variation."""
    if energy_var <= 0:
        return sig
    rng = np.random.default_rng(seed)
    n_ctrl = max(2, len(sig) // 4800)
    ctrl = 1.0 + energy_var * (rng.random(n_ctrl) - 0.5) * 2.0
    contour = np.interp(
        np.linspace(0, n_ctrl - 1, len(sig)), np.arange(n_ctrl), ctrl
    ).astype(np.float32)
    return (sig * contour).astype(np.float32)


@dataclass
class SynthSpec:
    """Controllable parameters for one synthetic voice-like clip."""

    duration_s: float = 6.0
    sr: int = DEFAULT_SR
    f0_hz: float = 180.0
    f0_glide_hz: float = 40.0  # peak-to-peak f0 movement (prosody dynamism)
    amplitude: float = 0.3
    energy_var: float = 0.3
    noise_amp: float = 0.02
    pauses: Sequence[tuple[float, float]] = field(default_factory=tuple)  # (start,end) s
    seed: int = 0


def synth_voice(spec: SynthSpec) -> np.ndarray:
    """Synthesize a single voice-like clip from a :class:`SynthSpec`.

    Builds a frequency-modulated sine (to emulate F0 movement), applies an
    energy contour, inserts silent pauses and adds low-level noise. The result
    is float32 mono clipped to ``[-1, 1]``.
    """
    sr = spec.sr
    n = int(round(spec.duration_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    # F0 glide via a slow sinusoidal modulation of instantaneous frequency.
    f0 = spec.f0_hz + 0.5 * spec.f0_glide_hz * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = spec.amplitude * np.sin(phase).astype(np.float32)
    sig = _apply_energy_contour(sig, spec.energy_var, spec.seed)
    sig = sig + noise(spec.duration_s, sr, spec.noise_amp, seed=spec.seed + 1)[: len(sig)]
    for start_s, end_s in spec.pauses:
        i0 = max(0, int(round(start_s * sr)))
        i1 = min(n, int(round(end_s * sr)))
        sig[i0:i1] = 0.0
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def synth_multispeaker(
    spec_a: SynthSpec, spec_b: SynthSpec, overlap_frac: float = 0.4
) -> np.ndarray:
    """Mix two voices with a trailing overlap region (overlap/intrusion stand-in).

    Speaker A runs full length; speaker B is added over the final
    ``overlap_frac`` of the clip. Useful for exercising S3 overlap verdicts.
    """
    a = synth_voice(spec_a)
    b = synth_voice(spec_b)
    n = len(a)
    out = a.copy()
    overlap_n = int(round(n * overlap_frac))
    if overlap_n > 0:
        seg = b[:overlap_n] if len(b) >= overlap_n else np.pad(b, (0, overlap_n - len(b)))
        out[n - overlap_n :] += 0.8 * seg
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Encoding + Emilia-style metadata
# ---------------------------------------------------------------------------


def encode_flac(arr: np.ndarray, sr: int = DEFAULT_SR) -> bytes:
    """Encode a float32 mono signal to FLAC bytes."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(arr, dtype=np.float32), sr, format="FLAC")
    return buf.getvalue()


def emilia_metadata(
    *,
    clip_key: str,
    text: str,
    speaker: str,
    duration_s: float,
    language: str = "zh",
    dnsmos: float = 3.5,
) -> dict:
    """Build an Emilia-style metadata dict for one clip's sidecar JSON."""
    return {
        "id": clip_key,
        "text": text,
        "speaker": speaker,
        "language": language,
        "duration": round(float(duration_s), 3),
        "dnsmos": round(float(dnsmos), 3),
    }


@dataclass
class SynthClip:
    """One synthetic clip: audio + its Emilia-style metadata."""

    key: str
    audio: np.ndarray
    sr: int
    meta: dict


def make_synth_clip(
    key: str,
    *,
    spec: Optional[SynthSpec] = None,
    text: str = "这是一段合成的测试语音。",
    speaker: str = "ZH_TEST_S00",
    language: str = "zh",
    dnsmos: float = 3.5,
) -> SynthClip:
    """Create a single synthetic clip with matching Emilia metadata."""
    spec = spec or SynthSpec()
    audio = synth_voice(spec)
    meta = emilia_metadata(
        clip_key=key,
        text=text,
        speaker=speaker,
        duration_s=len(audio) / spec.sr,
        language=language,
        dnsmos=dnsmos,
    )
    return SynthClip(key=key, audio=audio, sr=spec.sr, meta=meta)


def write_webdataset_shard(
    clips: Sequence[SynthClip], path: str | os.PathLike[str]
) -> Path:
    """Write clips to a WebDataset-style tar shard (``{key}.flac`` + ``{key}.json``).

    Members are appended in key order so the shard reads sequentially, matching
    Emilia's layout and the Phase-1 ordered-read assumption.

    Args:
        clips: The synthetic clips to pack.
        path: Destination ``.tar`` path.

    Returns:
        The tar path written.
    """
    tar_path = Path(path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        for clip in clips:
            flac = encode_flac(clip.audio, clip.sr)
            _add_member(tar, f"{clip.key}.flac", flac)
            meta_bytes = json.dumps(clip.meta, ensure_ascii=False).encode("utf-8")
            _add_member(tar, f"{clip.key}.json", meta_bytes)
    return tar_path


def _add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Append an in-memory bytes member to an open tar."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def build_synthetic_shard(
    path: str | os.PathLike[str],
    *,
    n_clips: int = 8,
    sr: int = DEFAULT_SR,
    seed: int = 0,
    shard_name: str = "00000",
) -> tuple[Path, list[SynthClip]]:
    """Build a small varied synthetic shard for fixtures.

    Generates a spread of clips: expressive single voices, a flat reading, a
    too-short clip, a too-long clip, and a multi-speaker overlap clip, so the
    fixture exercises S0-S3 verdict branches. Returns the tar path and the clip
    list (so tests can assert against ground truth).

    Args:
        path: Destination ``.tar`` path.
        n_clips: Number of single-voice clips before the special cases.
        sr: Sample rate.
        seed: Base RNG seed for reproducibility.
        shard_name: Shard token embedded in clip keys.

    Returns:
        ``(tar_path, clips)``.
    """
    rng = np.random.default_rng(seed)
    clips: list[SynthClip] = []
    speakers = [f"ZH_B{shard_name}_S{i:02d}" for i in range(3)]

    for i in range(n_clips):
        spk = speakers[i % len(speakers)]
        key = f"emilia_zh_{shard_name}_{i:04d}"
        spec = SynthSpec(
            duration_s=float(rng.uniform(4.0, 12.0)),
            sr=sr,
            f0_hz=float(rng.uniform(120.0, 260.0)),
            f0_glide_hz=float(rng.uniform(5.0, 80.0)),
            energy_var=float(rng.uniform(0.05, 0.5)),
            pauses=((2.0, 2.3),) if i % 2 == 0 else (),
            seed=seed + i,
        )
        clips.append(
            make_synth_clip(
                key,
                spec=spec,
                text="合成测试语音内容，用于单元测试。",
                speaker=spk,
                dnsmos=float(rng.uniform(3.0, 4.2)),
            )
        )

    # Edge cases exercising S0 / S3 branches.
    clips.append(
        make_synth_clip(
            f"emilia_zh_{shard_name}_short",
            spec=SynthSpec(duration_s=1.5, sr=sr, seed=seed + 100),
            text="太短",
            speaker=speakers[0],
        )
    )
    clips.append(
        make_synth_clip(
            f"emilia_zh_{shard_name}_long",
            spec=SynthSpec(duration_s=22.0, sr=sr, seed=seed + 101),
            text="这是一段超过二十秒的合成语音用于测试时长上限。",
            speaker=speakers[1],
        )
    )
    overlap_audio = synth_multispeaker(
        SynthSpec(duration_s=8.0, sr=sr, f0_hz=180.0, seed=seed + 200),
        SynthSpec(duration_s=8.0, sr=sr, f0_hz=95.0, seed=seed + 201),
        overlap_frac=0.4,
    )
    ov_key = f"emilia_zh_{shard_name}_overlap"
    clips.append(
        SynthClip(
            key=ov_key,
            audio=overlap_audio,
            sr=sr,
            meta=emilia_metadata(
                clip_key=ov_key,
                text="两个人同时说话的重叠片段。",
                speaker=speakers[2],
                duration_s=len(overlap_audio) / sr,
                dnsmos=3.6,
            ),
        )
    )

    tar_path = write_webdataset_shard(clips, path)
    return tar_path, clips


__all__ = [
    "DEFAULT_SR",
    "sine_tone",
    "noise",
    "SynthSpec",
    "synth_voice",
    "synth_multispeaker",
    "encode_flac",
    "emilia_metadata",
    "SynthClip",
    "make_synth_clip",
    "write_webdataset_shard",
    "build_synthetic_shard",
]
