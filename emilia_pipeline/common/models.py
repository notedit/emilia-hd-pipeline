"""Mock-aware factories for GPU models and the S4 cloud API client.

Every GPU model and every network dependency sits behind a lazy loader with a
deterministic MOCK fallback so unit tests run with zero GPU / key / data
(project convention). The mock output is seeded by a hash of the clip content,
so a given input always yields the same numbers across runs and processes.

Selection logic:
  * ``config.runtime.use_mocks == True``  -> always mock.
  * otherwise try to build the real impl; if weights/GPU/deps are unavailable,
    fall back to the mock and record the reason.

Public surface:
  * :func:`get_model(name, config)` -> :class:`BaseAudioModel`
  * :func:`get_s4_client(config)`   -> :class:`BaseS4Client`
  * Interfaces: ``BaseAudioModel.predict(batch) -> list[dict]`` and
    ``BaseS4Client.label(...) -> S4GuidedJSON``.
"""

from __future__ import annotations

import abc
import hashlib
import os
from typing import Any, Optional, Sequence

import numpy as np

from .config import Config
from .contracts import (
    Accent,
    ContextLabel,
    Defect,
    EmotionLabel,
    EmotionPrimary,
    GenderPred,
    LanguageLabel,
    Paralinguistic,
    ProsodyLabel,
    Register,
    Rhythm,
    S4GuidedJSON,
    Scenario,
    SpeakerVerdict,
    SpeakingStyle,
    TextVerdict,
)

# Canonical model names understood by :func:`get_model`.
MODEL_AESTHETICS = "aesthetics"
MODEL_DNSMOS = "dnsmos"
MODEL_CAMPP = "campplus"


# ---------------------------------------------------------------------------
# Deterministic hashing helpers
# ---------------------------------------------------------------------------


def content_hash(arr: np.ndarray | bytes | str) -> str:
    """Return a stable hex digest for arbitrary clip content.

    Used to seed deterministic mock outputs so the same input always produces
    the same fake metrics across runs, processes and machines.
    """
    if isinstance(arr, str):
        payload = arr.encode("utf-8")
    elif isinstance(arr, bytes):
        payload = arr
    else:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        payload = a.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _seeded_rng(*parts: Any) -> np.random.Generator:
    """Build a numpy Generator seeded by the sha256 of the joined parts."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(seed)


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * rng.random())


# ---------------------------------------------------------------------------
# Audio-model interface
# ---------------------------------------------------------------------------


class BaseAudioModel(abc.ABC):
    """Interface every GPU audio model exposes.

    Implementations take a batch of ``(samples, sr)`` clips and return one dict
    of metrics per clip, in input order. Field names match the corresponding
    per-stage row schema in :mod:`emilia_pipeline.common.contracts`.
    """

    name: str
    is_mock: bool = False

    @abc.abstractmethod
    def predict(
        self, batch: Sequence[tuple[np.ndarray, int]]
    ) -> list[dict[str, Any]]:
        """Run inference on a batch of ``(samples, sample_rate)`` clips."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any GPU / native resources. Default is a no-op."""


# ----- Mock implementations -------------------------------------------------


class MockAestheticsModel(BaseAudioModel):
    """Deterministic Audiobox-Aesthetics mock (pq / pc / ce / cu)."""

    name = MODEL_AESTHETICS
    is_mock = True

    def predict(self, batch: Sequence[tuple[np.ndarray, int]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for arr, sr in batch:
            rng = _seeded_rng("aesthetics", content_hash(arr))
            out.append(
                {
                    "aes_pq": round(_uniform(rng, 4.0, 9.0), 3),
                    "aes_pc": round(_uniform(rng, 1.0, 5.0), 3),
                    "aes_ce": round(_uniform(rng, 4.0, 9.0), 3),
                    "aes_cu": round(_uniform(rng, 4.0, 9.0), 3),
                }
            )
        return out


class MockDnsmosModel(BaseAudioModel):
    """Deterministic DNSMOS P.835 mock (sig / bak / ovrl)."""

    name = MODEL_DNSMOS
    is_mock = True

    def predict(self, batch: Sequence[tuple[np.ndarray, int]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for arr, sr in batch:
            rng = _seeded_rng("dnsmos", content_hash(arr))
            out.append(
                {
                    "dnsmos_sig": round(_uniform(rng, 2.5, 4.5), 3),
                    "dnsmos_bak": round(_uniform(rng, 2.5, 4.5), 3),
                    "dnsmos_ovrl": round(_uniform(rng, 2.5, 4.5), 3),
                }
            )
        return out


class MockCampPlusModel(BaseAudioModel):
    """Deterministic CAM++ mock: sliding-window embeddings + purity stats.

    Returns one dict per clip with a clip-level mean embedding (``embedding``,
    shape ``(embedding_dim,)`` fp16) and the S3 window statistics. The verdict
    is derived from the fake window cosines so it is internally consistent.
    """

    name = MODEL_CAMPP
    is_mock = True

    def __init__(self, embedding_dim: int = 192) -> None:
        self.embedding_dim = embedding_dim

    def predict(self, batch: Sequence[tuple[np.ndarray, int]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for arr, sr in batch:
            h = content_hash(arr)
            rng = _seeded_rng("campplus", h)
            emb = rng.standard_normal(self.embedding_dim).astype(np.float16)
            n_windows = int(rng.integers(4, 12))
            mean_win_cos = round(_uniform(rng, 0.70, 0.99), 3)
            min_win_cos = round(min(mean_win_cos, _uniform(rng, 0.50, mean_win_cos)), 3)
            f0_stability = round(_uniform(rng, 0.55, 0.99), 3)
            gender = GenderPred.FEMALE if rng.random() < 0.5 else GenderPred.MALE
            verdict = self._derive_verdict(mean_win_cos, min_win_cos, f0_stability)
            out.append(
                {
                    "embedding": emb,
                    "n_windows": n_windows,
                    "mean_win_cos": mean_win_cos,
                    "min_win_cos": min_win_cos,
                    "f0_stability": f0_stability,
                    "gender_pred": gender.value,
                    "verdict": verdict.value,
                    "intrusion_span_ms": None,
                }
            )
        return out

    @staticmethod
    def _derive_verdict(
        mean_cos: float, min_cos: float, f0_stab: float
    ) -> SpeakerVerdict:
        """Map fake cosine/F0 stats to a purity verdict (mirrors §4 S3b table)."""
        if min_cos < 0.60:
            return SpeakerVerdict.INTRUDED_TRIMMED
        if mean_cos < 0.80:
            return (
                SpeakerVerdict.OVERLAP_REJECTED
                if f0_stab < 0.60
                else SpeakerVerdict.DEGRADED_PASS
            )
        return SpeakerVerdict.SINGLE


# ----- Real implementation stubs (lazy; fall back to mock on failure) -------


class _RealModelUnavailable(RuntimeError):
    """Raised internally when a real model cannot be constructed."""


class RealAestheticsModel(BaseAudioModel):
    """Real Audiobox-Aesthetics model (pq / pc / ce / cu).

    Wraps ``audiobox_aesthetics``' ``AesPredictor``. The predictor resamples to
    16 kHz internally and returns the four axes CE / CU / PC / PQ per clip; we
    remap them to the ``aes_*`` keys the S1 stage expects. The model is loaded
    once from a local checkpoint and reused across the batch.
    """

    name = MODEL_AESTHETICS
    is_mock = False

    def __init__(self, checkpoint_pth: str) -> None:
        from audiobox_aesthetics.infer import initialize_predictor

        self._predictor = initialize_predictor(checkpoint_pth)

    def predict(self, batch: Sequence[tuple[np.ndarray, int]]) -> list[dict[str, Any]]:
        import torch

        items = []
        for arr, sr in batch:
            wav = torch.as_tensor(np.ascontiguousarray(arr, dtype=np.float32))
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # (C=1, T) as the predictor expects
            items.append({"path": wav, "sample_rate": int(sr)})
        raw = self._predictor.forward(items)
        out: list[dict[str, Any]] = []
        for r in raw:
            out.append(
                {
                    "aes_pq": float(r["PQ"]),
                    "aes_pc": float(r["PC"]),
                    "aes_ce": float(r["CE"]),
                    "aes_cu": float(r["CU"]),
                }
            )
        return out


class RealDnsmosModel(BaseAudioModel):
    """Real DNSMOS P.835 model (sig / bak / ovrl) via onnxruntime.

    Uses the ``sig_bak_ovr.onnx`` graph, whose input is a raw 16 kHz waveform
    window of exactly ``9.01 s`` (144160 samples) and whose output is the three
    P.835 scores directly. Audio is resampled to 16 kHz, split into 9.01 s
    windows (hop 1 s), scored per window, and averaged. Runs on the CPU execution
    provider by default (fast enough; avoids the onnxruntime-gpu version-match
    risk).
    """

    name = MODEL_DNSMOS
    is_mock = False

    _SR = 16000
    _INPUT_LENGTH = 9.01
    _WIN = 144160  # int(9.01 * 16000)

    def __init__(self, onnx_path: str, providers: Optional[list[str]] = None) -> None:
        import onnxruntime as ort

        self._sess = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name

    def predict(self, batch: Sequence[tuple[np.ndarray, int]]) -> list[dict[str, Any]]:
        import librosa

        out: list[dict[str, Any]] = []
        for arr, sr in batch:
            audio = np.ascontiguousarray(arr, dtype=np.float32)
            if sr != self._SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self._SR)
            if len(audio) < self._WIN:  # tile up to one full window
                audio = np.tile(audio, int(np.ceil(self._WIN / max(1, len(audio)))))
            hop = self._SR
            segs = []
            for start in range(0, max(1, len(audio) - self._WIN + 1), hop):
                seg = audio[start : start + self._WIN]
                if len(seg) == self._WIN:
                    segs.append(seg)
            if not segs:
                segs = [audio[: self._WIN]]
            feat = np.stack(segs).astype(np.float32)  # (n_win, 144160)
            res = self._sess.run(None, {self._input_name: feat})[0]  # (n_win, 3)
            sig, bak, ovr = res.mean(axis=0)
            # P.835 MOS scores are defined on [1, 5]; clamp model artifacts that
            # dip just outside on non-speech / degraded input.
            clamp = lambda v: float(min(5.0, max(1.0, v)))
            out.append(
                {
                    "dnsmos_sig": clamp(sig),
                    "dnsmos_bak": clamp(bak),
                    "dnsmos_ovrl": clamp(ovr),
                }
            )
        return out


class RealCampPlusModel(BaseAudioModel):
    """Real CAM++ speaker model via ModelScope (sliding-window purity geometry).

    Loads the ModelScope ``speech_campplus_sv_zh-cn_16k-common`` verification
    pipeline once. For each clip it extracts a per-window embedding sequence over
    ``window_s`` / ``window_overlap`` windows (design §4 S3a) at 16 kHz and
    returns the raw ``window_embeddings`` matrix so the S3 stage recomputes the
    cosine geometry (mean/min window cosine, low-cos spans -> intrusion) with its
    own centroid-free logic. Also returns the clip-level mean ``embedding``.
    """

    name = MODEL_CAMPP
    is_mock = False

    _SR = 16000

    def __init__(self, model_dir: str, window_s: float, window_overlap: float,
                 embedding_dim: int) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self._pipe = pipeline(task=Tasks.speaker_verification, model=model_dir)
        self.window_s = window_s
        self.window_overlap = window_overlap
        self.embedding_dim = embedding_dim

    def _embed(self, audio16k: np.ndarray) -> np.ndarray:
        """Return the CAM++ embedding for one 16 kHz mono segment, shape (dim,)."""
        r = self._pipe([audio16k], output_emb=True)
        return np.asarray(r["embs"], dtype=np.float32).reshape(-1)[: self.embedding_dim]

    def predict(self, batch: Sequence[tuple[np.ndarray, int]]) -> list[dict[str, Any]]:
        import librosa

        out: list[dict[str, Any]] = []
        win = max(1, int(round(self.window_s * self._SR)))
        hop = max(1, int(round(win * (1.0 - self.window_overlap))))
        for arr, sr in batch:
            audio = np.ascontiguousarray(arr, dtype=np.float32)
            if sr != self._SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self._SR)
            n = len(audio)
            # Frame into windows (mirrors s3_speaker.frame_windows keep-rule).
            spans: list[tuple[int, int]] = []
            if n <= win:
                spans = [(0, n)]
            else:
                start = 0
                while start < n:
                    stop = min(start + win, n)
                    if stop - start >= win // 2 or not spans:
                        spans.append((start, stop))
                    if stop >= n:
                        break
                    start += hop
            win_embs = []
            for a, b in spans:
                seg = audio[a:b]
                if len(seg) < self._SR // 2:  # pad very short tail to >=0.5s
                    seg = np.pad(seg, (0, self._SR // 2 - len(seg)))
                win_embs.append(self._embed(seg))
            win_mat = np.stack(win_embs).astype(np.float16)  # (n_windows, dim)
            mean_emb = win_mat.mean(axis=0).astype(np.float16)
            # Aggregate cosine geometry (also recomputed centroid-free by
            # s3_speaker; emitting it here keeps the model output a superset of
            # the mock and self-describing). cos of each window to the mean.
            wm = win_mat.astype(np.float32)
            center = wm.mean(axis=0)
            cnorm = float(np.linalg.norm(center)) + 1e-8
            wnorm = np.linalg.norm(wm, axis=1) + 1e-8
            cos = (wm @ center) / (wnorm * cnorm)
            out.append(
                {
                    "window_embeddings": win_mat,
                    "embedding": mean_emb,
                    "n_windows": int(win_mat.shape[0]),
                    "mean_win_cos": float(np.mean(cos)),
                    "min_win_cos": float(np.min(cos)),
                    "intrusion_span_ms": None,
                    "gender_pred": GenderPred.UNKNOWN.value,
                }
            )
        return out


def _try_build_real_model(name: str, config: Config) -> Optional[BaseAudioModel]:
    """Attempt to construct a real GPU/CPU model; return None if unavailable.

    Real construction requires the model's weights to be configured and present
    on disk. Any missing dependency, weight file, or construction error results
    in ``None`` so :func:`get_model` cleanly falls back to the deterministic mock
    (project convention: never hard-fail on a missing real model).
    """
    weights = {
        MODEL_AESTHETICS: config.models.aesthetics_weights,
        MODEL_DNSMOS: config.models.dnsmos_onnx,
        MODEL_CAMPP: config.models.campplus_weights,
    }.get(name)
    if not weights or not os.path.exists(str(weights)):
        return None
    try:
        if name == MODEL_AESTHETICS:
            return RealAestheticsModel(str(weights))
        if name == MODEL_DNSMOS:
            return RealDnsmosModel(str(weights))
        if name == MODEL_CAMPP:
            return RealCampPlusModel(
                str(weights),
                window_s=config.s3.window_s,
                window_overlap=config.s3.window_overlap,
                embedding_dim=config.s3.embedding_dim,
            )
    except Exception:  # missing dep / bad weight / load error -> fall back to mock
        return None
    return None


# ----- Factory --------------------------------------------------------------


def get_model(name: str, config: Config) -> BaseAudioModel:
    """Return a real or mock audio model for ``name``.

    Args:
        name: One of :data:`MODEL_AESTHETICS`, :data:`MODEL_DNSMOS`,
            :data:`MODEL_CAMPP`.
        config: Pipeline config. ``config.runtime.use_mocks`` forces a mock.

    Returns:
        A :class:`BaseAudioModel`. Falls back to the deterministic mock when
        ``use_mocks`` is set or the real weights/GPU are unavailable.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    name = name.lower()
    if name not in (MODEL_AESTHETICS, MODEL_DNSMOS, MODEL_CAMPP):
        raise ValueError(f"unknown model name: {name!r}")

    if not config.runtime.use_mocks:
        real = _try_build_real_model(name, config)
        if real is not None:
            return real

    if name == MODEL_AESTHETICS:
        return MockAestheticsModel()
    if name == MODEL_DNSMOS:
        return MockDnsmosModel()
    return MockCampPlusModel(embedding_dim=config.s3.embedding_dim)


# ---------------------------------------------------------------------------
# S4 cloud API client
# ---------------------------------------------------------------------------


class BaseS4Client(abc.ABC):
    """Interface for the S4 Qwen3-Omni labeling transport.

    Implementations accept a clip's audio + reference text and return a
    schema-valid :class:`S4GuidedJSON`. Real impls call the DashScope /
    OpenAI-compatible endpoint; the mock returns deterministic valid JSON.
    """

    is_mock: bool = False

    @abc.abstractmethod
    async def label(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        reference_text: str,
        clip_id: str = "",
    ) -> S4GuidedJSON:
        """Produce structured labels for one clip. Async for concurrency."""

    async def close(self) -> None:  # pragma: no cover - default no-op
        """Release any client resources (session, connector)."""


class MockS4Client(BaseS4Client):
    """Deterministic S4 client: schema-valid JSON seeded by audio+text hash."""

    is_mock = True

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = config.s4.model
        self.prompt_version = config.s4.prompt_version

    async def label(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        reference_text: str,
        clip_id: str = "",
    ) -> S4GuidedJSON:
        rng = _seeded_rng("s4", content_hash(audio), reference_text)
        emotions = list(EmotionPrimary)
        styles = list(SpeakingStyle)
        primary = emotions[int(rng.integers(0, len(emotions)))]
        return S4GuidedJSON(
            text_verdict=TextVerdict.MATCH if rng.random() < 0.7 else TextVerdict.FIXABLE,
            text_fixed=reference_text,
            text_punctuated=reference_text,
            emotion=EmotionLabel(
                primary=primary,
                secondary=None,
                intensity=int(rng.integers(1, 6)),
                confidence=round(_uniform(rng, 0.5, 1.0), 3),
            ),
            prosody=ProsodyLabel(
                expressiveness=int(rng.integers(1, 6)),
                speaking_style=styles[int(rng.integers(0, len(styles)))],
                rhythm=list(Rhythm)[int(rng.integers(0, len(Rhythm)))],
                prominent_stress=bool(rng.random() < 0.5),
            ),
            context=ContextLabel(
                scenario=list(Scenario)[int(rng.integers(0, len(Scenario)))],
                register=list(Register)[int(rng.integers(0, len(Register)))],
                summary="mock summary",
            ),
            language=LanguageLabel(
                primary="zh",
                code_switch=False,
                accent=list(Accent)[int(rng.integers(0, len(Accent)))],
            ),
            paralinguistic=(
                [Paralinguistic.BREATH_PROMINENT] if rng.random() < 0.3 else []
            ),
            defects=[],
            usable=bool(rng.random() < 0.9),
        )


class OmniApiClient(BaseS4Client):
    """Real S4 client against an OpenAI-compatible Qwen3-Omni endpoint.

    Default target is the internal **Venus LLM proxy** (``provider="venus"``),
    which uses a proprietary ``venus_multimodal_url`` audio content type; set
    ``provider="openai"`` for the standard ``input_audio`` content type. The
    actual request/response plumbing (base64 audio, message construction via the
    AsyncOpenAI SDK, retry/backoff) is owned by the Phase-2 stage module
    (:mod:`emilia_pipeline.phase2.s4_client`); this holds the resolved settings so
    the factory contract is complete. Calling :meth:`label` directly raises
    ``NotImplementedError`` (the stage module drives the request).
    """

    is_mock = False

    def __init__(self, config: Config, api_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self.provider = config.s4.provider
        self.model = config.s4.model
        self.base_url = config.s4.resolved_base_url()
        self.prompt_version = config.s4.prompt_version

    async def label(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        reference_text: str,
        clip_id: str = "",
    ) -> S4GuidedJSON:  # pragma: no cover - implemented by the Phase-2 module
        raise NotImplementedError(
            "OmniApiClient.label is driven by emilia_pipeline.phase2.s4_client"
        )


# Backwards-compatible alias (older import sites / tests).
DashScopeS4Client = OmniApiClient


def get_s4_client(config: Config) -> BaseS4Client:
    """Return a real or mock S4 client.

    Uses the mock when ``config.runtime.use_mocks`` is set or when the API key
    (``config.s4.api_key_env``) is absent from the environment -- so tests and
    key-less runs degrade gracefully (project convention).

    Args:
        config: Pipeline config.

    Returns:
        A :class:`BaseS4Client`.
    """
    if config.runtime.use_mocks:
        return MockS4Client(config)
    api_key = config.api_key()
    if not api_key:
        return MockS4Client(config)
    return OmniApiClient(config, api_key)


__all__ = [
    "MODEL_AESTHETICS",
    "MODEL_DNSMOS",
    "MODEL_CAMPP",
    "content_hash",
    "BaseAudioModel",
    "MockAestheticsModel",
    "MockDnsmosModel",
    "MockCampPlusModel",
    "RealAestheticsModel",
    "RealDnsmosModel",
    "RealCampPlusModel",
    "get_model",
    "BaseS4Client",
    "MockS4Client",
    "OmniApiClient",
    "DashScopeS4Client",
    "get_s4_client",
]
