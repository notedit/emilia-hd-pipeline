"""Stage S3: sliding-window speaker purity detection (design doc v1.1 §4 S3).

POST-simplification: there is NO global speaker clustering, NO centroids. Speaker
identity is Emilia's ``original_speaker`` and is never re-derived here. S3 answers
one question per clip: *is this clip internally pure (single speaker, no intrusion
/ overlap)?* using only within-clip window-cosine shape plus the F0 tracker
confidence carried over from S2.

Pipeline (design §4 S3a / S3b):

  1. **S3a features (GPU, via the mock-aware CAM++ factory)**: CAM++ extracts a
     window-embedding sequence over 1.5 s windows at 50 % overlap; the clip-level
     embedding is the window mean. Self-consistency metrics:

       * ``mean_win_cos`` / ``min_win_cos``  -- each window's cosine to the window
         mean center, averaged / minimized.
       * contiguous low-cosine window spans -> ``intrusion_span_ms``.

  2. **S3b verdict (online, in-worker; pure window-sequence arithmetic, no
     centroid)**: maps the cosine shape + ``f0_tracker_confidence`` to a verdict in
     ``{single, intruded_trimmed, intruded_rejected, overlap_rejected,
     degraded_pass}``. Head/tail intrusions are trimmed (residual must stay
     >= ``s3.min_trim_residual_s``); a mid-clip intrusion is rejected.

The public entrypoint :func:`process_clip` returns the :class:`S3SpeakerRow`
(minus the embedding pointer, filled by the worker) *plus* the clip-level mean
embedding so the fusion worker can pack it into ``emb-{shard}.npy`` and set
``emb_file`` / ``emb_row``. All numeric metrics are stored; nothing is hard-dropped
here -- pass/reject is a downstream query condition (project convention). The
verdict itself IS the S3 judgment, but even a rejected clip's row is written.

The CAM++ model is obtained via :func:`emilia_pipeline.common.models.get_model`,
so unit tests run with the deterministic mock and zero GPU. When the real CAM++
model returns a full window-embedding sequence we recompute the cosine geometry
here; when only clip-level stats are available (the current mock surface) we trust
the model-provided ``mean_win_cos`` / ``min_win_cos`` / ``n_windows`` and derive
the verdict from them. Either path yields a schema-valid row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..common.config import Config
from ..common.contracts import GenderPred, S3SpeakerRow, SpeakerVerdict
from ..common.models import MODEL_CAMPP, BaseAudioModel, get_model

__all__ = [
    "S3Result",
    "WindowGeometry",
    "frame_windows",
    "window_geometry",
    "locate_low_cos_spans",
    "decide_verdict",
    "process_clip",
    "process_batch",
]


# ---------------------------------------------------------------------------
# Result container returned to the fusion worker
# ---------------------------------------------------------------------------


@dataclass
class S3Result:
    """S3 output for one clip.

    Attributes:
        row: The :class:`S3SpeakerRow` with every purity metric populated. Its
            ``emb_file`` / ``emb_row`` are placeholders (``""`` / ``-1``); the
            fusion worker overwrites them once it knows the clip's row index in
            the shard ``emb-{shard}.npy``. Use :meth:`with_embedding_ref`.
        embedding: The clip-level mean embedding (window mean), fp16, shape
            ``(embedding_dim,)`` -- to be stored in the sidecar npy, never parquet.
        trimmed_audio: If the verdict is ``intruded_trimmed``, the trimmed signal
            (float32 mono, same sample rate as input); otherwise ``None``. The
            worker uses this to update the clip's persisted audio / duration.
        trim_start_s / trim_end_s: The kept ``[start, end)`` span in seconds when
            trimmed (``s3_trim`` bookkeeping, design §4 S3b); ``None`` otherwise.
    """

    row: S3SpeakerRow
    embedding: np.ndarray
    trimmed_audio: Optional[np.ndarray] = None
    trim_start_s: Optional[float] = None
    trim_end_s: Optional[float] = None

    def with_embedding_ref(self, emb_file: str, emb_row: int) -> S3SpeakerRow:
        """Return a copy of :attr:`row` with the embedding pointer filled in.

        Args:
            emb_file: Basename of the shard embedding npy (e.g. ``emb-00042.npy``).
            emb_row: This clip's row index within that npy.

        Returns:
            An updated :class:`S3SpeakerRow`.
        """
        return self.row.model_copy(update={"emb_file": emb_file, "emb_row": emb_row})


@dataclass
class WindowGeometry:
    """Geometry of the window-embedding sequence for one clip.

    Attributes:
        n_windows: Number of windows.
        window_cos: Per-window cosine to the clip mean center, shape ``(n,)``.
        mean_win_cos: Mean of ``window_cos``.
        min_win_cos: Minimum of ``window_cos``.
        mean_embedding: The clip-level mean embedding (fp32, unit-agnostic).
    """

    n_windows: int
    window_cos: np.ndarray
    mean_win_cos: float
    min_win_cos: float
    mean_embedding: np.ndarray


# ---------------------------------------------------------------------------
# Windowing + cosine geometry (used when a real window-embedding sequence exists)
# ---------------------------------------------------------------------------


def frame_windows(
    n_samples: int, sr: int, window_s: float, overlap: float
) -> list[tuple[int, int]]:
    """Compute ``[start, stop)`` sample spans for 1.5 s / 50 %-overlap windows.

    The final partial window is kept only if it is at least half a window long,
    so a short clip still yields at least one window.

    Args:
        n_samples: Total sample count of the clip.
        sr: Sample rate in Hz.
        window_s: Window length in seconds (S3 config ``window_s``).
        overlap: Fractional overlap in ``[0, 1)`` (S3 config ``window_overlap``).

    Returns:
        A list of ``(start, stop)`` sample-index tuples (at least one).
    """
    win = max(1, int(round(window_s * sr)))
    if n_samples <= win:
        return [(0, n_samples)]
    hop = max(1, int(round(win * (1.0 - overlap))))
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_samples:
        stop = min(start + win, n_samples)
        # 保留末窗仅当其长度 >= 半窗，避免尾部碎片污染 cosine。
        if stop - start >= win // 2 or not spans:
            spans.append((start, stop))
        if stop >= n_samples:
            break
        start += hop
    return spans


def _cosine_to_center(windows: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Cosine of each row in ``windows`` to ``center`` (numerically safe)."""
    w = np.asarray(windows, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32)
    c_norm = float(np.linalg.norm(c)) + 1e-8
    w_norm = np.linalg.norm(w, axis=1) + 1e-8
    return (w @ c) / (w_norm * c_norm)


def window_geometry(window_embeddings: np.ndarray) -> WindowGeometry:
    """Compute self-consistency geometry from a window-embedding sequence.

    The clip center is the window mean; ``window_cos`` is each window's cosine to
    that center. This is the centroid-free, within-clip measure the design calls
    for (design §4 S3a).

    Args:
        window_embeddings: Array of shape ``(n_windows, dim)``.

    Returns:
        A :class:`WindowGeometry`.

    Raises:
        ValueError: If ``window_embeddings`` is empty.
    """
    emb = np.asarray(window_embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError("window_embeddings must be a non-empty (n, dim) array")
    center = emb.mean(axis=0)
    cos = _cosine_to_center(emb, center)
    return WindowGeometry(
        n_windows=int(emb.shape[0]),
        window_cos=cos.astype(np.float32),
        mean_win_cos=float(np.mean(cos)),
        min_win_cos=float(np.min(cos)),
        mean_embedding=center.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Low-cosine span localization -> intrusion_span_ms
# ---------------------------------------------------------------------------


@dataclass
class _Span:
    """A contiguous run of low-cosine windows [start_win, end_win] inclusive."""

    start_win: int
    end_win: int  # inclusive

    @property
    def length(self) -> int:
        return self.end_win - self.start_win + 1


def locate_low_cos_spans(window_cos: np.ndarray, threshold: float) -> list[_Span]:
    """Find maximal contiguous runs of windows with ``cos < threshold``.

    Args:
        window_cos: Per-window cosine to the clip center, shape ``(n,)``.
        threshold: Windows strictly below this are candidate intrusions
            (S3 config ``min_win_cos_threshold``).

    Returns:
        A list of :class:`_Span` (possibly empty), in window order.
    """
    low = np.asarray(window_cos) < threshold
    spans: list[_Span] = []
    start: Optional[int] = None
    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = i
        elif not is_low and start is not None:
            spans.append(_Span(start, i - 1))
            start = None
    if start is not None:
        spans.append(_Span(start, len(low) - 1))
    return spans


def _span_time_bounds(
    span: _Span, spans_samples: Sequence[tuple[int, int]], sr: int
) -> tuple[float, float]:
    """Return the ``[start_s, end_s)`` time span covered by a window run."""
    s0 = spans_samples[span.start_win][0]
    s1 = spans_samples[span.end_win][1]
    return s0 / float(sr), s1 / float(sr)


# ---------------------------------------------------------------------------
# S3b verdict decision (online, centroid-free)
# ---------------------------------------------------------------------------


@dataclass
class VerdictDecision:
    """Outcome of :func:`decide_verdict`."""

    verdict: SpeakerVerdict
    intrusion_span_ms: Optional[float] = None
    trimmed: bool = False
    trim_start_s: Optional[float] = None
    trim_end_s: Optional[float] = None
    trimmed_duration_s: Optional[float] = None


def decide_verdict(
    *,
    cfg: Config,
    n_windows: int,
    window_cos: Optional[np.ndarray],
    mean_win_cos: float,
    min_win_cos: float,
    f0_tracker_confidence: float,
    window_spans_samples: Optional[Sequence[tuple[int, int]]],
    sr: int,
    total_duration_s: float,
) -> VerdictDecision:
    """Map within-clip window-cosine shape + F0 confidence to a purity verdict.

    Implements the design §4 S3b decision table, centroid-free:

    ==============================  =========================  ==================
    window cosine shape             f0_tracker_confidence      verdict
    ==============================  =========================  ==================
    local contiguous collapse       --                         intrusion ->
                                                                head/tail trim
                                                                (residual >= 3s)
                                                                else mid ->
                                                                intruded_rejected
    uniformly depressed (spread)    poor                       overlap_rejected
    uniformly depressed             normal                     degraded_pass
    normal                          normal                     single
    ==============================  =========================  ==================

    The stage biases toward NOT over-killing (design: "本级阈值向不误杀偏"); the
    last line of defense is S1's PC<=2.5 and Tier-S manual sampling.

    Args:
        cfg: Pipeline config (reads ``cfg.s3`` thresholds).
        n_windows: Number of windows.
        window_cos: Per-window cosine sequence, or ``None`` when only clip-level
            aggregate stats are available (mock / minimal real path). Span
            localization and trimming require this; without it a clip that looks
            like a local collapse degrades to ``degraded_pass`` (no false trim).
        mean_win_cos / min_win_cos: Aggregate cosines.
        f0_tracker_confidence: Carried over from S2 (design: "透传给 S3").
        window_spans_samples: The ``(start, stop)`` sample spans matching
            ``window_cos`` (needed to convert a window run into ms / trim points).
        sr: Sample rate.
        total_duration_s: Full clip duration before any trim.

    Returns:
        A :class:`VerdictDecision`.
    """
    s3 = cfg.s3
    # 单窗片段无法判断内部一致性：直接判 single（无戏可挑）。
    if n_windows <= 1:
        return VerdictDecision(verdict=SpeakerVerdict.SINGLE)

    depressed = mean_win_cos < s3.mean_win_cos_threshold
    has_low_window = min_win_cos < s3.min_win_cos_threshold

    # --- Case A: local contiguous collapse (a real intrusion we can localize). ---
    if has_low_window and window_cos is not None and window_spans_samples is not None:
        spans = locate_low_cos_spans(window_cos, s3.min_win_cos_threshold)
        if spans:
            # 取最长的低相似窗段作为侵入段。
            span = max(spans, key=lambda sp: sp.length)
            start_s, end_s = _span_time_bounds(span, window_spans_samples, sr)
            intrusion_ms = (end_s - start_s) * 1000.0
            at_head = span.start_win == 0
            at_tail = span.end_win == n_windows - 1
            # 若整段都塌陷（覆盖所有窗）视为均匀压低，交给下面的 depressed 分支。
            covers_all = span.length >= n_windows
            if not covers_all and (at_head or at_tail) and not (at_head and at_tail):
                # 首/尾侵入 -> 修剪回收，剩余需 >= min_trim_residual_s。
                if at_head:
                    keep_start_s, keep_end_s = end_s, total_duration_s
                else:
                    keep_start_s, keep_end_s = 0.0, start_s
                residual = keep_end_s - keep_start_s
                if residual >= s3.min_trim_residual_s:
                    return VerdictDecision(
                        verdict=SpeakerVerdict.INTRUDED_TRIMMED,
                        intrusion_span_ms=intrusion_ms,
                        trimmed=True,
                        trim_start_s=keep_start_s,
                        trim_end_s=keep_end_s,
                        trimmed_duration_s=residual,
                    )
                # 修剪后过短 -> 无法回收，判 rejected。
                return VerdictDecision(
                    verdict=SpeakerVerdict.INTRUDED_REJECTED,
                    intrusion_span_ms=intrusion_ms,
                )
            if not covers_all:
                # 中段侵入无法靠首尾修剪回收 -> rejected。
                return VerdictDecision(
                    verdict=SpeakerVerdict.INTRUDED_REJECTED,
                    intrusion_span_ms=intrusion_ms,
                )

    # --- Case B: uniformly depressed cosine (spread across all windows). ---
    if depressed:
        if f0_tracker_confidence < s3.f0_confidence_poor:
            # 均匀压低 + F0 差 -> 疑似 overlap，拒。
            return VerdictDecision(verdict=SpeakerVerdict.OVERLAP_REJECTED)
        # 均匀压低但 F0 正常 -> 放行，交给 S1/S5 分数体系兜底。
        return VerdictDecision(verdict=SpeakerVerdict.DEGRADED_PASS)

    # --- Case C: a low window we could not localize (no window sequence). ---
    if has_low_window:
        # 无法定位侵入段（仅有聚合统计），偏向不误杀 -> degraded_pass。
        return VerdictDecision(verdict=SpeakerVerdict.DEGRADED_PASS)

    # --- Case D: normal shape + (implicitly) normal confidence -> single. ---
    return VerdictDecision(verdict=SpeakerVerdict.SINGLE)


# ---------------------------------------------------------------------------
# Model-output normalization
# ---------------------------------------------------------------------------


def _extract_window_embeddings(pred: dict[str, Any]) -> Optional[np.ndarray]:
    """Pull a ``(n_windows, dim)`` window-embedding matrix from a model dict.

    Real CAM++ implementations are expected to expose the per-window sequence
    under one of these keys; the current mock does not, so this returns ``None``
    and the caller falls back to the model-provided aggregate stats.
    """
    for key in ("window_embeddings", "win_embeddings", "windows"):
        val = pred.get(key)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            return arr
    return None


def _coerce_gender(value: Any) -> GenderPred:
    """Coerce a model gender string/enum into :class:`GenderPred` (safe default)."""
    if isinstance(value, GenderPred):
        return value
    try:
        return GenderPred(str(value))
    except ValueError:
        return GenderPred.UNKNOWN


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def process_clip(
    audio: np.ndarray,
    sr: int,
    *,
    clip_id: str,
    shard: str,
    original_speaker: str,
    f0_tracker_confidence: float,
    cfg: Config,
    model: Optional[BaseAudioModel] = None,
) -> S3Result:
    """Run S3 sliding-window purity detection on a single clip.

    Extracts CAM++ window embeddings (via the mock-aware factory), computes the
    self-consistency geometry, localizes low-cosine spans, and decides the S3b
    verdict online -- all centroid-free. Returns the metric row and the clip mean
    embedding for the worker to persist.

    Args:
        audio: float32 mono samples in ``[-1, 1]`` at ``sr``.
        sr: Sample rate in Hz (Emilia native 24 kHz upstream).
        clip_id: Stable clip identifier.
        shard: Source shard token (for the row's ``shard`` field).
        original_speaker: Emilia's ``original_speaker`` (identity is NOT re-derived).
        f0_tracker_confidence: Carried over from S2 (design: "透传给 S3").
        cfg: Pipeline config (``cfg.s3`` thresholds, ``cfg.runtime.use_mocks``).
        model: Optional pre-built CAM++ model to reuse across clips in a batch;
            when ``None`` one is obtained from :func:`get_model`.

    Returns:
        An :class:`S3Result`. The row's ``emb_file`` / ``emb_row`` are placeholders
        for the worker to fill via :meth:`S3Result.with_embedding_ref`.
    """
    campp = model if model is not None else get_model(MODEL_CAMPP, cfg)
    pred = campp.predict([(audio, sr)])[0]
    return _build_result(
        pred=pred,
        audio=audio,
        sr=sr,
        clip_id=clip_id,
        shard=shard,
        original_speaker=original_speaker,
        f0_tracker_confidence=f0_tracker_confidence,
        cfg=cfg,
    )


def process_batch(
    clips: Sequence[tuple[np.ndarray, int]],
    *,
    clip_ids: Sequence[str],
    shard: str,
    original_speakers: Sequence[str],
    f0_tracker_confidences: Sequence[float],
    cfg: Config,
    model: Optional[BaseAudioModel] = None,
) -> list[S3Result]:
    """Batched variant of :func:`process_clip` (one CAM++ forward for all clips).

    The fusion worker (design §6.2) batches CAM++ over all S1-passing clips in a
    shard. This runs the model once and post-processes each clip independently.

    Args:
        clips: Sequence of ``(samples, sr)`` for each clip, in row order.
        clip_ids: Clip identifiers, aligned with ``clips``.
        shard: Source shard token.
        original_speakers: Emilia ``original_speaker`` per clip, aligned.
        f0_tracker_confidences: S2 F0 confidences per clip, aligned.
        cfg: Pipeline config.
        model: Optional pre-built CAM++ model.

    Returns:
        A list of :class:`S3Result`, aligned with ``clips``.

    Raises:
        ValueError: If the per-clip sequences have mismatched lengths.
    """
    n = len(clips)
    if not (len(clip_ids) == len(original_speakers) == len(f0_tracker_confidences) == n):
        raise ValueError("process_batch: all per-clip sequences must have equal length")
    campp = model if model is not None else get_model(MODEL_CAMPP, cfg)
    preds = campp.predict(list(clips)) if n else []
    results: list[S3Result] = []
    for i in range(n):
        arr, sr = clips[i]
        results.append(
            _build_result(
                pred=preds[i],
                audio=arr,
                sr=sr,
                clip_id=clip_ids[i],
                shard=shard,
                original_speaker=original_speakers[i],
                f0_tracker_confidence=f0_tracker_confidences[i],
                cfg=cfg,
            )
        )
    return results


def _build_result(
    *,
    pred: dict[str, Any],
    audio: np.ndarray,
    sr: int,
    clip_id: str,
    shard: str,
    original_speaker: str,
    f0_tracker_confidence: float,
    cfg: Config,
) -> S3Result:
    """Assemble an :class:`S3Result` from one model prediction + audio.

    Prefers a real window-embedding sequence when the model provides it
    (recomputes geometry + span localization here); otherwise trusts the model's
    aggregate ``mean_win_cos`` / ``min_win_cos`` / ``n_windows`` (the mock path).
    """
    s3 = cfg.s3
    total_dur = float(len(audio)) / float(sr) if sr else 0.0

    win_emb = _extract_window_embeddings(pred)
    window_cos: Optional[np.ndarray] = None
    window_spans_samples: Optional[list[tuple[int, int]]] = None

    if win_emb is not None:
        geom = window_geometry(win_emb)
        n_windows = geom.n_windows
        mean_win_cos = geom.mean_win_cos
        min_win_cos = geom.min_win_cos
        window_cos = geom.window_cos
        mean_embedding = geom.mean_embedding.astype(np.float16)
        window_spans_samples = frame_windows(
            len(audio), sr, s3.window_s, s3.window_overlap
        )
        # 窗数与实际切窗对齐（模型窗序列长度优先）。
        if len(window_spans_samples) != n_windows:
            window_spans_samples = _resize_spans(window_spans_samples, n_windows)
    else:
        # Mock / minimal-real path: trust aggregate stats from the model dict.
        n_windows = int(pred.get("n_windows", 1))
        mean_win_cos = float(pred.get("mean_win_cos", 1.0))
        min_win_cos = float(pred.get("min_win_cos", mean_win_cos))
        emb = pred.get("embedding")
        mean_embedding = (
            np.asarray(emb, dtype=np.float16)
            if emb is not None
            else np.zeros(s3.embedding_dim, dtype=np.float16)
        )

    # f0_stability: model may provide it; else reuse the S2 confidence passthrough.
    f0_stability = float(pred.get("f0_stability", f0_tracker_confidence))
    gender = _coerce_gender(pred.get("gender_pred", GenderPred.UNKNOWN))

    decision = decide_verdict(
        cfg=cfg,
        n_windows=n_windows,
        window_cos=window_cos,
        mean_win_cos=mean_win_cos,
        min_win_cos=min_win_cos,
        f0_tracker_confidence=f0_tracker_confidence,
        window_spans_samples=window_spans_samples,
        sr=sr,
        total_duration_s=total_dur,
    )

    # Model may have pre-computed an intrusion span; prefer our localized value.
    intrusion_ms = decision.intrusion_span_ms
    if intrusion_ms is None and pred.get("intrusion_span_ms") is not None:
        intrusion_ms = float(pred["intrusion_span_ms"])

    trimmed_audio: Optional[np.ndarray] = None
    if decision.trimmed and decision.trim_start_s is not None:
        from ..common.audio import trim_segment

        trimmed_audio = trim_segment(
            audio, sr, decision.trim_start_s, decision.trim_end_s
        )

    row = S3SpeakerRow(
        clip_id=clip_id,
        shard=shard,
        original_speaker=original_speaker,
        emb_file="",  # filled by the worker (with_embedding_ref)
        emb_row=-1,
        gender_pred=gender,
        n_windows=n_windows,
        mean_win_cos=mean_win_cos,
        min_win_cos=min_win_cos,
        f0_stability=f0_stability,
        verdict=decision.verdict,
        intrusion_span_ms=intrusion_ms,
        trimmed=decision.trimmed,
        trimmed_duration_s=decision.trimmed_duration_s,
        trim_start_s=decision.trim_start_s,
        trim_end_s=decision.trim_end_s,
    )
    return S3Result(
        row=row,
        embedding=np.asarray(mean_embedding, dtype=np.float16).reshape(-1),
        trimmed_audio=trimmed_audio,
        trim_start_s=decision.trim_start_s,
        trim_end_s=decision.trim_end_s,
    )


def _resize_spans(
    spans: list[tuple[int, int]], n: int
) -> list[tuple[int, int]]:
    """Pad/truncate a span list to exactly ``n`` entries (defensive alignment)."""
    if len(spans) == n:
        return spans
    if len(spans) > n:
        return spans[:n]
    last = spans[-1] if spans else (0, 0)
    return spans + [last] * (n - len(spans))
