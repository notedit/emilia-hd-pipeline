"""Phase 2 · S4 Qwen3-Omni structured labeling client (asyncio, cloud API).

Design refs: §5 (S4 detail), §6.1 (task claiming), §6.3 (asyncio client).

One task == one worklist slice (default 5,000 clips). For a slice we:

  1. Load the slice's worklist entries (clip_id + reference text) and order the
     audio reads by ``repack_index`` (sequential tar reads, design §6.3).
  2. Decode each clip's audio, resample to ``config.s4.sample_rate`` and encode
     it to a base64 WAV data-URI.
  3. Build chat messages for the target provider: a system prompt (Chinese
     annotation expert with per-field rubric anchors, incl. intensity /
     expressiveness 1-5 anchors) plus a user turn (audio + reference text + the
     §5.3 guided-JSON instruction). The default provider is the internal **Venus
     LLM proxy**, which uses a proprietary ``venus_multimodal_url`` audio content
     part; ``config.s4.provider="openai"`` switches to the standard
     ``input_audio`` part.
  4. Call the transport via the **AsyncOpenAI SDK** with ``temperature=0`` and
     ``max_tokens`` from config. On the Venus path JSON is prompt-enforced and
     validated client-side (no ``response_format``); ``config.s4.use_guided_json``
     enables a ``json_schema`` response_format for endpoints that support it.
  5. Bound in-flight concurrency with a global :class:`asyncio.Semaphore` and
     retry transient 429 / 5xx (and transient parse) errors with exponential
     backoff; on final failure the row is written with ``s4_status=failed`` and
     an ``error`` field (never dropped, design §6.3).
  6. At slice end, atomically write ``s4_labels/part-{slice}.parquet`` then the
     done marker (write discipline, §3).

The transport itself sits behind the shared mock factory
(:func:`emilia_pipeline.common.models.get_s4_client`): a deterministic
schema-valid mock when ``use_mocks`` is set or no API key is present, otherwise
the real :class:`~emilia_pipeline.common.models.OmniApiClient` (driven here).
This keeps the whole module importable and unit-testable with zero GPU / key /
network.

Two-pass triage toggle (§5.2): when ``config.s4.two_pass_triage`` is on (or a
pilot pass-rate falls below ``triage_pass_rate_switch``), each clip is first sent
through a cheap triage call; only clips whose triage says ``usable`` proceed to
full labeling. Triaged-out clips are recorded as ``s4_status=ok`` with
``labels=None`` (not a failure, just not fully labeled).
"""

from __future__ import annotations

import abc
import asyncio
import base64
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

from ..common.audio import decode_tar_member, duration_s, resample
from ..common.config import Config
from ..common.contracts import S4GuidedJSON, S4LabelRow, S4Status
from ..common.io_utils import atomic_write_parquet, write_done_marker
from ..common.models import BaseS4Client, OmniApiClient, get_s4_client

# Columns the worklist / repack_index may expose (tolerant extraction).
_CLIP_ID_KEYS = ("clip_id", "id", "key")
_SLICE_ID_KEYS = ("slice_id", "slice")
_TEXT_KEYS = ("original_text", "reference_text", "text")
_SHARD_KEYS = ("shard", "repack_shard")
_OFFSET_KEYS = ("offset", "repack_offset", "row", "emb_row")
_MEMBER_KEYS = ("member", "flac", "audio_member")


# ---------------------------------------------------------------------------
# Worklist / audio-source plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorklistEntry:
    """One clip to label within a slice.

    Attributes:
        clip_id: Stable clip identifier (worklist / repack_index key).
        slice_id: The slice this clip belongs to.
        reference_text: Emilia ``original_text`` used as the prompt reference.
        shard: Repacked shard token (for ordered reads); "" if unknown.
        offset: Position within the repacked shard (for ordered reads); -1 unknown.
        member: Explicit tar member name for the audio; "" -> ``{clip_id}.flac``.
    """

    clip_id: str
    slice_id: str
    reference_text: str = ""
    shard: str = ""
    offset: int = -1
    member: str = ""


def _first_key(mapping: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def load_slice_worklist(
    config: Config, slice_id: str, *, worklist_path: Optional[Path] = None
) -> list[WorklistEntry]:
    """Load and order the worklist entries for one slice.

    Reads ``manifests/s4_worklist_v1.parquet`` (or ``worklist_path``), keeps rows
    whose ``slice_id`` matches, and orders them by repack position
    ``(shard, offset)`` so audio is read sequentially (design §6.3). When the
    repack columns are absent the parquet row order is preserved.

    Args:
        config: Pipeline config (supplies the manifests directory).
        slice_id: The slice task id.
        worklist_path: Optional override of the worklist parquet path.

    Returns:
        Ordered :class:`WorklistEntry` list for the slice (possibly empty).
    """
    path = worklist_path or (config.paths.manifests / "s4_worklist_v1.parquet")
    path = Path(path)
    if not path.exists():
        return []
    table = pq.read_table(path)
    rows = table.to_pylist()
    entries: list[WorklistEntry] = []
    for row in rows:
        row_slice = str(_first_key(row, _SLICE_ID_KEYS, ""))
        if row_slice != str(slice_id):
            continue
        entries.append(_entry_from_row(row, slice_id))
    entries.sort(key=lambda e: (e.shard, e.offset if e.offset >= 0 else 0))
    return entries


def _entry_from_row(row: Mapping[str, Any], slice_id: str) -> WorklistEntry:
    """Build a :class:`WorklistEntry` from a tolerant worklist parquet row."""
    offset_raw = _first_key(row, _OFFSET_KEYS, -1)
    try:
        offset = int(offset_raw)
    except (TypeError, ValueError):
        offset = -1
    return WorklistEntry(
        clip_id=str(_first_key(row, _CLIP_ID_KEYS, "")),
        slice_id=str(slice_id),
        reference_text=str(_first_key(row, _TEXT_KEYS, "")),
        shard=str(_first_key(row, _SHARD_KEYS, "")),
        offset=offset,
        member=str(_first_key(row, _MEMBER_KEYS, "")),
    )


class AudioSource(abc.ABC):
    """Resolves a ``clip_id`` to decoded ``(float32 mono samples, sample_rate)``.

    Injected so tests can feed synthetic audio in-memory while production reads
    from the repacked WebDataset shards via the repack index.
    """

    @abc.abstractmethod
    def load(self, entry: "WorklistEntry") -> tuple[np.ndarray, int]:
        """Return decoded ``(samples, sr)`` for one worklist entry."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any open handles (tar files, etc.)."""


class DictAudioSource(AudioSource):
    """In-memory audio source keyed by ``clip_id`` (used by tests)."""

    def __init__(self, clips: Mapping[str, tuple[np.ndarray, int]]) -> None:
        self._clips = dict(clips)

    def load(self, entry: WorklistEntry) -> tuple[np.ndarray, int]:
        if entry.clip_id not in self._clips:
            raise KeyError(f"clip not found in DictAudioSource: {entry.clip_id!r}")
        arr, sr = self._clips[entry.clip_id]
        return np.asarray(arr, dtype=np.float32), int(sr)


class RepackIndexAudioSource(AudioSource):
    """Reads clip audio from the repacked WebDataset shards via ``repack_index``.

    ``repacked/repack_index.parquet`` maps ``clip_id -> (shard, member)``. Opened
    tar handles are cached so a slice (whose clips are grouped by shard) reads
    each shard sequentially. Real-data path; never exercised in CI.
    """

    def __init__(
        self, config: Config, *, repack_index_path: Optional[Path] = None
    ) -> None:
        self._config = config
        self._repacked_dir = Path(config.paths.repacked)
        self._index_path = Path(
            repack_index_path or (self._repacked_dir / "repack_index.parquet")
        )
        self._index: Optional[dict[str, tuple[str, str]]] = None
        self._open_tars: dict[str, Any] = {}

    def _load_index(self) -> dict[str, tuple[str, str]]:
        if self._index is not None:
            return self._index
        index: dict[str, tuple[str, str]] = {}
        if self._index_path.exists():
            table = pq.read_table(self._index_path)
            for row in table.to_pylist():
                clip_id = str(_first_key(row, _CLIP_ID_KEYS, ""))
                shard = str(_first_key(row, _SHARD_KEYS, ""))
                member = str(_first_key(row, _MEMBER_KEYS, ""))
                index[clip_id] = (shard, member)
        self._index = index
        return index

    def load(self, entry: WorklistEntry) -> tuple[np.ndarray, int]:
        import tarfile

        index = self._load_index()
        shard = entry.shard
        member = entry.member
        if entry.clip_id in index:
            idx_shard, idx_member = index[entry.clip_id]
            shard = shard or idx_shard
            member = member or idx_member
        if not member:
            member = f"{entry.clip_id}.flac"
        if not shard:
            raise KeyError(f"no repack shard for clip {entry.clip_id!r}")
        tar = self._open_tars.get(shard)
        if tar is None:
            tar = tarfile.open(self._repacked_dir / f"shard-{shard}.tar", "r")
            self._open_tars[shard] = tar
        return decode_tar_member(tar, member)

    def close(self) -> None:
        for tar in self._open_tars.values():
            try:
                tar.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        self._open_tars.clear()


# ---------------------------------------------------------------------------
# Prompt construction (§5.4)
# ---------------------------------------------------------------------------

# System prompt: Chinese annotation expert with per-field rubric anchors. The
# intensity / expressiveness 1-5 anchors are the primary calibration lever (§10),
# so they are spelled out explicitly.
_SYSTEM_PROMPT = """你是一名资深的中文语音数据标注专家，负责为语音片段产出结构化的情感 / 韵律 / 语境 / 语种标签。
只依据你"听到的音频"和给定"参考文本"进行判断，输出必须严格符合给定 JSON schema，使用封闭词表，不得编造词表外的取值。

字段判据：
- text_verdict: 音频内容与参考文本是否一致。match=完全一致；fixable=仅标点/少量错别字可修；broken=大段不符或无法对齐。
- text_fixed: 依据音频纠正后的文本（无需纠正则原样返回）。
- text_punctuated: 加上自然标点、语气停顿的文本。
- emotion.primary / secondary: 主要 / 次要情感，取自封闭词表；无次要情感则 secondary=null。
- emotion.intensity（1-5 情感强度锚点）:
    1=几乎无情感起伏，平铺直叙；
    2=有轻微情感色彩但克制；
    3=情感明确可辨，中等强度；
    4=情感强烈，语气明显起伏；
    5=极强烈，情绪外放/激烈（如嚎啕、暴怒、狂喜）。
- emotion.confidence: 你对情感判断的把握度 0.0-1.0。
- prosody.expressiveness（1-5 表现力锚点）:
    1=单调朗读，音高/节奏几乎无变化；
    2=略有变化但整体平淡；
    3=有一定抑扬顿挫，表达自然；
    4=富有表现力，音高/重音/节奏变化明显；
    5=极具表现力/戏剧化，充满张力与对比。
- prosody.speaking_style: 说话风格（叙述/对话/讲故事/演讲/播报/演绎/vlog/访谈）。
- prosody.rhythm: steady=平稳；varied=有变化；dramatic=戏剧化强对比。
- prosody.prominent_stress: 是否有明显的重音强调。
- context.scenario / register / summary: 场景、语域（正式/随意/亲密）、一句话内容摘要。
- language.primary / code_switch / accent: 主语种、是否夹杂其他语种、口音（标准/带口音/方言）。
- paralinguistic: 副语言事件多标签列表（笑声/叹气/哭腔/明显呼吸/口头语多/不流畅），无则空列表。
- defects: 缺陷多标签列表（头截断/尾截断/伪影/其他），无则空列表。
- usable: 该片段是否适合用于高质量情感 TTS 训练。

务必区分易混情况：激动(excited) vs 愤怒(angry)；演绎(acting) vs 朗读(narration)；疑问 vs 反问的标点。"""

# Triage system prompt (§5.2): short, ~30-token output, decides only usability.
_TRIAGE_SYSTEM_PROMPT = """你是中文语音数据分流员。快速判断该语音片段是否值得进入完整标注：
只需评估其情感表达是否充分、音频与参考文本是否大致一致、是否适合高质量情感 TTS 训练。
输出严格符合给定 JSON schema。"""

_USER_INSTRUCTION = (
    "请听音频，并结合下面的参考文本进行标注。仅输出符合 schema 的 JSON，不要输出任何解释文字。\n"
    "参考文本：「{reference_text}」"
)

_TRIAGE_USER_INSTRUCTION = (
    "请听音频并结合参考文本，快速判断是否值得完整标注。仅输出符合 schema 的 JSON。\n"
    "参考文本：「{reference_text}」"
)


def build_system_prompt(*, triage: bool = False) -> str:
    """Return the system prompt (full annotation or short triage variant)."""
    return _TRIAGE_SYSTEM_PROMPT if triage else _SYSTEM_PROMPT


def build_messages(
    *,
    reference_text: str,
    audio_b64: str,
    audio_format: str = "wav",
    triage: bool = False,
    provider: str = "venus",
) -> list[dict[str, Any]]:
    """Build chat messages for one clip, per the target provider's audio schema.

    Two audio content encodings are supported:

    * ``provider="venus"`` (default) -- the internal Venus LLM proxy's proprietary
      ``venus_multimodal_url`` part carrying ``{mimeType, url}`` where ``url`` is a
      ``data:<mime>;base64,<payload>`` URI. Venus puts the audio + text in ONE
      user message's content list and takes no separate system turn semantics for
      the audio, but a system turn is still accepted, so we keep it.
    * ``provider="openai"`` -- the standard OpenAI ``input_audio`` part.

    Args:
        reference_text: Emilia original text for the clip.
        audio_b64: Base64-encoded audio payload (no data-URI prefix).
        audio_format: Container format hint (``wav``); also drives the MIME type.
        triage: Use the short triage prompt when True (§5.2).
        provider: ``"venus"`` or ``"openai"``.

    Returns:
        A ``messages`` list with a system turn and a multimodal user turn
        (text instruction + audio).
    """
    instruction = (_TRIAGE_USER_INSTRUCTION if triage else _USER_INSTRUCTION).format(
        reference_text=reference_text
    )
    mime = _AUDIO_MIME.get(audio_format.lower(), "audio/wav")
    if provider == "venus":
        audio_part = {
            "type": "venus_multimodal_url",
            "venus_multimodal_url": {
                "mimeType": mime,
                "url": f"data:{mime};base64,{audio_b64}",
            },
        }
    else:  # standard OpenAI input_audio
        audio_part = {
            "type": "input_audio",
            "input_audio": {
                "data": f"data:{mime};base64,{audio_b64}",
                "format": audio_format,
            },
        }
    return [
        {"role": "system", "content": build_system_prompt(triage=triage)},
        {
            "role": "user",
            "content": [
                audio_part,
                {"type": "text", "text": instruction},
            ],
        },
    ]


# MIME types the audio encoder / message builder recognize.
_AUDIO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mp3",
    "mpeg": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


def guided_json_schema() -> dict[str, Any]:
    """Return the JSON schema used for guided/structured output (from §5.3)."""
    return S4GuidedJSON.model_json_schema()


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


def encode_audio_datauri(
    audio: np.ndarray, sr_in: int, target_sr: int
) -> tuple[str, str]:
    """Resample to ``target_sr`` and return ``(base64_wav, "wav")``.

    Args:
        audio: float32 mono samples in ``[-1, 1]``.
        sr_in: Input sample rate.
        target_sr: API-required sample rate (``config.s4.sample_rate``).

    Returns:
        ``(base64_string, format)`` where ``format`` is ``"wav"``.
    """
    import soundfile as sf

    resampled = resample(np.asarray(audio, dtype=np.float32), sr_in, target_sr)
    buf = io.BytesIO()
    sf.write(buf, resampled, target_sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii"), "wav"


# ---------------------------------------------------------------------------
# CER (stored, never gates)
# ---------------------------------------------------------------------------


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character-level error rate = edit_distance(ref, hyp) / len(ref).

    Returns 0.0 when both are empty and 1.0 when only the reference is empty but
    the hypothesis is not. Stored as ``cer_vs_original``; never used to drop rows.
    """
    ref = reference or ""
    hyp = hypothesis or ""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        cur = [i]
        for j, hc in enumerate(hyp, start=1):
            cost = 0 if rc == hc else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(ref)


# ---------------------------------------------------------------------------
# Transport: retry / backoff around BaseS4Client (mock) or real DashScope
# ---------------------------------------------------------------------------


class S4TransportError(RuntimeError):
    """Raised by the real transport; ``retryable`` marks 429 / 5xx / transient."""

    def __init__(self, message: str, *, retryable: bool, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


def _backoff_delay(attempt: int, config: Config) -> float:
    """Exponential backoff with full jitter for retry ``attempt`` (0-based)."""
    retry = config.s4.retry
    raw = retry.base_delay_s * (retry.backoff_multiplier**attempt)
    capped = min(raw, retry.max_delay_s)
    return random.uniform(0.0, capped)


async def _call_transport(
    client: BaseS4Client,
    *,
    audio: np.ndarray,
    sample_rate: int,
    reference_text: str,
    clip_id: str,
    config: Config,
    triage: bool,
) -> S4GuidedJSON:
    """One transport call. Mock uses the factory client; real hits the Omni API.

    The mock (:class:`~emilia_pipeline.common.models.MockS4Client`) works for
    both triage and full labeling (deterministic schema-valid JSON). The real
    :class:`OmniApiClient` defers its ``label`` to this module, which issues the
    request via the AsyncOpenAI SDK (this module owns the request / base64 /
    provider-specific message plumbing per the Foundation contract).
    """
    if getattr(client, "is_mock", False):
        return await client.label(
            audio=audio,
            sample_rate=sample_rate,
            reference_text=reference_text,
            clip_id=clip_id,
        )
    if isinstance(client, OmniApiClient):
        return await _omni_request(
            client,
            audio=audio,
            sample_rate=sample_rate,
            reference_text=reference_text,
            config=config,
            triage=triage,
        )
    # Unknown real client: defer to its own label() implementation.
    return await client.label(
        audio=audio,
        sample_rate=sample_rate,
        reference_text=reference_text,
        clip_id=clip_id,
    )


async def _omni_request(
    client: OmniApiClient,
    *,
    audio: np.ndarray,
    sample_rate: int,
    reference_text: str,
    config: Config,
    triage: bool,
) -> S4GuidedJSON:
    """Issue one chat/completions request via the AsyncOpenAI SDK.

    Mirrors the working Venus example: build base64 audio + provider-specific
    messages, ``temperature=0`` (greedy), ``max_tokens`` from config, and NO
    ``response_format`` on the Venus path (JSON is prompt-enforced and validated
    client-side). Optionally sends a ``json_schema`` response_format when
    ``config.s4.use_guided_json`` (for endpoints that support it). OpenAI
    SDK errors are mapped to a retryable :class:`S4TransportError` for 429/5xx.
    Never runs in CI (no key).
    """
    from openai import AsyncOpenAI

    audio_b64, fmt = encode_audio_datauri(audio, sample_rate, config.s4.sample_rate)
    messages = build_messages(
        reference_text=reference_text,
        audio_b64=audio_b64,
        audio_format=fmt,
        triage=triage,
        provider=client.provider,
    )
    kwargs: dict[str, Any] = {
        "model": client.model,
        "messages": messages,
        "temperature": config.s4.temperature,
        "max_tokens": config.s4.max_tokens,
    }
    if config.s4.use_guided_json:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "s4_guided", "schema": guided_json_schema()},
        }

    sdk = AsyncOpenAI(
        base_url=client.base_url,
        api_key=client.api_key,
        timeout=config.s4.request_timeout_s,
    )
    try:
        resp = await sdk.chat.completions.create(**kwargs)
    except Exception as exc:  # map OpenAI SDK errors -> retryable transport error
        raise _as_transport_error(exc) from exc
    finally:
        await sdk.close()

    content = resp.choices[0].message.content
    if isinstance(content, list):  # some endpoints return content parts
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return _parse_guided_json(content)


def _as_transport_error(exc: Exception) -> S4TransportError:
    """Classify an OpenAI SDK exception into a (possibly retryable) transport error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    name = type(exc).__name__
    # 429 / 5xx / connection / timeout are transient; other 4xx are not.
    retryable = (
        (isinstance(status, int) and (status == 429 or status >= 500))
        or "RateLimit" in name
        or "Timeout" in name
        or "APIConnection" in name
        or "InternalServer" in name
    )
    return S4TransportError(f"{name}: {exc}", retryable=retryable, status=status if isinstance(status, int) else None)


def _parse_guided_json(content: str) -> S4GuidedJSON:
    """Parse + validate a model response into :class:`S4GuidedJSON`.

    Raises a retryable :class:`S4TransportError` on malformed JSON / schema
    mismatch so a transient bad generation gets one more attempt.
    """
    text = (content or "").strip()
    if text.startswith("```"):  # strip markdown fences if present
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise S4TransportError(f"invalid JSON: {exc}", retryable=True) from exc
    try:
        return S4GuidedJSON.model_validate(obj)
    except Exception as exc:  # pydantic ValidationError
        raise S4TransportError(f"schema validation failed: {exc}", retryable=True) from exc


async def label_with_retry(
    client: BaseS4Client,
    *,
    audio: np.ndarray,
    sample_rate: int,
    reference_text: str,
    clip_id: str,
    config: Config,
    semaphore: asyncio.Semaphore,
    triage: bool = False,
) -> S4GuidedJSON:
    """Label one clip with bounded concurrency and exponential backoff.

    Retries transient failures (429 / 5xx / transport / transient parse) up to
    ``config.s4.retry.max_attempts`` times with jittered exponential backoff.
    Non-retryable errors (4xx other than 429) raise immediately. The global
    ``semaphore`` caps in-flight requests (design §6.3).

    Raises:
        Exception: The last error if all attempts are exhausted; the caller turns
            it into a ``s4_status=failed`` row (never dropped).
    """
    max_attempts = max(1, config.s4.retry.max_attempts)
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            async with semaphore:
                return await _call_transport(
                    client,
                    audio=audio,
                    sample_rate=sample_rate,
                    reference_text=reference_text,
                    clip_id=clip_id,
                    config=config,
                    triage=triage,
                )
        except S4TransportError as exc:
            last_exc = exc
            if not exc.retryable or attempt == max_attempts - 1:
                raise
        except Exception as exc:  # transport-agnostic: retry then surface
            last_exc = exc
            if attempt == max_attempts - 1:
                raise
        await asyncio.sleep(_backoff_delay(attempt, config))
    assert last_exc is not None  # unreachable
    raise last_exc


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _ok_row(
    entry: WorklistEntry, config: Config, labels: Optional[S4GuidedJSON]
) -> S4LabelRow:
    """Build a successful (or triaged-out) row. ``labels=None`` == triaged out."""
    cer = None
    if labels is not None:
        cer = char_error_rate(entry.reference_text, labels.text_fixed)
    return S4LabelRow(
        clip_id=entry.clip_id,
        slice_id=entry.slice_id,
        model=config.s4.model,
        prompt_version=config.s4.prompt_version,
        s4_status=S4Status.OK,
        error=None,
        cer_vs_original=cer,
        labels=labels,
    )


def _failed_row(entry: WorklistEntry, config: Config, error: str) -> S4LabelRow:
    """Build a failed row (kept, not dropped; design §6.3)."""
    return S4LabelRow(
        clip_id=entry.clip_id,
        slice_id=entry.slice_id,
        model=config.s4.model,
        prompt_version=config.s4.prompt_version,
        s4_status=S4Status.FAILED,
        error=error[:500],
        cer_vs_original=None,
        labels=None,
    )


async def _label_entry(
    entry: WorklistEntry,
    *,
    client: BaseS4Client,
    audio_source: AudioSource,
    config: Config,
    semaphore: asyncio.Semaphore,
    two_pass: bool,
) -> S4LabelRow:
    """Decode + (optionally triage +) label one entry, producing one row."""
    try:
        audio, sr = audio_source.load(entry)
    except Exception as exc:  # missing/undecodable audio -> failed row
        return _failed_row(entry, config, f"audio load failed: {exc}")

    if duration_s(audio, sr) <= 0.0:
        return _failed_row(entry, config, "empty audio")

    try:
        if two_pass:
            triage = await label_with_retry(
                client,
                audio=audio,
                sample_rate=sr,
                reference_text=entry.reference_text,
                clip_id=entry.clip_id,
                config=config,
                semaphore=semaphore,
                triage=True,
            )
            if not triage.usable:
                # Triaged out: recorded OK with no full labels (§5.2).
                return _ok_row(entry, config, None)
        labels = await label_with_retry(
            client,
            audio=audio,
            sample_rate=sr,
            reference_text=entry.reference_text,
            clip_id=entry.clip_id,
            config=config,
            semaphore=semaphore,
            triage=False,
        )
        return _ok_row(entry, config, labels)
    except Exception as exc:  # all retries exhausted -> failed row
        return _failed_row(entry, config, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Slice-level entry points
# ---------------------------------------------------------------------------


def should_use_two_pass(config: Config, pilot_pass_rate: Optional[float] = None) -> bool:
    """Decide single- vs two-pass labeling (§5.2).

    Two-pass is used when the config toggle is on, or when an observed pilot
    pass-rate is below ``config.s4.triage_pass_rate_switch`` (default 0.60).
    """
    if config.s4.two_pass_triage:
        return True
    if pilot_pass_rate is None:
        return False
    return pilot_pass_rate < config.s4.triage_pass_rate_switch


def pilot_pass_rate(rows: Sequence[S4LabelRow]) -> float:
    """Fraction of OK-labeled rows whose ``usable`` is True (pilot metric, §5.2)."""
    labeled = [r for r in rows if r.s4_status == S4Status.OK and r.labels is not None]
    if not labeled:
        return 0.0
    usable = sum(1 for r in labeled if r.labels is not None and r.labels.usable)
    return usable / len(labeled)


def s4_part_path(config: Config, slice_id: str) -> Path:
    """Return ``s4_labels/part-{slice}.parquet`` for a slice."""
    return Path(config.paths.s4_labels) / f"part-{slice_id}.parquet"


def write_s4_rows(
    rows: Sequence[S4LabelRow], config: Config, slice_id: str, *, mark_done: bool = True
) -> Path:
    """Atomically write S4 rows for a slice, then (optionally) the done marker.

    Follows the global write discipline (§3): parquet via ``*.tmp`` -> rename,
    done marker created only after the rename succeeds.
    """
    payload = [r.model_dump() for r in rows]
    out_path = atomic_write_parquet(payload, s4_part_path(config, slice_id))
    if mark_done:
        write_done_marker("s4", slice_id, config.paths.done)
    return out_path


async def process_slice(
    slice_id: str,
    config: Config,
    *,
    client: Optional[BaseS4Client] = None,
    audio_source: Optional[AudioSource] = None,
    worklist: Optional[Sequence[WorklistEntry]] = None,
    worklist_path: Optional[Path] = None,
    write: bool = True,
    mark_done: bool = True,
    two_pass: Optional[bool] = None,
) -> list[S4LabelRow]:
    """Label an entire worklist slice and (optionally) persist it.

    Steps: resolve entries (given ``worklist`` or loaded from the manifest) ->
    for each, decode audio, resample, base64, build messages, call the transport
    under a global semaphore with backoff/retry -> assemble rows (failures kept)
    -> atomic write ``part-{slice}.parquet`` + done marker.

    Args:
        slice_id: The slice task id.
        config: Pipeline config.
        client: Transport; defaults to :func:`get_s4_client` (mock unless a real
            key is present).
        audio_source: Audio resolver; defaults to :class:`RepackIndexAudioSource`.
        worklist: Explicit entries (skips manifest loading; used by tests).
        worklist_path: Optional worklist parquet override.
        write: Persist the parquet + done marker when True.
        mark_done: Write the done marker after the parquet lands (when ``write``).
        two_pass: Force single/two-pass for this slice. When ``None`` the decision
            comes from :func:`should_use_two_pass` (config-only). The pilot-driven
            override is supplied by :func:`run_s4_phase` after measuring the pilot
            slice's pass-rate (§5.2).

    Returns:
        The ordered :class:`S4LabelRow` list (also written when ``write``).
    """
    entries = list(worklist) if worklist is not None else load_slice_worklist(
        config, slice_id, worklist_path=worklist_path
    )
    owns_client = client is None
    owns_source = audio_source is None
    client = client or get_s4_client(config)
    audio_source = audio_source or RepackIndexAudioSource(config)
    resolved_two_pass = two_pass if two_pass is not None else should_use_two_pass(config)
    semaphore = asyncio.Semaphore(max(1, config.s4.max_concurrency))

    try:
        tasks = [
            _label_entry(
                entry,
                client=client,
                audio_source=audio_source,
                config=config,
                semaphore=semaphore,
                two_pass=resolved_two_pass,
            )
            for entry in entries
        ]
        rows = list(await asyncio.gather(*tasks)) if tasks else []
    finally:
        if owns_source:
            audio_source.close()
        if owns_client:
            await client.close()

    if write:
        write_s4_rows(rows, config, slice_id, mark_done=mark_done)
    return rows


def run_slice(
    slice_id: str,
    config: Config,
    *,
    client: Optional[BaseS4Client] = None,
    audio_source: Optional[AudioSource] = None,
    worklist: Optional[Sequence[WorklistEntry]] = None,
    worklist_path: Optional[Path] = None,
    write: bool = True,
    mark_done: bool = True,
) -> list[S4LabelRow]:
    """Synchronous wrapper around :func:`process_slice` (``asyncio.run``).

    Convenience for the dispatch worker loop, which drives tasks synchronously.
    """
    return asyncio.run(
        process_slice(
            slice_id,
            config,
            client=client,
            audio_source=audio_source,
            worklist=worklist,
            worklist_path=worklist_path,
            write=write,
            mark_done=mark_done,
        )
    )


def run_s4_phase(
    config: Config,
    *,
    slice_ids: Optional[Sequence[str]] = None,
    worklist_path: Optional[Path] = None,
    mark_done: bool = True,
) -> dict[str, Any]:
    """Drive S4 over all pending slices, wiring the §5.2 two-pass pilot decision.

    This is the single-process orchestrator that makes the pilot mechanism live
    (the reviewer flagged it as dead code otherwise). Flow:

      1. Enumerate slices from the worklist (or use ``slice_ids``); slice ids are
         in priority order, so slice 0 is the highest-priority "pilot".
      2. If ``config.s4.two_pass_triage`` is already on, honor it and skip the
         pilot (the decision is fixed). Otherwise label the first (pilot) slice
         single-pass, measure :func:`pilot_pass_rate`, and call
         :func:`should_use_two_pass` with it to decide the mode for the rest.
      3. Label the remaining slices with that decision.

    Idempotency is preserved: already-done slices are skipped (the pilot decision
    is recomputed from the pilot slice's persisted rows when it is already done).

    Args:
        config: Pipeline config.
        slice_ids: Explicit ordered slice ids; enumerated from the worklist when
            None.
        worklist_path: Optional worklist parquet override.
        mark_done: Write per-slice done markers.

    Returns:
        Summary dict: ``{"n_slices", "pilot_pass_rate", "two_pass", "completed"}``.
    """
    from ..common.io_utils import enumerate_slice_tasks, is_done
    from ..phase1.repack import WORKLIST_NAME

    wl_path = worklist_path or (config.paths.manifests / WORKLIST_NAME)
    ids = list(slice_ids) if slice_ids is not None else enumerate_slice_tasks(wl_path)
    if not ids:
        return {"n_slices": 0, "pilot_pass_rate": None, "two_pass": False, "completed": []}

    completed: list[str] = []
    # Config toggle wins outright -- no pilot needed.
    if config.s4.two_pass_triage:
        decided_two_pass = True
        pilot_rate: Optional[float] = None
    else:
        pilot_id = ids[0]
        if is_done("s4", pilot_id, config.paths.done):
            pilot_rows = _load_slice_rows(config, pilot_id)
        else:
            pilot_rows = run_slice(
                pilot_id, config, worklist_path=wl_path, two_pass=False,
                mark_done=mark_done,
            )
        completed.append(pilot_id)
        pilot_rate = pilot_pass_rate(pilot_rows)
        decided_two_pass = should_use_two_pass(config, pilot_rate)

    for sid in ids:
        if sid in completed:
            continue
        if is_done("s4", sid, config.paths.done):
            continue
        run_slice(
            sid, config, worklist_path=wl_path, two_pass=decided_two_pass, mark_done=mark_done
        )
        completed.append(sid)

    return {
        "n_slices": len(ids),
        "pilot_pass_rate": pilot_rate,
        "two_pass": decided_two_pass,
        "completed": completed,
    }


def _load_slice_rows(config: Config, slice_id: str) -> list[S4LabelRow]:
    """Re-read a completed slice's persisted S4 rows (for pilot-rate recompute)."""
    from ..common.io_utils import query_parquet

    path = s4_part_path(config, slice_id)
    if not path.exists():
        return []
    result = query_parquet("SELECT * FROM s4", s4=str(path))
    rows: list[S4LabelRow] = []
    for rec in result.df().to_dict(orient="records"):
        try:
            rows.append(S4LabelRow.model_validate(rec))
        except Exception:  # tolerate schema drift on recompute path
            continue
    return rows


__all__ = [
    # worklist / audio
    "WorklistEntry",
    "load_slice_worklist",
    "AudioSource",
    "DictAudioSource",
    "RepackIndexAudioSource",
    # prompt / encoding
    "build_system_prompt",
    "build_messages",
    "guided_json_schema",
    "encode_audio_datauri",
    "char_error_rate",
    # transport
    "S4TransportError",
    "label_with_retry",
    # slice
    "should_use_two_pass",
    "pilot_pass_rate",
    "s4_part_path",
    "write_s4_rows",
    "process_slice",
    "run_slice",
    "run_s4_phase",
]
