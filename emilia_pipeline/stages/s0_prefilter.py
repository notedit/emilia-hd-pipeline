"""Stage S0 - metadata prefilter (design doc v1.1 §4 S0).

Pure-metadata coarse filter that runs as the first, inline step of the Phase-1
fusion worker (design §4: "实现为 Phase 1 融合 worker 的第一步（内联），不单独跑一遍").
It never decodes audio; it only reads Emilia's per-clip sidecar JSON and applies
four cheap gates:

    | field       | condition                | rationale                          |
    |-------------|--------------------------|------------------------------------|
    | duration    | 3.0 <= d <= 20.0 s       | <3s no prosody, >20s emotion impure |
    | language    | == "zh"                  | Chinese only this cycle             |
    | dnsmos      | >= 3.2 (Emilia original) | coarser-than-Emilia (3.0) knife     |
    | text        | non-empty, >= 4 chars    | empty text cannot be verified       |

Following the global "numeric-in, judgment-out" principle, every gate outcome is
recorded on the row and the row is emitted regardless of pass/fail; the ``passed``
flag is the AND of the individual gates and downstream SQL is free to override
the thresholds. The worker short-circuits S1/S2/S3 for clips whose ``passed`` is
False, but the S0 row itself is always written.

Public API:
    * :func:`normalize_emilia_meta` -- tolerant field extraction from raw JSON.
    * :func:`evaluate_s0` -- the core pure function; one meta dict -> one
      :class:`S0PrefilterRow`.
    * :func:`prefilter_shard` -- batch convenience over an iterable of clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional

from ..common.config import Config, S0Config
from ..common.contracts import S0PrefilterRow

# ---------------------------------------------------------------------------
# Reject-reason tokens (stable strings; downstream may group on them).
# ---------------------------------------------------------------------------

REASON_DURATION_SHORT = "duration_short"
REASON_DURATION_LONG = "duration_long"
REASON_DURATION_MISSING = "duration_missing"
REASON_LANGUAGE = "language_not_target"
REASON_DNSMOS_LOW = "dnsmos_low"
REASON_DNSMOS_MISSING = "dnsmos_missing"
REASON_TEXT_SHORT = "text_short"

# Tolerant key aliases: Emilia-ZH sidecar JSON and the synthetic fixtures use
# short keys ("id", "text", "speaker", "duration", "dnsmos"); some exports use
# the fully-qualified "original_*" names. Accept both, first match wins.
_ID_KEYS = ("original_id", "id", "wav", "key")
_TEXT_KEYS = ("original_text", "text", "transcript")
_SPEAKER_KEYS = ("original_speaker", "speaker", "spk", "speaker_id")
_LANGUAGE_KEYS = ("original_language", "language", "lang")
_DURATION_KEYS = ("duration_s", "duration", "dur")
_DNSMOS_KEYS = ("original_dnsmos", "dnsmos", "dnsmos_ovrl")


@dataclass(frozen=True)
class NormalizedMeta:
    """Emilia sidecar metadata normalized to the S0 row's source fields."""

    original_id: str
    original_text: str
    original_speaker: str
    original_language: str
    duration_s: Optional[float]
    original_dnsmos: Optional[float]


def _first(meta: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Return the first present, non-None value among ``keys`` (else None)."""
    for key in keys:
        if key in meta and meta[key] is not None:
            return meta[key]
    return None


def _as_float(value: Any) -> Optional[float]:
    """Coerce to float, returning None on missing/unparseable values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str:
    """Coerce to a stripped string ("" for None)."""
    if value is None:
        return ""
    return str(value).strip()


def normalize_emilia_meta(meta: Mapping[str, Any]) -> NormalizedMeta:
    """Extract the S0-relevant source fields from a raw Emilia metadata dict.

    Tolerant of the short ("id"/"text"/...) and fully-qualified ("original_*")
    key conventions. Missing numeric fields become None (gated as failures);
    missing string fields become "".

    Args:
        meta: One clip's Emilia sidecar JSON, already parsed to a dict.

    Returns:
        A :class:`NormalizedMeta` with normalized, typed fields.
    """
    language = _as_str(_first(meta, _LANGUAGE_KEYS)).lower()
    return NormalizedMeta(
        original_id=_as_str(_first(meta, _ID_KEYS)),
        original_text=_as_str(_first(meta, _TEXT_KEYS)),
        original_speaker=_as_str(_first(meta, _SPEAKER_KEYS)),
        original_language=language,
        duration_s=_as_float(_first(meta, _DURATION_KEYS)),
        original_dnsmos=_as_float(_first(meta, _DNSMOS_KEYS)),
    )


def _text_char_count(text: str) -> int:
    """Count non-whitespace characters (CJK-safe: len counts code points)."""
    return len("".join(text.split()))


def evaluate_s0(
    meta: Mapping[str, Any],
    *,
    clip_id: str,
    shard: str,
    config: Config | S0Config,
) -> S0PrefilterRow:
    """Apply the four S0 gates to one clip's metadata and build its row.

    Pure function: no audio decode, no IO, deterministic. All gate booleans are
    recorded; ``passed`` is their conjunction. ``reject_reason`` is a
    ``;``-joined list of every failing gate's token (None when ``passed``).

    Args:
        meta: One clip's Emilia sidecar metadata dict.
        clip_id: The pipeline clip id (constructed by the worker).
        shard: The source shard token (Phase-1 task id).
        config: A :class:`Config` (its ``.s0`` is used) or an :class:`S0Config`.

    Returns:
        A fully-populated :class:`S0PrefilterRow`.
    """
    s0: S0Config = config.s0 if isinstance(config, Config) else config
    norm = normalize_emilia_meta(meta)

    reasons: List[str] = []

    # --- duration gate: 3.0 <= d <= 20.0 s ---
    if norm.duration_s is None:
        dur_ok = False
        reasons.append(REASON_DURATION_MISSING)
    elif norm.duration_s < s0.min_duration_s:
        dur_ok = False
        reasons.append(REASON_DURATION_SHORT)
    elif norm.duration_s > s0.max_duration_s:
        dur_ok = False
        reasons.append(REASON_DURATION_LONG)
    else:
        dur_ok = True

    # --- language gate: == config.s0.language ("zh") ---
    lang_ok = norm.original_language == s0.language.lower()
    if not lang_ok:
        reasons.append(REASON_LANGUAGE)

    # --- dnsmos gate: original dnsmos >= min_original_dnsmos ---
    if norm.original_dnsmos is None:
        # Cannot verify the coarse quality knife -> fail closed (recorded).
        dnsmos_ok = False
        reasons.append(REASON_DNSMOS_MISSING)
    elif norm.original_dnsmos < s0.min_original_dnsmos:
        dnsmos_ok = False
        reasons.append(REASON_DNSMOS_LOW)
    else:
        dnsmos_ok = True

    # --- text gate: non-empty and >= min_text_chars characters ---
    text_ok = _text_char_count(norm.original_text) >= s0.min_text_chars
    if not text_ok:
        reasons.append(REASON_TEXT_SHORT)

    passed = dur_ok and lang_ok and dnsmos_ok and text_ok
    reject_reason = None if passed else ";".join(reasons)

    return S0PrefilterRow(
        clip_id=clip_id,
        shard=shard,
        original_id=norm.original_id,
        original_speaker=norm.original_speaker,
        original_text=norm.original_text,
        original_language=norm.original_language,
        duration_s=float(norm.duration_s) if norm.duration_s is not None else 0.0,
        original_dnsmos=norm.original_dnsmos,
        dur_ok=dur_ok,
        lang_ok=lang_ok,
        dnsmos_ok=dnsmos_ok,
        text_ok=text_ok,
        passed=passed,
        reject_reason=reject_reason,
    )


def prefilter_shard(
    clips: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    shard: str,
    config: Config | S0Config,
) -> List[S0PrefilterRow]:
    """Evaluate S0 for every clip of a shard.

    Args:
        clips: Iterable of ``(clip_id, meta_dict)`` pairs, e.g. produced by the
            worker while iterating the source tar members in order.
        shard: The source shard token (also the Phase-1 task id).
        config: A :class:`Config` or :class:`S0Config`.

    Returns:
        One :class:`S0PrefilterRow` per input clip, in input order. Passing rows
        keep their input order so the worker can feed them straight into the
        S1/S2/S3 stages; failing rows are still returned (never dropped here).
    """
    return [
        evaluate_s0(meta, clip_id=clip_id, shard=shard, config=config)
        for clip_id, meta in clips
    ]


__all__ = [
    "REASON_DURATION_SHORT",
    "REASON_DURATION_LONG",
    "REASON_DURATION_MISSING",
    "REASON_LANGUAGE",
    "REASON_DNSMOS_LOW",
    "REASON_DNSMOS_MISSING",
    "REASON_TEXT_SHORT",
    "NormalizedMeta",
    "normalize_emilia_meta",
    "evaluate_s0",
    "prefilter_shard",
]
