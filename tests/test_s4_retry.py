"""S4 transport retry/backoff tests against a fake mock transport.

Exercises ``label_with_retry`` directly: a 429 that succeeds after backoff, a
non-retryable 4xx that surfaces immediately, retry exhaustion, and the
slice-level guarantee that an exhausted clip is kept as a ``failed`` row (never
dropped). Backoff delays are zeroed so the tests are fast; no network is ever
touched (the client is ``is_mock`` so ``_call_transport`` uses its ``label``).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from emilia_pipeline.common.config import Config
from emilia_pipeline.common.contracts import (
    ContextLabel,
    EmotionLabel,
    LanguageLabel,
    ProsodyLabel,
    S4GuidedJSON,
    S4Status,
)
from emilia_pipeline.common.models import BaseS4Client
from emilia_pipeline.phase2 import s4_client as sc


def _valid_label() -> S4GuidedJSON:
    return S4GuidedJSON(
        text_verdict="match", text_fixed="x", text_punctuated="x.",
        emotion=EmotionLabel(primary="happy", secondary=None, intensity=3, confidence=0.8),
        prosody=ProsodyLabel(expressiveness=3, speaking_style="narration",
                             rhythm="varied", prominent_stress=False),
        context=ContextLabel(scenario="podcast", register="casual", summary="s"),
        language=LanguageLabel(primary="zh", code_switch=False, accent="standard"),
        paralinguistic=[], defects=[], usable=True,
    )


class _FlakyClient(BaseS4Client):
    """Mock-flagged client that raises retryable 429s ``fail_n`` times, then OK."""

    is_mock = True

    def __init__(self, fail_n: int, *, retryable: bool = True, status: int = 429) -> None:
        self.calls = 0
        self.fail_n = fail_n
        self._retryable = retryable
        self._status = status

    async def label(self, *, audio, sample_rate, reference_text, clip_id=""):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise sc.S4TransportError(
                f"HTTP {self._status}", retryable=self._retryable, status=self._status
            )
        return _valid_label()

    async def close(self) -> None:
        pass


def _fast_retry_config(base_config: Config, *, max_attempts: int = 5) -> Config:
    retry = base_config.s4.retry.model_copy(
        update={"base_delay_s": 0.0, "max_delay_s": 0.0, "max_attempts": max_attempts}
    )
    return base_config.model_copy(update={"s4": base_config.s4.model_copy(update={"retry": retry})})


async def _label(client, cfg):
    return await sc.label_with_retry(
        client, audio=np.zeros(2000, dtype="float32"), sample_rate=16000,
        reference_text="hi", clip_id="c", config=cfg, semaphore=asyncio.Semaphore(1),
    )


def test_429_then_success(base_config) -> None:
    cfg = _fast_retry_config(base_config, max_attempts=5)
    client = _FlakyClient(fail_n=2)
    label = asyncio.run(_label(client, cfg))
    assert label.usable is True
    assert client.calls == 3  # 2 failed attempts + 1 success


def test_non_retryable_raises_immediately(base_config) -> None:
    cfg = _fast_retry_config(base_config, max_attempts=5)
    client = _FlakyClient(fail_n=99, retryable=False, status=400)
    with pytest.raises(sc.S4TransportError) as exc:
        asyncio.run(_label(client, cfg))
    assert exc.value.status == 400
    assert client.calls == 1  # no retries for a non-retryable error


def test_retry_exhaustion_raises_last_error(base_config) -> None:
    cfg = _fast_retry_config(base_config, max_attempts=4)
    client = _FlakyClient(fail_n=99)
    with pytest.raises(sc.S4TransportError):
        asyncio.run(_label(client, cfg))
    assert client.calls == 4  # all attempts consumed


def test_backoff_delay_is_capped_and_nonnegative(base_config) -> None:
    retry = base_config.s4.retry.model_copy(update={"base_delay_s": 1.0, "max_delay_s": 5.0})
    cfg = base_config.model_copy(update={"s4": base_config.s4.model_copy(update={"retry": retry})})
    for attempt in range(6):
        d = sc._backoff_delay(attempt, cfg)
        assert 0.0 <= d <= 5.0  # full-jitter within the cap


def test_slice_keeps_failed_row_after_exhaustion(base_config) -> None:
    """A clip whose transport never recovers is written as a failed row, not dropped."""
    cfg = _fast_retry_config(base_config, max_attempts=2)
    entries = [sc.WorklistEntry(clip_id="c0", slice_id="00001", reference_text="t")]
    audio = {"c0": (np.ones(4000, dtype="float32") * 0.1, 24000)}

    rows = sc.run_slice(
        "00001", cfg, client=_FlakyClient(fail_n=99),
        audio_source=sc.DictAudioSource(audio), worklist=entries, write=False,
    )
    assert len(rows) == 1
    assert rows[0].s4_status == S4Status.FAILED
    assert rows[0].labels is None and rows[0].error


def test_two_pass_triage_toggle(base_config) -> None:
    # off by default; explicitly toggled on -> should_use_two_pass True.
    assert sc.should_use_two_pass(base_config) is False
    cfg_on = base_config.model_copy(
        update={"s4": base_config.s4.model_copy(update={"two_pass_triage": True})}
    )
    assert sc.should_use_two_pass(cfg_on) is True
    # pilot pass-rate below the switch also flips it.
    assert sc.should_use_two_pass(base_config, pilot_pass_rate=0.10) is True
    assert sc.should_use_two_pass(base_config, pilot_pass_rate=0.90) is False
