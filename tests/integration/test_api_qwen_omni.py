"""Real cloud-API (Qwen3-Omni via DashScope) integration tests -- marker: ``api``.

Exercises the *real* OpenAI-compatible DashScope transport that
:func:`emilia_pipeline.common.models.get_s4_client` selects when a live API key
is present. The request/response plumbing (base64 audio, guided-JSON,
retry/backoff) lives in :mod:`emilia_pipeline.phase2.s4_client`; here we drive it
end-to-end against the live endpoint with ONE tiny clip.

Skipped by default: no API key is exported in this environment, so each test
reports a clear skip reason rather than failing. When ``$DASHSCOPE_API_KEY``
(whatever ``config.s4.api_key_env`` names) is set, the tests assert that:

  * the response validates against the §5.3 guided-JSON schema
    (:class:`~emilia_pipeline.common.contracts.S4GuidedJSON`), and
  * ``temperature=0`` is deterministic: two identical requests yield identical
    labels (greedy decode, design §5.1).

These tests make real, billable API calls. Keep the clip tiny (2 s) and the
count minimal. Run with::

    pytest tests/integration -m api
"""

from __future__ import annotations

import asyncio

import pytest

from emilia_pipeline.common import synthesize
from emilia_pipeline.common.config import Config
from emilia_pipeline.common.contracts import EmotionPrimary, S4GuidedJSON, TextVerdict
from emilia_pipeline.common.models import OmniApiClient, get_s4_client
from emilia_pipeline.phase2 import s4_client

from .conftest import api_key_skip_reason

pytestmark = pytest.mark.api

_REFERENCE_TEXT = "今天的天气非常好，我们一起出去走走吧。"


def _require_key(config: Config) -> None:
    reason = api_key_skip_reason(config)
    if reason:
        pytest.skip(reason)


def _tiny_clip():
    """A single tiny 2 s synthetic clip (keeps the billable call minimal)."""
    arr = synthesize.synth_voice(
        synthesize.SynthSpec(duration_s=2.0, f0_hz=180.0, seed=7)
    )
    return arr, synthesize.DEFAULT_SR


async def _label_once(config: Config, client) -> S4GuidedJSON:
    """One real label call through the retry/backoff transport wrapper."""
    audio, sr = _tiny_clip()
    semaphore = asyncio.Semaphore(1)
    return await s4_client.label_with_retry(
        client,
        audio=audio,
        sample_rate=sr,
        reference_text=_REFERENCE_TEXT,
        clip_id="api-smoke",
        config=config,
        semaphore=semaphore,
    )


def test_real_api_returns_valid_schema(real_config: Config) -> None:
    _require_key(real_config)
    client = get_s4_client(real_config)
    assert isinstance(client, OmniApiClient), (
        "expected the real OmniApiClient when a key is present"
    )
    assert client.is_mock is False

    async def _run() -> S4GuidedJSON:
        try:
            return await _label_once(real_config, client)
        finally:
            await client.close()

    labels = asyncio.run(_run())

    # Validates against the §5.3 closed-vocabulary guided-JSON schema.
    assert isinstance(labels, S4GuidedJSON)
    assert S4GuidedJSON.model_validate(labels.model_dump()) == labels
    assert isinstance(labels.text_verdict, (TextVerdict, str))
    assert isinstance(labels.emotion.primary, (EmotionPrimary, str))
    assert 1 <= labels.emotion.intensity <= 5
    assert 0.0 <= labels.emotion.confidence <= 1.0
    assert 1 <= labels.prosody.expressiveness <= 5
    assert isinstance(labels.usable, bool)


def test_real_api_temperature0_deterministic(real_config: Config) -> None:
    _require_key(real_config)
    assert real_config.s4.temperature == 0.0, "greedy decode expected (§5.1)"
    client = get_s4_client(real_config)

    async def _run() -> tuple[S4GuidedJSON, S4GuidedJSON]:
        try:
            first = await _label_once(real_config, client)
            second = await _label_once(real_config, client)
            return first, second
        finally:
            await client.close()

    first, second = asyncio.run(_run())
    # Greedy (temperature=0) decode should be deterministic for identical input.
    assert first.model_dump() == second.model_dump()
