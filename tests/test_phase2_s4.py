"""Phase-2 S4 client + dispatch tests. Zero GPU / key / network (all mocks)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from emilia_pipeline.common import synthesize
from emilia_pipeline.common.config import Config, load_config
from emilia_pipeline.common.contracts import S4LabelRow, S4Status
from emilia_pipeline.common.io_utils import is_done, query_parquet, parquet_glob
from emilia_pipeline.phase2 import dispatch, s4_client

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.yaml"


@pytest.fixture()
def mock_config(tmp_path: Path) -> Config:
    cfg = load_config(CONFIG_PATH)
    return cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(update={"use_mocks": True}),
            "paths": cfg.paths.model_copy(
                update={
                    "s4_labels": tmp_path / "s4",
                    "done": tmp_path / "done",
                    "failed": tmp_path / "failed",
                    "manifests": tmp_path / "manifests",
                }
            ),
        }
    )


def _entries(n: int = 3) -> list[s4_client.WorklistEntry]:
    return [
        s4_client.WorklistEntry(
            clip_id=f"c{i}", slice_id="00001", reference_text="合成测试文本内容"
        )
        for i in range(n)
    ]


def _audio_source(entries) -> s4_client.DictAudioSource:
    clips = {}
    for i, e in enumerate(entries):
        a = synthesize.synth_voice(synthesize.SynthSpec(duration_s=4.0, seed=i))
        clips[e.clip_id] = (a, synthesize.DEFAULT_SR)
    return s4_client.DictAudioSource(clips)


def test_process_slice_mock(mock_config: Config) -> None:
    entries = _entries(3)
    rows = s4_client.run_slice(
        "00001",
        mock_config,
        audio_source=_audio_source(entries),
        worklist=entries,
    )
    assert len(rows) == 3
    assert all(isinstance(r, S4LabelRow) for r in rows)
    assert all(r.s4_status == S4Status.OK for r in rows)
    assert all(r.labels is not None for r in rows)
    assert all(r.model == mock_config.s4.model for r in rows)
    # done marker + queryable parquet
    assert is_done("s4", "00001", mock_config.paths.done)
    res = query_parquet(
        "SELECT count(*) FROM s4", s4=parquet_glob(mock_config.paths.s4_labels)
    ).fetchall()
    assert res[0][0] == 3


def test_determinism(mock_config: Config) -> None:
    entries = _entries(2)
    r1 = s4_client.run_slice(
        "00001", mock_config, audio_source=_audio_source(entries),
        worklist=entries, write=False,
    )
    r2 = s4_client.run_slice(
        "00001", mock_config, audio_source=_audio_source(entries),
        worklist=entries, write=False,
    )
    assert [r.model_dump() for r in r1] == [r.model_dump() for r in r2]


def test_failed_row_kept_on_missing_audio(mock_config: Config) -> None:
    entries = _entries(2)
    src = s4_client.DictAudioSource({})  # nothing -> load raises
    rows = s4_client.run_slice(
        "00001", mock_config, audio_source=src, worklist=entries, write=False
    )
    assert len(rows) == 2
    assert all(r.s4_status == S4Status.FAILED for r in rows)
    assert all(r.labels is None and r.error for r in rows)


def test_two_pass_toggle(mock_config: Config) -> None:
    cfg = mock_config.model_copy(
        update={"s4": mock_config.s4.model_copy(update={"two_pass_triage": True})}
    )
    assert s4_client.should_use_two_pass(cfg)
    entries = _entries(3)
    rows = s4_client.run_slice(
        "00002", cfg, audio_source=_audio_source(entries), worklist=entries, write=False
    )
    assert len(rows) == 3  # some may be triaged out (labels None) but all OK
    assert all(r.s4_status == S4Status.OK for r in rows)


def test_cer_and_messages() -> None:
    assert s4_client.char_error_rate("abcd", "abcd") == 0.0
    assert s4_client.char_error_rate("abcd", "abxd") == 0.25
    assert s4_client.char_error_rate("", "") == 0.0
    b64, fmt = s4_client.encode_audio_datauri(
        synthesize.synth_voice(synthesize.SynthSpec(duration_s=2.0)),
        synthesize.DEFAULT_SR,
        16000,
    )
    assert fmt == "wav" and isinstance(b64, str) and len(b64) > 0
    # Default provider is Venus -> venus_multimodal_url content part.
    msgs = s4_client.build_messages(reference_text="x", audio_b64=b64)
    assert msgs[0]["role"] == "system"
    venus_parts = [p for p in msgs[1]["content"] if p["type"] == "venus_multimodal_url"]
    assert len(venus_parts) == 1
    vp = venus_parts[0]["venus_multimodal_url"]
    assert vp["mimeType"] == "audio/wav"
    assert vp["url"].startswith("data:audio/wav;base64,")
    assert any(p["type"] == "text" for p in msgs[1]["content"])
    # OpenAI provider path still emits the standard input_audio part.
    msgs_oa = s4_client.build_messages(reference_text="x", audio_b64=b64, provider="openai")
    assert any(p["type"] == "input_audio" for p in msgs_oa[1]["content"])
    assert "properties" in s4_client.guided_json_schema()


def test_load_slice_worklist_ordering(mock_config: Config) -> None:
    from emilia_pipeline.common.io_utils import atomic_write_parquet

    wl = mock_config.paths.manifests / "s4_worklist_v1.parquet"
    atomic_write_parquet(
        [
            {"clip_id": "b", "slice_id": "00001", "text": "t", "shard": "0", "offset": 2},
            {"clip_id": "a", "slice_id": "00001", "text": "t", "shard": "0", "offset": 1},
            {"clip_id": "z", "slice_id": "00002", "text": "t", "shard": "0", "offset": 0},
        ],
        wl,
    )
    entries = s4_client.load_slice_worklist(mock_config, "00001")
    assert [e.clip_id for e in entries] == ["a", "b"]  # ordered by offset


# ------------------------- dispatch tests -------------------------


def test_dispatch_pending_and_idempotent(mock_config: Config) -> None:
    all_tasks = ["s0", "s1", "s2"]
    processed: list[str] = []

    def proc(task_id: str) -> None:
        processed.append(task_id)

    done1 = dispatch.dispatch_and_run(all_tasks, "phase1", mock_config, proc)
    assert sorted(done1) == ["s0", "s1", "s2"]
    assert all(is_done("phase1", t, mock_config.paths.done) for t in all_tasks)

    # Re-run: everything already done -> nothing reprocessed.
    processed.clear()
    done2 = dispatch.dispatch_and_run(all_tasks, "phase1", mock_config, proc)
    assert done2 == [] and processed == []


def test_dispatch_failure_not_requeued_then_retried(mock_config: Config) -> None:
    from emilia_pipeline.common.io_utils import write_failed  # noqa: F401

    calls = {"n": 0}

    def flaky(task_id: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    done1 = dispatch.dispatch_and_run(["t0"], "stg", mock_config, flaky)
    assert done1 == []  # failed, no done marker
    assert (mock_config.paths.failed / "stg" / "t0.json").exists()

    # Rerun re-enqueues (no done marker) and now succeeds.
    done2 = dispatch.dispatch_and_run(["t0"], "stg", mock_config, flaky)
    assert done2 == ["t0"]
    assert is_done("stg", "t0", mock_config.paths.done)


def test_mp_queue_roundtrip() -> None:
    q = dispatch.MpTaskQueue()
    q.push(["a", "b", "c"])
    assert q.pop() == "a"
    assert q.pop() == "b"
    assert q.pop() == "c"
    assert q.pop() is None
