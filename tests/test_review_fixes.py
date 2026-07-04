"""Regression tests for the adversarial-review fixes (mock tier, no GPU/key).

Each test pins a specific defect the reviewer found, so the fix cannot silently
regress:

  #1 intruded_trimmed clips ship TRIMMED audio whose length matches the meta.
  #2 prosody_dsp_score is GLOBAL (population z-score), not per-shard.
  #4 published audio.loudness_lufs is a real value threaded from S1, not 0.0.
  #6 degraded_pass is eligible for labeling (repack survivor set), separate from
     the S5 publish gate.
  #7 the §5.2 two-pass pilot decision is wired (run_s4_phase), not dead code.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pytest

from emilia_pipeline.common import audio as audio_utils
from emilia_pipeline.common import io_utils, synthesize
from emilia_pipeline.common.config import load_config
from emilia_pipeline.common.contracts import (
    S0PrefilterRow,
    S1AcousticsRow,
    S2ProsodyRow,
    S3SpeakerRow,
)
from emilia_pipeline.common.prosody_sql import prosody_dsp_score_sql
from emilia_pipeline.phase1 import repack as repack_mod
from emilia_pipeline.scoring import s5_score

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.yaml"


def _cfg(tmp_path: Path):
    cfg = load_config(CONFIG_PATH)
    cfg = cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"use_mocks": True})}
    )
    p = cfg.paths
    return cfg.model_copy(
        update={
            "paths": p.model_copy(
                update={
                    "source": tmp_path / "source",
                    "s0_prefilter": tmp_path / "s0",
                    "s1_acoustics": tmp_path / "s1",
                    "s2_prosody": tmp_path / "s2",
                    "s3_speaker_features": tmp_path / "s3",
                    "s3_speaker_embeddings": tmp_path / "s3emb",
                    "s4_labels": tmp_path / "s4",
                    "repacked": tmp_path / "repacked",
                    "manifests": tmp_path / "manifests",
                    "export": tmp_path / "export",
                    "done": tmp_path / "done",
                    "failed": tmp_path / "failed",
                }
            )
        }
    )


# ---------------------------------------------------------------------------
# #1 -- trim is actually applied to the released audio
# ---------------------------------------------------------------------------


def test_intruded_trimmed_audio_is_actually_trimmed(tmp_path):
    """A trimmed clip's repacked audio length == its kept [start,end) span."""
    cfg = _cfg(tmp_path)
    cfg.paths.source.mkdir(parents=True, exist_ok=True)
    sr = cfg.runtime.audio_sample_rate
    full = synthesize.synth_voice(synthesize.SynthSpec(duration_s=10.0, seed=1))
    src_tar = cfg.paths.source / "00000.tar"
    with tarfile.open(src_tar, "w") as tar:
        flac = audio_utils.encode_audio(full, sr)
        info = tarfile.TarInfo("clipT.flac")
        info.size = len(flac)
        import io as _io

        tar.addfile(info, _io.BytesIO(flac))

    # Keep [2.0, 8.0) -> 6.0 s expected in the output shard.
    keep_start, keep_end = 2.0, 8.0
    for path, rows in _trimmed_fixture_rows(keep_start, keep_end):
        io_utils.atomic_write_parquet([r.model_dump() for r in rows], _stage_path(cfg, path))

    slice_rows = [
        {
            "clip_id": "clipT",
            "source_shard": "00000",
            "original_text": "x",
            "original_speaker": "SPK",
            "duration_s": keep_end - keep_start,
            "priority": 1.0,
            "priority_rank": 0,
            "slice_id": "00000",
            "trimmed": True,
            "trim_start_s": keep_start,
            "trim_end_s": keep_end,
        }
    ]
    repack_mod.repack_slice("00000", slice_rows, cfg)

    out_tar = cfg.paths.repacked / "shard-00000.tar"
    with tarfile.open(out_tar) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".flac"))
        data = tar.extractfile(member).read()
    arr, out_sr = audio_utils.decode_bytes(data)
    got_dur = len(arr) / out_sr
    assert got_dur == pytest.approx(keep_end - keep_start, abs=0.05)
    # And it is genuinely shorter than the 10 s source (the intrusion was cut).
    assert got_dur < 9.0


def _stage_path(cfg, key):
    return {
        "s0": cfg.paths.s0_prefilter / "part-00000.parquet",
        "s1": cfg.paths.s1_acoustics / "part-00000.parquet",
        "s2": cfg.paths.s2_prosody / "part-00000.parquet",
        "s3": cfg.paths.s3_speaker_features / "part-00000.parquet",
    }[key]


def _trimmed_fixture_rows(keep_start, keep_end):
    dur = keep_end - keep_start
    s0 = [S0PrefilterRow(clip_id="clipT", shard="00000", original_id="o", original_speaker="SPK",
        original_text="x", original_language="zh", duration_s=dur, original_dnsmos=3.4,
        dur_ok=True, lang_ok=True, dnsmos_ok=True, text_ok=True, passed=True, reject_reason=None)]
    s1 = [S1AcousticsRow(clip_id="clipT", shard="00000", aes_pq=7.5, aes_pc=1.9, aes_ce=7.0,
        aes_cu=7.0, dnsmos_sig=3.9, dnsmos_bak=4.1, dnsmos_ovrl=3.7, snr_db=28.0,
        clipping_ratio=0.0, bandwidth_hz=11000.0, loudness_lufs=-19.0, passed=True, reject_reason=None)]
    s2 = [S2ProsodyRow(clip_id="clipT", shard="00000", f0_mean_hz=200.0, f0_std_st=4.0,
        f0_range_st=14.0, energy_std_db=6.0, speech_rate_cps=4.9, rate_var_cv=0.3,
        pause_count=2, pause_total_ms=600.0, f0_tracker_confidence=0.9)]
    s3 = [S3SpeakerRow(clip_id="clipT", shard="00000", original_speaker="SPK",
        emb_file="emb-00000.npy", emb_row=0, gender_pred="female", n_windows=9,
        mean_win_cos=0.9, min_win_cos=0.6, f0_stability=0.9, verdict="intruded_trimmed",
        intrusion_span_ms=2000.0, trimmed=True, trimmed_duration_s=dur,
        trim_start_s=keep_start, trim_end_s=keep_end)]
    return [("s0", s0), ("s1", s1), ("s2", s2), ("s3", s3)]


# ---------------------------------------------------------------------------
# #2 -- prosody_dsp_score is global, not per-shard
# ---------------------------------------------------------------------------


def test_prosody_score_sql_is_population_relative():
    """The SQL z-score normalizes over the whole result set (single row -> 0)."""
    import duckdb

    cfg = load_config(CONFIG_PATH)
    expr = prosody_dsp_score_sql(cfg.s2.z_weights)  # bare columns
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t(f0_std_st DOUBLE, f0_range_st DOUBLE, energy_std_db DOUBLE, "
        "speech_rate_cps DOUBLE, rate_var_cv DOUBLE, pause_count DOUBLE)"
    )
    # Population of 3 with spread -> nonzero, distinct z-scores.
    con.executemany(
        "INSERT INTO t VALUES (?,?,?,?,?,?)",
        [(1, 1, 1, 1, 1, 1), (5, 5, 5, 5, 5, 5), (9, 9, 9, 9, 9, 9)],
    )
    scores = [r[0] for r in con.execute(f"SELECT {expr} FROM t ORDER BY f0_std_st").fetchall()]
    assert scores[0] < scores[1] < scores[2]          # monotone with the metrics
    assert scores[1] == pytest.approx(0.0, abs=1e-9)  # the mean row -> 0

    # A single-row population is all-zero (matches numpy std==0 -> 0 reference).
    con.execute("DELETE FROM t")
    con.execute("INSERT INTO t VALUES (7,7,7,7,7,7)")
    solo = con.execute(f"SELECT {expr} FROM t").fetchall()[0][0]
    assert solo == pytest.approx(0.0, abs=1e-9)


def test_s2_row_does_not_persist_score():
    """The per-shard score column is gone from the stored contract (#2)."""
    assert "prosody_dsp_score" not in S2ProsodyRow.model_fields


# ---------------------------------------------------------------------------
# #4 -- loudness threaded to the published meta
# ---------------------------------------------------------------------------


def test_published_loudness_is_not_zero(tmp_path):
    cfg = _cfg(tmp_path)
    row = {
        "clip_id": "c", "shard": "s", "duration_s": 6.0, "loudness_lufs": -19.3,
        "aes_pq": 7.5, "aes_pc": 1.9, "aes_ce": 7.0, "aes_cu": 7.0,
        "dnsmos_sig": 3.9, "dnsmos_bak": 4.1, "dnsmos_ovrl": 3.7,
        "snr_db": 28.0, "clipping_ratio": 0.0, "bandwidth_hz": 11000.0,
        "f0_mean_hz": 200.0, "f0_std_st": 4.0, "f0_range_st": 14.0,
        "energy_std_db": 6.0, "speech_rate_cps": 4.9, "rate_var_cv": 0.3,
        "pause_count": 2, "pause_total_ms": 600.0, "f0_tracker_confidence": 0.9,
        "prosody_dsp_score": 0.5, "emb_file": "e.npy", "emb_row": 0,
        "gender_pred": "female", "n_windows": 9, "mean_win_cos": 0.9,
        "min_win_cos": 0.8, "f0_stability": 0.9, "verdict": "single",
        "intrusion_span_ms": None, "trimmed": False, "trimmed_duration_s": None,
    }
    rec = s5_score.build_meta_record(row, cfg, selection_score=0.5, tier=None, reject_reason=None)
    assert rec.audio.loudness_lufs == pytest.approx(-19.3)


def test_s1_row_stores_loudness():
    assert "loudness_lufs" in S1AcousticsRow.model_fields


# ---------------------------------------------------------------------------
# #6 -- degraded_pass is labeled (repack survivor set) but publish-gated in S5
# ---------------------------------------------------------------------------


def test_degraded_pass_labeled_but_not_published():
    cfg = load_config(CONFIG_PATH)
    assert "degraded_pass" in cfg.repack.survivor_verdicts       # enters labeling
    assert "degraded_pass" not in cfg.s5.allowed_verdicts        # not published
    # S5 hard-constraint rejects degraded_pass with the verdict reason.
    row = {"text_verdict": "match", "defects": [], "verdict": "degraded_pass"}
    assert s5_score.hard_constraint_reason(row, cfg) == s5_score.REASON_VERDICT


# ---------------------------------------------------------------------------
# #7 -- the two-pass pilot decision is wired
# ---------------------------------------------------------------------------


def test_run_s4_phase_uses_pilot_pass_rate(tmp_path, monkeypatch):
    """run_s4_phase measures the pilot slice and feeds it to the two-pass switch."""
    from emilia_pipeline.phase2 import s4_client

    cfg = _cfg(tmp_path)
    # Force a low pilot pass-rate so the switch flips to two-pass for later slices.
    calls = {"two_pass_flags": []}

    def fake_run_slice(slice_id, config, *, worklist_path=None, two_pass=None, mark_done=True):
        calls["two_pass_flags"].append((slice_id, two_pass))
        # Return rows with usable=... shaped so pilot_pass_rate is low (<0.6).
        from emilia_pipeline.common.contracts import S4LabelRow

        return [
            S4LabelRow(clip_id=f"{slice_id}-{i}", slice_id=slice_id,
                       model="m", prompt_version="v", s4_status="failed", labels=None)
            for i in range(3)
        ]

    monkeypatch.setattr(s4_client, "run_slice", fake_run_slice)
    summary = s4_client.run_s4_phase(cfg, slice_ids=["00000", "00001", "00002"])
    # Pilot ran single-pass; low pass-rate -> the rest run two-pass.
    assert calls["two_pass_flags"][0] == ("00000", False)
    assert summary["two_pass"] is True
    assert all(flag is True for _, flag in calls["two_pass_flags"][1:])
    assert summary["pilot_pass_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Venus proxy adaptation -- the real S4 endpoint is the internal Venus LLM
# proxy with a proprietary venus_multimodal_url audio content type (NOT the
# standard OpenAI input_audio). Pin the message contract + config defaults.
# ---------------------------------------------------------------------------


def test_venus_message_contract():
    """Default provider builds the Venus venus_multimodal_url part, audio first."""
    from emilia_pipeline.phase2 import s4_client

    b64, fmt = s4_client.encode_audio_datauri(
        np.zeros(16000, dtype=np.float32), 16000, 16000
    )
    msgs = s4_client.build_messages(reference_text="你好", audio_b64=b64, audio_format=fmt)
    user = msgs[1]
    assert [p["type"] for p in user["content"]] == ["venus_multimodal_url", "text"]
    vp = user["content"][0]["venus_multimodal_url"]
    assert set(vp) == {"mimeType", "url"}
    assert vp["mimeType"] == "audio/wav"
    assert vp["url"] == f"data:audio/wav;base64,{b64}"


def test_openai_provider_still_supported():
    """provider='openai' falls back to the standard input_audio content part."""
    from emilia_pipeline.phase2 import s4_client

    b64, _ = s4_client.encode_audio_datauri(np.zeros(8000, dtype=np.float32), 16000, 16000)
    msgs = s4_client.build_messages(reference_text="x", audio_b64=b64, provider="openai")
    types = [p["type"] for p in msgs[1]["content"]]
    assert "input_audio" in types and "venus_multimodal_url" not in types


def test_s4_config_defaults_target_venus():
    cfg = load_config(CONFIG_PATH)
    assert cfg.s4.provider == "venus"
    assert cfg.s4.model == "server:272349"
    assert "venus.oa.com" in cfg.s4.base_url
    assert cfg.s4.api_key_env == "OPENAI_API_KEY"
    # Venus path validates JSON client-side, not via response_format.
    assert cfg.s4.use_guided_json is False


def test_real_client_is_omni_api_client(monkeypatch):
    """With a key present (and mocks off), the factory yields the real OmniApiClient."""
    from emilia_pipeline.common import models

    cfg = load_config(CONFIG_PATH)
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"use_mocks": False})})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    client = models.get_s4_client(cfg)
    assert isinstance(client, models.OmniApiClient)
    assert client.is_mock is False
    assert client.provider == "venus"
    assert client.model == "server:272349"
