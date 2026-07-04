"""End-to-end tests for the S5 scoring + HF packaging + monitor tail (mocks only).

Builds a synthetic full-pipeline parquet fixture (S0..S4) with deterministic
mock models, runs S5 scoring/tiering/sampling, packages the export folder, and
snapshots the monitor -- all with zero GPU / API key / real data.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from emilia_pipeline.common import io_utils, synthesize
from emilia_pipeline.common.config import load_config
from emilia_pipeline.common.contracts import (
    ContextLabel,
    EmotionLabel,
    LanguageLabel,
    ProsodyLabel,
    S0PrefilterRow,
    S1AcousticsRow,
    S2ProsodyRow,
    S3SpeakerRow,
    S4GuidedJSON,
    S4LabelRow,
)
from emilia_pipeline.scoring import hf_package, monitor, s5_score

CONFIG_PATH = "/workspace/user_code/emilia-hd-pipeline/configs/pipeline_v1.yaml"


def _cfg(tmp_path: Path):
    cfg = load_config(CONFIG_PATH)
    cfg = cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"use_mocks": True})}
    )
    paths = cfg.paths.model_copy(
        update={
            "root": tmp_path,
            "source": tmp_path / "source",
            "s0_prefilter": tmp_path / "s0",
            "s1_acoustics": tmp_path / "s1",
            "s2_prosody": tmp_path / "s2",
            "s3_speaker_features": tmp_path / "s3",
            "s4_labels": tmp_path / "s4",
            "repacked": tmp_path / "repacked",
            "manifests": tmp_path / "manifests",
            "export": tmp_path / "export",
            "done": tmp_path / "done",
            "failed": tmp_path / "failed",
        }
    )
    return cfg.model_copy(update={"paths": paths})


def _mk_labels(emo: str, intensity: int, expr: int, tv="match", defects=None):
    return S4GuidedJSON(
        text_verdict=tv,
        text_fixed="hello",
        text_punctuated="hello.",
        emotion=EmotionLabel(primary=emo, secondary=None, intensity=intensity, confidence=0.8),
        prosody=ProsodyLabel(
            expressiveness=expr, speaking_style="storytelling", rhythm="dramatic",
            prominent_stress=True,
        ),
        context=ContextLabel(scenario="podcast", register="intimate", summary="s"),
        language=LanguageLabel(primary="zh", code_switch=False, accent="standard"),
        paralinguistic=[],
        defects=defects or [],
        usable=True,
    )


def _write_fixture(cfg, clips):
    """clips: list of (clip_id, speaker, emo, intensity, expr, pq, pros, tv, defects, verdict)."""
    s0r, s1r, s2r, s3r, s4r = [], [], [], [], []
    for (cid, spk, emo, inten, expr, pq, pros, tv, defects, verdict) in clips:
        s0r.append(S0PrefilterRow(clip_id=cid, shard="00000", original_id="o" + cid,
            original_speaker=spk, original_text="hello there friend",
            original_language="zh", duration_s=6.0, original_dnsmos=3.4,
            dur_ok=True, lang_ok=True, dnsmos_ok=True, text_ok=True, passed=True,
            reject_reason=None).model_dump())
        s1r.append(S1AcousticsRow(clip_id=cid, shard="00000", aes_pq=pq, aes_pc=1.9,
            aes_ce=7.0, aes_cu=7.0, dnsmos_sig=3.9, dnsmos_bak=4.1, dnsmos_ovrl=3.7,
            snr_db=28.0, clipping_ratio=0.0, bandwidth_hz=11000.0, loudness_lufs=-19.3,
            passed=True, reject_reason=None).model_dump())
        s2r.append(S2ProsodyRow(clip_id=cid, shard="00000", f0_mean_hz=200.0,
            f0_std_st=4.0, f0_range_st=14.0, energy_std_db=6.0, speech_rate_cps=4.9,
            rate_var_cv=0.3, pause_count=2, pause_total_ms=600.0,
            f0_tracker_confidence=0.9).model_dump())
        s3r.append(S3SpeakerRow(clip_id=cid, shard="00000", original_speaker=spk,
            emb_file="emb-00000.npy", emb_row=0, gender_pred="female", n_windows=9,
            mean_win_cos=0.91, min_win_cos=0.83, f0_stability=0.94, verdict=verdict,
            intrusion_span_ms=None, trimmed=False, trimmed_duration_s=None,
            trim_start_s=None, trim_end_s=None).model_dump())
        s4r.append(S4LabelRow(clip_id=cid, slice_id="000", model="qwen3-omni-30b-a3b-instruct",
            prompt_version="v3.1", s4_status="ok", cer_vs_original=0.0,
            labels=_mk_labels(emo, inten, expr, tv=tv, defects=defects)).model_dump())
    io_utils.atomic_write_parquet(s0r, cfg.paths.s0_prefilter / "part-00000.parquet")
    io_utils.atomic_write_parquet(s1r, cfg.paths.s1_acoustics / "part-00000.parquet")
    io_utils.atomic_write_parquet(s2r, cfg.paths.s2_prosody / "part-00000.parquet")
    io_utils.atomic_write_parquet(s3r, cfg.paths.s3_speaker_features / "part-00000.parquet")
    io_utils.atomic_write_parquet(s4r, cfg.paths.s4_labels / "part-000.parquet")


_CLIPS = [
    ("c1", "SPK_A", "sad", 4, 5, 7.9, 0.95, "match", None, "single"),
    ("c2", "SPK_A", "neutral", 2, 2, 7.2, 0.50, "match", None, "single"),
    ("c3", "SPK_B", "happy", 3, 4, 7.4, 0.60, "match", None, "intruded_trimmed"),
    ("c4", "SPK_B", "excited", 5, 5, 7.0, 0.30, "broken", None, "single"),
    ("c5", "SPK_C", "angry", 4, 4, 7.6, 0.70, "match", ["truncated_tail"], "single"),
    ("c6", "SPK_C", "fearful", 3, 3, 7.1, 0.40, "match", None, "overlap_rejected"),
]


# ---------------------------------------------------------------------------
# S5 scoring
# ---------------------------------------------------------------------------


def test_hard_constraints_reject_correct_rows():
    cfg = load_config(CONFIG_PATH)
    rows = s5_score.score_rows(_joined_rows(), cfg)
    by_id = {r.joined["clip_id"]: r for r in rows}
    assert by_id["c4"].reject_reason == s5_score.REASON_TEXT_BROKEN
    assert by_id["c5"].reject_reason == s5_score.REASON_TRUNCATED
    assert by_id["c6"].reject_reason == s5_score.REASON_VERDICT
    # Rejected rows carry no tier and zero score.
    for cid in ("c4", "c5", "c6"):
        assert by_id[cid].tier is None
        assert by_id[cid].selection_score == 0.0


def test_tier_ab_split_is_by_expressiveness_not_emotion():
    """Tier A/B split on expressiveness (design §7), decoupled from neutral gate.

    The old behavior keyed B on neutral emotion; this pins the corrected rule:
      * a non-neutral clip with LOW expressiveness -> Tier-B (not A),
      * a neutral clip with HIGH expressiveness -> Tier-A (not forced B),
      * neutral is only a Tier-S exclusion, never the A/B discriminator.
    """
    cfg = load_config(CONFIG_PATH)
    assert cfg.s5.tier_b_max_expressiveness == 2
    # (cid, spk, emo, intensity, expr, pq, pros, tv, defects, verdict)
    clips = [
        ("hi", "S1", "sad", 4, 5, 7.9, 0.9, "match", None, "single"),      # top -> S
        ("lo_expr_emo", "S2", "happy", 4, 1, 7.0, 0.2, "match", None, "single"),  # non-neutral, expr=1 -> B
        ("neu_hi_expr", "S3", "neutral", 2, 5, 7.5, 0.5, "match", None, "single"),  # neutral, expr=5 -> A
        ("mid", "S4", "angry", 3, 3, 7.2, 0.4, "match", None, "single"),   # expr=3 -> A
    ]
    rows = {r.joined["clip_id"]: r for r in s5_score.score_rows(_joined_from(clips), cfg)}
    assert rows["hi"].tier.value == "S"
    # non-neutral but expressiveness<=2 -> Tier-B (old code wrongly said A)
    assert rows["lo_expr_emo"].tier.value == "B"
    # neutral but expressiveness>2 -> Tier-A (old code wrongly forced B)
    assert rows["neu_hi_expr"].tier.value == "A"
    assert rows["mid"].tier.value == "A"


def test_selection_score_monotone_and_bounded():
    cfg = load_config(CONFIG_PATH)
    rows = {r.joined["clip_id"]: r for r in s5_score.score_rows(_joined_rows(), cfg)}
    # c1 tops pq/prosody/expr*intensity; ce is constant across the fixture so the
    # ce term is 0 for everyone -> the achievable max is 0.35+0.25+0.30 = 0.90.
    kept = [r for r in rows.values() if r.reject_reason is None]
    scores = [r.selection_score for r in kept]
    assert max(scores) == pytest.approx(0.90)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert rows["c1"].selection_score == max(scores)


def test_stratified_sampling_caps_head_speaker():
    cfg = load_config(CONFIG_PATH)
    cfg = cfg.model_copy(update={"s5": cfg.s5.model_copy(update={"max_clips_per_speaker": 1})})
    rows = {r.joined["clip_id"]: r for r in s5_score.score_rows(_joined_rows(), cfg)}
    # SPK_A has two survivors (c1, c2); cap=1 keeps the higher-scored one.
    spk_a_kept = [r for r in rows.values()
                  if r.joined["original_speaker"] == "SPK_A" and r.reject_reason is None]
    assert len(spk_a_kept) == 1
    assert spk_a_kept[0].joined["clip_id"] == "c1"
    # The dropped one is marked with the quota reason, not deleted.
    assert rows["c2"].reject_reason == s5_score.REASON_SPEAKER_QUOTA


def test_run_s5_writes_flat_parquet_and_json(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg, _CLIPS)
    result = s5_score.run_s5(cfg, write=True)
    assert result.flat_parquet_path.exists()
    assert io_utils.is_done("s5", "all", cfg.paths.done)
    # Only kept (tiered) clips get JSON files.
    json_names = sorted(p.stem for p in result.json_dir.glob("*.json"))
    assert json_names == ["c1", "c2", "c3"]
    # Published JSON validates back into a MetaRecord.
    from emilia_pipeline.common.contracts import MetaRecord
    rec = MetaRecord.model_validate(json.loads((result.json_dir / "c1.json").read_text()))
    assert rec.selection.tier == "S"
    assert rec.omni_labels.emotion.primary == "sad"


def test_flat_parquet_queryable(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg, _CLIPS)
    s5_score.run_s5(cfg, write=True)
    glob = io_utils.parquet_glob(s5_score.s5_flat_parquet_path(cfg).parent)
    res = io_utils.query_parquet(
        "SELECT count(*) FROM m WHERE selection_tier IS NOT NULL", m=glob
    )
    assert res.fetchall()[0][0] == 3


# ---------------------------------------------------------------------------
# HF packaging
# ---------------------------------------------------------------------------


def test_package_export_meta_only(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg, _CLIPS)
    s5_score.run_s5(cfg, write=True)
    result = hf_package.package_export(cfg, target_shard_bytes=10_000_000)
    assert result.n_clips == 3
    assert result.n_with_audio == 0  # no repacked audio in this fixture
    assert len(result.shard_paths) >= 1
    assert result.manifest_path.exists()
    # Each shard contains {clip_id}.json members.
    names = set()
    for sp in result.shard_paths:
        with tarfile.open(sp) as tar:
            names.update(tar.getnames())
    assert {"c1.json", "c2.json", "c3.json"} <= names
    assert io_utils.is_done("pack", "all", cfg.paths.done)


def test_package_export_with_audio(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg, _CLIPS)
    s5_score.run_s5(cfg, write=True)
    # Build a repacked tar + index for c1..c3.
    cfg.paths.repacked.mkdir(parents=True, exist_ok=True)
    audio = synthesize.synth_voice(synthesize.SynthSpec(duration_s=4.0))
    flac = synthesize.encode_flac(audio, 24000)
    tar_path = cfg.paths.repacked / "shard-00000.tar"
    with tarfile.open(tar_path, "w") as tar:
        for cid in ("c1", "c2", "c3"):
            info = tarfile.TarInfo(name=f"{cid}.flac")
            info.size = len(flac)
            import io as _io
            tar.addfile(info, _io.BytesIO(flac))
    io_utils.atomic_write_parquet(
        [{"clip_id": c, "shard": "shard-00000", "member": f"{c}.flac"}
         for c in ("c1", "c2", "c3")],
        cfg.paths.repacked / "repack_index.parquet",
    )
    result = hf_package.package_export(cfg, target_shard_bytes=10_000_000)
    assert result.n_with_audio == 3
    names = set()
    for sp in result.shard_paths:
        with tarfile.open(sp) as tar:
            names.update(tar.getnames())
    assert {"c1.flac", "c1.json"} <= names


def test_upload_skips_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = _cfg(tmp_path)
    _write_fixture(cfg, _CLIPS)
    s5_score.run_s5(cfg, write=True)
    result = hf_package.upload_to_hf(cfg, "org/emilia-expressive-zh")
    assert result.uploaded is False
    assert "HF_TOKEN" in (result.upload_skipped_reason or "")
    assert result.manifest_path.exists()


def test_shuffle_is_deterministic():
    entries = [hf_package.PackEntry(clip_id=f"c{i}", tier="A", meta_json={}) for i in range(20)]
    a = [e.clip_id for e in hf_package.shuffle_entries(entries, seed=7)]
    b = [e.clip_id for e in hf_package.shuffle_entries(entries, seed=7)]
    assert a == b


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def test_monitor_snapshot(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg, _CLIPS)
    s5_score.run_s5(cfg, write=True)
    # Mark a couple of done markers + a failed 429 record so counts are non-trivial.
    io_utils.write_done_marker("s0", "00000", cfg.paths.done)
    io_utils.write_failed("s4", "000", cfg.paths.failed, "HTTP 429 Too Many Requests")

    snap = monitor.build_snapshot(cfg)
    d = snap.to_dict()
    assert d["timestamp"]
    stage_names = {s["stage"] for s in d["stages"]}
    assert {"s0", "s1", "s2", "s3", "s4", "s5", "pack"} <= stage_names
    # Emotion distribution reflects the fixture.
    assert snap.emotion_distribution.get("sad", 0) >= 1
    # Verdict distribution reflects S3 rows.
    assert snap.verdict_distribution.get("single", 0) >= 1
    # S4 cost snapshot present with a positive cost estimate + detected 429.
    assert snap.s4_cost is not None
    assert snap.s4_cost.labels_ok == 6
    assert snap.s4_cost.est_cost_usd > 0.0
    assert snap.s4_cost.rate_limit_failures >= 1
    # Tier distribution comes from the flat parquet.
    assert snap.tier_distribution.get("S", 0) == 1


def test_monitor_write_and_format(tmp_path):
    cfg = _cfg(tmp_path)
    _write_fixture(cfg, _CLIPS)
    s5_score.run_s5(cfg, write=True)
    path = monitor.write_snapshot(cfg)
    assert path.exists()
    assert (cfg.paths.root / "monitor" / "latest.json").exists()
    text = monitor.format_snapshot(monitor.build_snapshot(cfg))
    assert "pipeline snapshot" in text
    assert "s4:" in text


def test_monitor_empty_pipeline(tmp_path):
    cfg = _cfg(tmp_path)
    # No parquet at all -> snapshot still builds with empty distributions.
    snap = monitor.build_snapshot(cfg)
    assert snap.emotion_distribution == {}
    assert snap.verdict_distribution == {}
    assert snap.s4_cost.labels_ok == 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _joined_from(clips):
    """Build flat joined dicts directly (no parquet) for the given clip tuples."""
    rows = []
    for (cid, spk, emo, inten, expr, pq, pros, tv, defects, verdict) in clips:
        rows.append(dict(
            clip_id=cid, shard="00000", original_id="o" + cid, original_speaker=spk,
            original_text="hello there friend", original_language="zh", duration_s=6.0,
            original_dnsmos=3.4, aes_pq=pq, aes_pc=1.9, aes_ce=7.0, aes_cu=7.0,
            dnsmos_sig=3.9, dnsmos_bak=4.1, dnsmos_ovrl=3.7, snr_db=28.0,
            clipping_ratio=0.0, bandwidth_hz=11000.0, loudness_lufs=-19.3,
            f0_mean_hz=200.0, f0_std_st=4.0,
            f0_range_st=14.0, energy_std_db=6.0, speech_rate_cps=4.9, rate_var_cv=0.3,
            pause_count=2, pause_total_ms=600.0, f0_tracker_confidence=0.9,
            prosody_dsp_score=pros, emb_file="emb-00000.npy", emb_row=0,
            gender_pred="female", n_windows=9, mean_win_cos=0.91, min_win_cos=0.83,
            f0_stability=0.94, verdict=verdict, intrusion_span_ms=None, trimmed=False,
            trimmed_duration_s=None, s4_model="qwen3-omni-30b-a3b-instruct",
            s4_prompt_version="v3.1", s4_status="ok", cer_vs_original=0.0,
            text_verdict=tv, text_fixed="hello", text_punctuated="hello.",
            emotion_primary=emo, emotion_secondary=None, emotion_intensity=inten,
            emotion_confidence=0.8, prosody_expressiveness=expr,
            prosody_speaking_style="storytelling", prosody_rhythm="dramatic",
            prosody_prominent_stress=True, context_scenario="podcast",
            context_register="intimate", context_summary="s", language_primary="zh",
            language_code_switch=False, language_accent="standard", paralinguistic=[],
            defects=defects or [], usable=True,
        ))
    return rows


def _joined_rows():
    """Build the flat joined dicts directly (no parquet) for pure scoring tests."""
    return _joined_from(_CLIPS)
