"""Tests for the Phase-1 -> HuggingFace short-circuit release (mock tier).

Exercises :mod:`emilia_pipeline.scoring.phase1_hf` end to end with no GPU / key /
network: write synthetic S0-S3 stage parquet + a source tar, then package the
filtered subset and assert the shards, meta JSON, manifest, card, and the
graceful upload no-op.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

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
from emilia_pipeline.scoring import phase1_hf

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
                    "repacked": tmp_path / "repacked",
                    "manifests": tmp_path / "manifests",
                    "export": tmp_path / "export",
                    "done": tmp_path / "done",
                    "failed": tmp_path / "failed",
                }
            )
        }
    )


def _write_source_tar(cfg, clip_ids, dur_s=8.0):
    """Write a source tar with one synthetic voice clip per id."""
    cfg.paths.source.mkdir(parents=True, exist_ok=True)
    sr = cfg.runtime.audio_sample_rate
    with tarfile.open(cfg.paths.source / "00000.tar", "w") as tar:
        for i, cid in enumerate(clip_ids):
            arr = synthesize.synth_voice(
                synthesize.SynthSpec(duration_s=dur_s, seed=i + 1)
            )
            flac = audio_utils.encode_audio(arr, sr)
            info = tarfile.TarInfo(f"{cid}.flac")
            info.size = len(flac)
            tar.addfile(info, io.BytesIO(flac))


def _write_stage_parquets(cfg, clip_ids, *, trimmed_id=None, trim=(2.0, 6.0)):
    """Write S0-S3 parquet so every clip is a survivor; one may be trimmed."""
    s0, s1, s2, s3 = [], [], [], []
    for i, cid in enumerate(clip_ids):
        is_trim = cid == trimmed_id
        dur = (trim[1] - trim[0]) if is_trim else 8.0
        s0.append(S0PrefilterRow(
            clip_id=cid, shard="00000", original_id=cid, original_speaker=f"SPK{i % 2}",
            original_text=f"文本{i}内容", original_language="zh", duration_s=dur,
            original_dnsmos=3.5, dur_ok=True, lang_ok=True, dnsmos_ok=True,
            text_ok=True, passed=True, reject_reason=None))
        s1.append(S1AcousticsRow(
            clip_id=cid, shard="00000", aes_pq=7.0 + i * 0.1, aes_pc=1.9, aes_ce=7.0,
            aes_cu=7.0, dnsmos_sig=3.9, dnsmos_bak=4.1, dnsmos_ovrl=3.7, snr_db=28.0,
            clipping_ratio=0.0, bandwidth_hz=11000.0, loudness_lufs=-19.0,
            passed=True, reject_reason=None))
        s2.append(S2ProsodyRow(
            clip_id=cid, shard="00000", f0_mean_hz=200.0, f0_std_st=3.0 + i,
            f0_range_st=10.0 + i, energy_std_db=6.0, speech_rate_cps=4.9,
            rate_var_cv=0.3, pause_count=2, pause_total_ms=600.0,
            f0_tracker_confidence=0.9))
        s3.append(S3SpeakerRow(
            clip_id=cid, shard="00000", original_speaker=f"SPK{i % 2}",
            emb_file="emb-00000.npy", emb_row=i, gender_pred="female", n_windows=9,
            mean_win_cos=0.9, min_win_cos=0.8, f0_stability=0.9,
            verdict="intruded_trimmed" if is_trim else "single",
            intrusion_span_ms=2000.0 if is_trim else None, trimmed=is_trim,
            trimmed_duration_s=dur if is_trim else None,
            trim_start_s=trim[0] if is_trim else None,
            trim_end_s=trim[1] if is_trim else None))
    io_utils.atomic_write_parquet([r.model_dump() for r in s0], cfg.paths.s0_prefilter / "part-00000.parquet")
    io_utils.atomic_write_parquet([r.model_dump() for r in s1], cfg.paths.s1_acoustics / "part-00000.parquet")
    io_utils.atomic_write_parquet([r.model_dump() for r in s2], cfg.paths.s2_prosody / "part-00000.parquet")
    io_utils.atomic_write_parquet([r.model_dump() for r in s3], cfg.paths.s3_speaker_features / "part-00000.parquet")


def test_query_survivors_priority_ordered_with_metrics(tmp_path):
    cfg = _cfg(tmp_path)
    clip_ids = [f"clip{i}" for i in range(5)]
    _write_stage_parquets(cfg, clip_ids)
    rows = phase1_hf.query_phase1_survivors(cfg, apply_s2_top_fraction=False)
    assert len(rows) == 5
    # Priority descending, ranks contiguous from 0.
    priorities = [r["priority"] for r in rows]
    assert priorities == sorted(priorities, reverse=True)
    assert [r["priority_rank"] for r in rows] == list(range(5))
    # Full metric columns present for the meta block.
    assert {"aes_pq", "f0_std_st", "mean_win_cos", "prosody_dsp_score"} <= set(rows[0])


def test_package_phase1_produces_shards_meta_and_card(tmp_path):
    cfg = _cfg(tmp_path)
    clip_ids = [f"clip{i}" for i in range(6)]
    _write_source_tar(cfg, clip_ids)
    _write_stage_parquets(cfg, clip_ids, trimmed_id="clip3")

    result = phase1_hf.package_phase1(
        cfg, apply_s2_top_fraction=False, target_shard_bytes=10_000_000
    )
    assert result.n_clips == 6
    assert result.n_with_audio == 6  # all resolved from the source tar
    assert result.shard_paths
    assert result.manifest_path.exists()
    assert (result.export_dir / "README.md").exists()
    assert io_utils.is_done("pack", phase1_hf.PACK_TASK_ID, cfg.paths.done)

    # Collect members across shards: each clip has audio + a JSON sidecar.
    metas = {}
    audio_members = set()
    for sp in result.shard_paths:
        with tarfile.open(sp) as tar:
            for m in tar.getmembers():
                if m.name.endswith(".json"):
                    metas[m.name] = json.loads(tar.extractfile(m).read())
                elif m.name.endswith(".flac"):
                    audio_members.add(m.name)
    assert len(metas) == 6
    assert len(audio_members) == 6

    sample = next(iter(metas.values()))
    assert sample["language"] == "zh"
    assert sample["stage"] == "phase1_filtered"
    assert sample["pipeline_version"] == cfg.version
    # include_metrics default -> the full acoustics/prosody block is embedded.
    assert "metrics" in sample and "acoustics" in sample["metrics"]
    assert "prosody_dsp_score" in sample["metrics"]["prosody"]


def test_trimmed_clip_ships_trimmed_audio(tmp_path):
    cfg = _cfg(tmp_path)
    clip_ids = ["clipA", "clipT", "clipB"]
    _write_source_tar(cfg, clip_ids, dur_s=8.0)
    _write_stage_parquets(cfg, clip_ids, trimmed_id="clipT", trim=(2.0, 6.0))

    result = phase1_hf.package_phase1(
        cfg, apply_s2_top_fraction=False, target_shard_bytes=10_000_000, shuffle=False
    )
    # Read clipT's audio back and confirm it matches the 4s kept span.
    got_dur = None
    for sp in result.shard_paths:
        with tarfile.open(sp) as tar:
            for m in tar.getmembers():
                if m.name == "clipT.flac":
                    arr, sr = audio_utils.decode_bytes(tar.extractfile(m).read())
                    got_dur = len(arr) / sr
    assert got_dur == pytest.approx(4.0, abs=0.05)


def test_include_metrics_false_omits_metric_block(tmp_path):
    cfg = _cfg(tmp_path)
    clip_ids = [f"clip{i}" for i in range(3)]
    _write_source_tar(cfg, clip_ids)
    _write_stage_parquets(cfg, clip_ids)
    result = phase1_hf.package_phase1(
        cfg, apply_s2_top_fraction=False, include_metrics=False,
        target_shard_bytes=10_000_000,
    )
    with tarfile.open(result.shard_paths[0]) as tar:
        meta = json.loads(
            tar.extractfile(next(m for m in tar.getmembers() if m.name.endswith(".json"))).read()
        )
    assert "metrics" not in meta
    assert meta["priority_rank"] >= 0  # lean fields still present


def test_upload_is_graceful_noop_without_repo_or_token(tmp_path, monkeypatch):
    monkeypatch.delenv(phase1_hf.HF_TOKEN_ENV, raising=False)
    cfg = _cfg(tmp_path)
    clip_ids = ["clip0", "clip1"]
    _write_source_tar(cfg, clip_ids)
    _write_stage_parquets(cfg, clip_ids)
    # repo_id defaults to None in the shipped config.
    res = phase1_hf.upload_phase1_to_hf(cfg)
    assert res.uploaded is False
    assert res.upload_skipped_reason and "repo_id" in res.upload_skipped_reason

    # With a repo but no token, it still skips (token gate).
    cfg2 = cfg.model_copy(update={"hf": cfg.hf.model_copy(update={"repo_id": "org/ds"})})
    res2 = phase1_hf.upload_phase1_to_hf(cfg2)
    assert res2.uploaded is False
    assert res2.upload_skipped_reason and phase1_hf.HF_TOKEN_ENV in res2.upload_skipped_reason
