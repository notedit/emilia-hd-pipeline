"""Foundation smoke tests: contracts, config, IO, audio, mock models are wired.

Runs with zero GPU / API key / real data (all mocks). Stage agents add their
own suites; this file only guards the Foundation contract surface.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from emilia_pipeline.common import audio, io_utils, models, synthesize
from emilia_pipeline.common.config import Config, load_config
from emilia_pipeline.common.contracts import (
    MetaRecord,
    S4GuidedJSON,
    flatten_meta,
    unflatten_meta,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.yaml"


@pytest.fixture()
def mock_config() -> Config:
    cfg = load_config(CONFIG_PATH)
    return cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"use_mocks": True})}
    )


def test_config_loads_thresholds() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.s1.min_aes_pq == 7.0
    assert cfg.s0.language == "zh"
    assert cfg.s4.temperature == 0.0
    assert cfg.api_key() is None  # no key exported in env


def test_synth_shard_decode(tmp_path: Path, mock_config: Config) -> None:
    tar, clips = synthesize.build_synthetic_shard(tmp_path / "s.tar", n_clips=3)
    with tarfile.open(tar) as t:
        arr, sr = audio.decode_tar_member(t, clips[0].key + ".flac")
    assert arr.dtype.name == "float32" and sr == synthesize.DEFAULT_SR
    assert audio.duration_s(arr, sr) > 0


def test_mock_models_deterministic(mock_config: Config) -> None:
    arr = synthesize.synth_voice(synthesize.SynthSpec(duration_s=4.0))
    for name in (models.MODEL_AESTHETICS, models.MODEL_DNSMOS, models.MODEL_CAMPP):
        m = models.get_model(name, mock_config)
        assert m.is_mock
        first = m.predict([(arr, 24000)])[0]
        second = m.predict([(arr, 24000)])[0]
        for key in first:
            a, b = first[key], second[key]
            if hasattr(a, "shape"):
                assert (a == b).all()
            else:
                assert a == b


@pytest.mark.asyncio
async def test_mock_s4_client(mock_config: Config) -> None:
    client = models.get_s4_client(mock_config)
    assert client.is_mock
    arr = synthesize.synth_voice(synthesize.SynthSpec(duration_s=4.0))
    label = await client.label(audio=arr, sample_rate=16000, reference_text="你好")
    assert isinstance(label, S4GuidedJSON)


def test_atomic_write_and_duckdb(tmp_path: Path) -> None:
    io_utils.atomic_write_parquet(
        [{"clip_id": "a", "aes_pq": 7.5}, {"clip_id": "b", "aes_pq": 8.0}],
        tmp_path / "s1" / "part-0.parquet",
    )
    assert not list((tmp_path / "s1").glob("*.tmp"))
    io_utils.write_done_marker("s1", "0", tmp_path / "done")
    assert io_utils.is_done("s1", "0", tmp_path / "done")
    assert io_utils.pending_tasks(["0", "1"], "s1", tmp_path / "done") == ["1"]
    rows = io_utils.query_parquet(
        "SELECT count(*) FROM s1", s1=io_utils.parquet_glob(tmp_path / "s1")
    ).fetchall()
    assert rows[0][0] == 2


def test_meta_flatten_roundtrip() -> None:
    example = {
        "clip_id": "c1",
        "schema_version": "1.3",
        "audio": {"path": "x", "duration_s": 8.4, "sample_rate": 24000,
                  "channels": 1, "loudness_lufs": -19.3},
        "source": {"dataset": "emilia", "dataset_version": "v1", "original_id": "O",
                   "original_text": "t", "original_speaker": "SP",
                   "original_dnsmos": 3.4, "original_language": "zh"},
        "acoustics": {"aesthetics": {"pq": 7.8, "pc": 1.9, "ce": 7.1, "cu": 7.4},
                      "dnsmos_p835": {"sig": 3.9, "bak": 4.1, "ovrl": 3.7},
                      "snr_db": 28.4, "clipping_ratio": 0.0, "bandwidth_hz": 11200},
        "prosody_dsp": {"f0_mean_hz": 218.5, "f0_std_st": 4.2, "f0_range_st": 14.1,
                        "energy_std_db": 6.8, "speech_rate_cps": 4.9,
                        "rate_var_cv": 0.31, "pause_count": 2, "pause_total_ms": 610,
                        "f0_tracker_confidence": 0.94, "prosody_dsp_score": 0.81},
        "speaker": {"original_speaker": "SP",
                    "embedding_ref": {"emb_file": "emb-1.npy", "emb_row": 187},
                    "gender_pred": "female",
                    "purity_check": {"n_windows": 9, "mean_win_cos": 0.91,
                                     "min_win_cos": 0.83, "f0_stability": 0.94,
                                     "verdict": "single", "intrusion_span_ms": None,
                                     "trimmed": False}},
        "omni_labels": {"model": "qwen", "prompt_version": "v3.1",
                        "text_verdict": "fixable", "text_fixed": "a",
                        "text_punctuated": "b", "cer_vs_original": 0.0,
                        "emotion": {"primary": "sad", "secondary": "affectionate",
                                    "intensity": 4, "confidence": 0.86},
                        "prosody": {"expressiveness": 4,
                                    "speaking_style": "storytelling",
                                    "rhythm": "dramatic", "prominent_stress": True},
                        "context": {"scenario": "podcast", "register": "intimate",
                                    "summary": "s"},
                        "language": {"primary": "zh", "code_switch": False,
                                     "accent": "standard"},
                        "paralinguistic": ["breath_prominent"], "defects": [],
                        "usable": True},
        "selection": {"selection_score": 0.87, "tier": "S", "reject_reason": None},
        "pipeline": {"version": "v1.3",
                     "stages_passed": ["s0", "s1", "s2", "s3", "s4"],
                     "processed_at": "2026-07-05T03:12:44Z"},
    }
    record = MetaRecord.model_validate(example)
    flat = flatten_meta(record)
    assert flat["omni_labels_emotion_primary"] == "sad"
    assert flat["speaker_original_speaker"] == "SP"
    rebuilt = MetaRecord.model_validate(unflatten_meta(flat))
    assert rebuilt == record
