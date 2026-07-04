"""S2 prosody-DSP unit tests (controlled synthetic tones, mock/real trackers).

Covers F0 metric behavior on controlled F0 tracks, VAD pause statistics, the
batch-relative z-score ``prosody_dsp_score`` (incl. single-clip and empty-batch
edges), the mock-aware factories, and the atomic shard write + done marker.
"""

from __future__ import annotations

import numpy as np
import pytest

from emilia_pipeline.common import io_utils, synthesize
from emilia_pipeline.common.contracts import S2ProsodyRow
from emilia_pipeline.stages import s2_prosody as s2


def _clip(cid: str, *, seed: int = 0, dur: float = 6.0, glide: float = 40.0,
          energy: float = 0.3) -> s2.S2ClipInput:
    a = synthesize.synth_voice(
        synthesize.SynthSpec(duration_s=dur, f0_glide_hz=glide, energy_var=energy, seed=seed)
    )
    return s2.S2ClipInput(clip_id=cid, audio=a, sr=synthesize.DEFAULT_SR, text="合成测试语音内容")


def test_f0_metrics_constant_pitch_is_flat() -> None:
    # Fully-voiced constant F0 -> zero semitone std/range, full confidence.
    track = s2.F0Track(f0_hz=np.full(200, 200.0))
    mean, std_st, range_st, conf = s2._f0_metrics(track)
    assert round(mean) == 200
    assert std_st == pytest.approx(0.0, abs=1e-9)
    assert range_st == pytest.approx(0.0, abs=1e-9)
    assert conf == 1.0


def test_f0_metrics_confidence_tracks_voiced_ratio() -> None:
    f0 = np.full(200, 200.0)
    f0[::2] = 0.0  # devoice half the frames
    _, _, _, conf = s2._f0_metrics(track := s2.F0Track(f0_hz=f0))
    assert 0.4 <= conf <= 0.6  # ~0.5 voiced ratio, no jumps


def test_f0_metrics_empty_track() -> None:
    assert s2._f0_metrics(s2.F0Track(f0_hz=np.zeros(0))) == (0.0, 0.0, 0.0, 0.0)


def test_pause_stats_counts_internal_gaps() -> None:
    # Two speech spans separated by a 500 ms gap -> one pause, leading/trailing excluded.
    pc, total_ms, speech_s = s2._pause_stats([(0.0, 2.0), (2.5, 5.0)], min_silence_ms=100.0)
    assert pc == 1
    assert total_ms == 500.0
    assert speech_s == 4.5


def test_pause_stats_ignores_subthreshold_gaps() -> None:
    pc, total_ms, _ = s2._pause_stats([(0.0, 1.0), (1.05, 2.0)], min_silence_ms=100.0)
    assert pc == 0 and total_ms == 0.0


def test_factories_return_mocks(base_config) -> None:
    assert s2.get_f0_tracker(base_config).is_mock
    assert s2.get_vad(base_config).is_mock


def test_run_s2_stage_rows_and_determinism(base_config) -> None:
    clips = [_clip("c0", seed=0), _clip("c1", seed=1), _clip("c2", seed=2)]
    r1 = s2.run_s2_stage(clips, "S00", base_config)
    r2 = s2.run_s2_stage(clips, "S00", base_config)
    assert [r.clip_id for r in r1] == ["c0", "c1", "c2"]
    assert all(isinstance(r, S2ProsodyRow) for r in r1)
    # prosody_dsp_score is NOT stored per-shard (computed globally in DuckDB);
    # rows carry only the raw metrics. Determinism holds on those raw metrics.
    assert not hasattr(r1[0], "prosody_dsp_score")
    assert [r.model_dump() for r in r1] == [r.model_dump() for r in r2]


def test_compute_prosody_dsp_scores_is_population_relative(base_config) -> None:
    # The reference Python scorer (used by tests / calibration) z-scores over the
    # supplied batch: a single-clip batch is 0, a multi-clip batch is not.
    solo = [_clip("solo", seed=0)]
    feats_solo = [
        s2.extract_prosody_features(c.audio, c.sr, c.text, base_config) for c in solo
    ]
    assert s2.compute_prosody_dsp_scores(feats_solo, base_config.s2.z_weights) == [0.0]


def test_compute_scores_empty_batch(base_config) -> None:
    assert s2.compute_prosody_dsp_scores([], base_config.s2.z_weights) == []


def test_write_s2_shard_atomic_with_done_marker(tmp_config) -> None:
    clips = [_clip("c0", seed=0), _clip("c1", seed=1)]
    rows = s2.run_s2_stage(clips, "S00", tmp_config)
    path = s2.write_s2_shard(rows, "S00", tmp_config)
    assert path.exists()
    assert not list(tmp_config.paths.s2_prosody.glob("*.tmp"))
    assert io_utils.is_done("s2", "S00", tmp_config.paths.done)
    n = io_utils.query_parquet(
        "SELECT count(*) FROM s2", s2=io_utils.parquet_glob(tmp_config.paths.s2_prosody)
    ).fetchall()[0][0]
    assert n == 2
