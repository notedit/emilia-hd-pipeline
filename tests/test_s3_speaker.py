"""S3 sliding-window speaker purity tests (mock CAM++, synthetic audio, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emilia_pipeline.common.config import Config, load_config
from emilia_pipeline.common.contracts import S3SpeakerRow, SpeakerVerdict
from emilia_pipeline.common import synthesize
from emilia_pipeline.stages import s3_speaker as s3

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.yaml"


@pytest.fixture()
def mock_config() -> Config:
    cfg = load_config(CONFIG_PATH)
    return cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"use_mocks": True})}
    )


def test_frame_windows_overlap() -> None:
    sr = 24000
    spans = s3.frame_windows(int(6.0 * sr), sr, window_s=1.5, overlap=0.5)
    assert len(spans) >= 2
    # 50% overlap -> hop = 0.75s = 18000 samples
    assert spans[1][0] - spans[0][0] == 18000
    assert spans[0] == (0, 36000)


def test_frame_windows_short_clip() -> None:
    sr = 24000
    spans = s3.frame_windows(int(1.0 * sr), sr, window_s=1.5, overlap=0.5)
    assert spans == [(0, 24000)]


def test_window_geometry_single_speaker_high_cos() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(192).astype(np.float32)
    # Windows are the same base + tiny jitter -> high cosine to the mean.
    wins = np.stack([base + 0.01 * rng.standard_normal(192) for _ in range(8)])
    geom = s3.window_geometry(wins.astype(np.float32))
    assert geom.n_windows == 8
    assert geom.mean_win_cos > 0.95
    assert geom.min_win_cos > 0.9


def test_locate_low_cos_spans() -> None:
    cos = np.array([0.9, 0.9, 0.4, 0.3, 0.9, 0.9])
    spans = s3.locate_low_cos_spans(cos, threshold=0.70)
    assert len(spans) == 1
    assert (spans[0].start_win, spans[0].end_win) == (2, 3)


def test_head_intrusion_trimmed() -> None:
    cfg = load_config(CONFIG_PATH)
    sr = 24000
    dur = 8.0
    # 10 windows; first two are the intrusion (low cos), rest clean.
    n = 10
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(192).astype(np.float32)
    intruder = -clean  # opposite direction -> negative cosine
    wins = np.stack([intruder, intruder] + [clean + 0.01 * rng.standard_normal(192) for _ in range(n - 2)])
    geom = s3.window_geometry(wins.astype(np.float32))
    spans_samples = s3.frame_windows(int(dur * sr), sr, cfg.s3.window_s, cfg.s3.window_overlap)
    spans_samples = s3._resize_spans(spans_samples, geom.n_windows)
    decision = s3.decide_verdict(
        cfg=cfg,
        n_windows=geom.n_windows,
        window_cos=geom.window_cos,
        mean_win_cos=geom.mean_win_cos,
        min_win_cos=geom.min_win_cos,
        f0_tracker_confidence=0.9,
        window_spans_samples=spans_samples,
        sr=sr,
        total_duration_s=dur,
    )
    assert decision.verdict == SpeakerVerdict.INTRUDED_TRIMMED
    assert decision.trimmed
    assert decision.trim_start_s > 0.0
    assert decision.trimmed_duration_s >= cfg.s3.min_trim_residual_s


def test_mid_intrusion_rejected() -> None:
    cfg = load_config(CONFIG_PATH)
    sr = 24000
    dur = 8.0
    n = 10
    rng = np.random.default_rng(2)
    clean = rng.standard_normal(192).astype(np.float32)
    intruder = -clean
    windows = [clean + 0.01 * rng.standard_normal(192) for _ in range(n)]
    windows[4] = intruder
    windows[5] = intruder
    wins = np.stack(windows)
    geom = s3.window_geometry(wins.astype(np.float32))
    spans_samples = s3._resize_spans(
        s3.frame_windows(int(dur * sr), sr, cfg.s3.window_s, cfg.s3.window_overlap),
        geom.n_windows,
    )
    decision = s3.decide_verdict(
        cfg=cfg,
        n_windows=geom.n_windows,
        window_cos=geom.window_cos,
        mean_win_cos=geom.mean_win_cos,
        min_win_cos=geom.min_win_cos,
        f0_tracker_confidence=0.9,
        window_spans_samples=spans_samples,
        sr=sr,
        total_duration_s=dur,
    )
    assert decision.verdict == SpeakerVerdict.INTRUDED_REJECTED
    assert not decision.trimmed


def test_uniform_depressed_poor_f0_overlap_rejected() -> None:
    cfg = load_config(CONFIG_PATH)
    decision = s3.decide_verdict(
        cfg=cfg,
        n_windows=6,
        window_cos=None,
        mean_win_cos=0.72,  # below mean threshold 0.80
        min_win_cos=0.71,  # above min threshold 0.70 -> not a local collapse
        f0_tracker_confidence=0.4,  # poor
        window_spans_samples=None,
        sr=24000,
        total_duration_s=6.0,
    )
    assert decision.verdict == SpeakerVerdict.OVERLAP_REJECTED


def test_uniform_depressed_normal_f0_degraded_pass() -> None:
    cfg = load_config(CONFIG_PATH)
    decision = s3.decide_verdict(
        cfg=cfg,
        n_windows=6,
        window_cos=None,
        mean_win_cos=0.75,
        min_win_cos=0.72,
        f0_tracker_confidence=0.9,
        window_spans_samples=None,
        sr=24000,
        total_duration_s=6.0,
    )
    assert decision.verdict == SpeakerVerdict.DEGRADED_PASS


def test_normal_single() -> None:
    cfg = load_config(CONFIG_PATH)
    decision = s3.decide_verdict(
        cfg=cfg,
        n_windows=8,
        window_cos=None,
        mean_win_cos=0.92,
        min_win_cos=0.85,
        f0_tracker_confidence=0.95,
        window_spans_samples=None,
        sr=24000,
        total_duration_s=6.0,
    )
    assert decision.verdict == SpeakerVerdict.SINGLE


def test_process_clip_mock_end_to_end(mock_config: Config) -> None:
    arr = synthesize.synth_voice(synthesize.SynthSpec(duration_s=6.0))
    result = s3.process_clip(
        arr,
        24000,
        clip_id="c1",
        shard="00000",
        original_speaker="ZH_TEST_S00",
        f0_tracker_confidence=0.9,
        cfg=mock_config,
    )
    assert isinstance(result.row, S3SpeakerRow)
    assert result.row.original_speaker == "ZH_TEST_S00"
    assert result.row.emb_file == "" and result.row.emb_row == -1
    assert result.embedding.shape == (mock_config.s3.embedding_dim,)
    assert result.embedding.dtype == np.float16
    # Determinism: same input -> same verdict + metrics.
    result2 = s3.process_clip(
        arr, 24000, clip_id="c1", shard="00000",
        original_speaker="ZH_TEST_S00", f0_tracker_confidence=0.9, cfg=mock_config,
    )
    assert result.row.verdict == result2.row.verdict
    assert result.row.mean_win_cos == result2.row.mean_win_cos


def test_with_embedding_ref(mock_config: Config) -> None:
    arr = synthesize.synth_voice(synthesize.SynthSpec(duration_s=5.0))
    result = s3.process_clip(
        arr, 24000, clip_id="c1", shard="00042",
        original_speaker="SP", f0_tracker_confidence=0.9, cfg=mock_config,
    )
    row = result.with_embedding_ref("emb-00042.npy", 7)
    assert row.emb_file == "emb-00042.npy"
    assert row.emb_row == 7
    # original row untouched (immutable update).
    assert result.row.emb_row == -1


def test_process_batch_matches_single(mock_config: Config) -> None:
    a = synthesize.synth_voice(synthesize.SynthSpec(duration_s=6.0, seed=1))
    b = synthesize.synth_voice(synthesize.SynthSpec(duration_s=7.0, seed=2))
    batch = s3.process_batch(
        [(a, 24000), (b, 24000)],
        clip_ids=["a", "b"],
        shard="00000",
        original_speakers=["S0", "S1"],
        f0_tracker_confidences=[0.9, 0.8],
        cfg=mock_config,
    )
    single_a = s3.process_clip(
        a, 24000, clip_id="a", shard="00000",
        original_speaker="S0", f0_tracker_confidence=0.9, cfg=mock_config,
    )
    assert len(batch) == 2
    assert batch[0].row.verdict == single_a.row.verdict
    assert batch[0].row.mean_win_cos == single_a.row.mean_win_cos
    assert batch[1].row.clip_id == "b"


def test_trim_produces_shorter_audio() -> None:
    """A real window-embedding sequence with a head intrusion yields trimmed audio."""
    cfg = load_config(CONFIG_PATH)
    sr = 24000
    dur = 8.0
    n_samples = int(dur * sr)
    audio = np.sin(2 * np.pi * 180 * np.arange(n_samples) / sr).astype(np.float32)

    n = 10
    rng = np.random.default_rng(3)
    clean = rng.standard_normal(192).astype(np.float32)
    intruder = -clean
    wins = np.stack([intruder] + [clean + 0.01 * rng.standard_normal(192) for _ in range(n - 1)])

    # Feed the window sequence through a tiny fake model to exercise the real path.
    class _FakeCampp:
        is_mock = True
        name = "campplus"

        def predict(self, batch):
            return [{"window_embeddings": wins.astype(np.float32),
                     "gender_pred": "female"}]

        def close(self):
            pass

    result = s3.process_clip(
        audio, sr, clip_id="c", shard="0",
        original_speaker="SP", f0_tracker_confidence=0.9,
        cfg=cfg, model=_FakeCampp(),
    )
    assert result.row.verdict == SpeakerVerdict.INTRUDED_TRIMMED
    assert result.row.trimmed
    assert result.trimmed_audio is not None
    assert len(result.trimmed_audio) < n_samples
    assert result.row.trimmed_duration_s >= cfg.s3.min_trim_residual_s
