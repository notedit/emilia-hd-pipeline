"""Real-model (GPU) integration tests -- marker: ``gpu``.

Exercises the *real* Audiobox-Aesthetics, DNSMOS-onnx and CAM++ implementations
that :func:`emilia_pipeline.common.models.get_model` (and the S1 bundle factory)
select when ``use_mocks=False`` AND CUDA is available AND the configured weights
exist on disk.

Skipped by default: this environment has no usable GPU and no configured weights,
so each test reports a clear skip reason rather than failing. When run on a real
GPU box with weights wired into ``configs/pipeline_v1.yaml`` (``models.*``), the
tests assert that:

  * the real model outputs are finite and inside their documented ranges, and
  * the real row schema is identical to the mock row schema (so downstream
    parquet / DuckDB code is agnostic to which tier produced the row).

Run with::

    pytest tests/integration -m gpu
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emilia_pipeline.common import synthesize
from emilia_pipeline.common.config import Config
from emilia_pipeline.common.contracts import S1AcousticsRow, S3SpeakerRow
from emilia_pipeline.common.models import (
    MODEL_AESTHETICS,
    MODEL_CAMPP,
    MODEL_DNSMOS,
    MockAestheticsModel,
    MockCampPlusModel,
    MockDnsmosModel,
    get_model,
)
from emilia_pipeline.stages import s1_acoustics, s3_speaker

from .conftest import cuda_skip_reason, weights_skip_reason

pytestmark = pytest.mark.gpu


def _require(config: Config, kind: str) -> None:
    """Skip the current test unless a real GPU + real ``kind`` weights exist."""
    reason = cuda_skip_reason()
    if reason:
        pytest.skip(reason)
    reason = weights_skip_reason(config, kind=kind)
    if reason:
        pytest.skip(reason)


def _synth_clips(n: int = 3) -> list[tuple[np.ndarray, int]]:
    """A few deterministic high-fidelity synthetic voice clips at 24 kHz."""
    clips: list[tuple[np.ndarray, int]] = []
    for i in range(n):
        arr = synthesize.synth_voice(
            synthesize.SynthSpec(
                duration_s=6.0,
                f0_hz=140.0 + 30.0 * i,
                f0_glide_hz=60.0,
                energy_var=0.4,
                seed=i,
            )
        )
        clips.append((arr, synthesize.DEFAULT_SR))
    return clips


# ---------------------------------------------------------------------------
# Real GPU models: finite + in-range, one dict per clip in order
# ---------------------------------------------------------------------------


def test_real_aesthetics_ranges(real_config: Config) -> None:
    _require(real_config, "aesthetics")
    model = get_model(MODEL_AESTHETICS, real_config)
    assert model.is_mock is False, "expected a real Audiobox-Aesthetics model"
    try:
        batch = _synth_clips(3)
        out = model.predict(batch)
    finally:
        model.close()
    assert len(out) == len(batch)
    # Same keys the mock emits (schema parity for downstream code).
    expected_keys = set(MockAestheticsModel().predict(_synth_clips(1))[0])
    for d in out:
        assert set(d) == expected_keys
        for key in ("aes_pq", "aes_pc", "aes_ce", "aes_cu"):
            v = float(d[key])
            assert np.isfinite(v)
            # Audiobox-Aesthetics scores live on a ~[0, 10] MOS-like scale.
            assert 0.0 <= v <= 10.0, f"{key}={v} out of [0,10]"


def test_real_dnsmos_ranges(real_config: Config) -> None:
    _require(real_config, "dnsmos")
    model = get_model(MODEL_DNSMOS, real_config)
    assert model.is_mock is False, "expected a real DNSMOS onnx model"
    try:
        batch = _synth_clips(3)
        out = model.predict(batch)
    finally:
        model.close()
    assert len(out) == len(batch)
    expected_keys = set(MockDnsmosModel().predict(_synth_clips(1))[0])
    for d in out:
        assert set(d) == expected_keys
        for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"):
            v = float(d[key])
            assert np.isfinite(v)
            # DNSMOS P.835 MOS scores are on [1, 5].
            assert 1.0 <= v <= 5.0, f"{key}={v} out of [1,5]"


def test_real_campplus_geometry(real_config: Config) -> None:
    _require(real_config, "campplus")
    model = get_model(MODEL_CAMPP, real_config)
    assert model.is_mock is False, "expected a real CAM++ model"
    dim = real_config.s3.embedding_dim
    try:
        batch = _synth_clips(2)
        out = model.predict(batch)
    finally:
        model.close()
    assert len(out) == len(batch)
    # Real CAM++ emits the window geometry + embeddings; the verdict / f0
    # stability are owned and recomputed by the S3 stage (not the model), so we
    # assert the essential geometry keys rather than the full mock key set.
    required = {"embedding", "window_embeddings", "n_windows", "mean_win_cos", "min_win_cos"}
    for d in out:
        assert required.issubset(set(d))
        emb = np.asarray(d["embedding"])
        assert emb.shape == (dim,)
        assert np.all(np.isfinite(emb.astype(np.float32)))
        win = np.asarray(d["window_embeddings"])
        assert win.ndim == 2 and win.shape[1] == dim and win.shape[0] == int(d["n_windows"])
        assert -1.0 <= float(d["mean_win_cos"]) <= 1.0
        assert -1.0 <= float(d["min_win_cos"]) <= 1.0
        assert int(d["n_windows"]) >= 1


# ---------------------------------------------------------------------------
# Real stage rows validate against the SAME pydantic schema as the mock tier
# ---------------------------------------------------------------------------


def test_real_s1_rows_schema_matches_mock(real_config: Config) -> None:
    _require(real_config, "aesthetics")
    if weights_skip_reason(real_config, kind="dnsmos"):
        pytest.skip(weights_skip_reason(real_config, kind="dnsmos"))
    models = s1_acoustics.get_s1_models(real_config)
    assert models.is_mock is False
    try:
        clips = [(f"c{i}", a, sr) for i, (a, sr) in enumerate(_synth_clips(3))]
        rows = s1_acoustics.compute_s1_rows(clips, models, real_config, shard="gpu00")
    finally:
        models.close()
    assert len(rows) == len(clips)
    for row in rows:
        assert isinstance(row, S1AcousticsRow)
        # model_dump must round-trip (schema parity with the mock tier).
        assert S1AcousticsRow.model_validate(row.model_dump()) == row
        for key in ("aes_pq", "dnsmos_ovrl", "snr_db", "bandwidth_hz"):
            assert np.isfinite(getattr(row, key))
        assert 0.0 <= row.clipping_ratio <= 1.0


def test_real_s3_rows_schema_matches_mock(real_config: Config) -> None:
    _require(real_config, "campplus")
    clips = _synth_clips(2)
    results = s3_speaker.process_batch(
        clips,
        clip_ids=[f"c{i}" for i in range(len(clips))],
        shard="gpu00",
        original_speakers=["ZH_GPU_S00"] * len(clips),
        f0_tracker_confidences=[0.9] * len(clips),
        cfg=real_config,
    )
    assert len(results) == len(clips)
    for res in results:
        row = res.row
        assert isinstance(row, S3SpeakerRow)
        assert S3SpeakerRow.model_validate(row.model_dump()) == row
        assert res.embedding.shape == (real_config.s3.embedding_dim,)
        assert np.all(np.isfinite(res.embedding.astype(np.float32)))
