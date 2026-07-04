"""S1 strict-acoustics unit tests (mock GPU models, pure-numpy CPU DSP).

Covers CPU metric sanity, the ``s1_pass`` query-condition predicate (including a
boundary case and model-vs-mapping equivalence), batch order/determinism, and
the atomic shard write. Zero GPU: the aesthetics/DNSMOS models come from the
mock-aware factory.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from emilia_pipeline.common import audio, io_utils, synthesize
from emilia_pipeline.common.contracts import S1AcousticsRow
from emilia_pipeline.stages import s1_acoustics as s1


def _clips_from_shard(cfg):
    cfg.paths.source.mkdir(parents=True, exist_ok=True)
    tar, clips = synthesize.build_synthetic_shard(
        cfg.paths.source / "00000.tar", n_clips=4, shard_name="00000"
    )
    out = []
    with tarfile.open(tar) as t:
        for c in clips:
            arr, sr = audio.decode_tar_member(t, c.key + ".flac")
            out.append((c.key, arr, sr))
    return out


def test_cpu_metrics_shape_and_range() -> None:
    sr = 24000
    sig = synthesize.synth_voice(synthesize.SynthSpec(duration_s=4.0))
    m = s1.compute_cpu_metrics(sig, sr)
    assert set(m) == {"snr_db", "clipping_ratio", "bandwidth_hz", "loudness_lufs"}
    assert m["clipping_ratio"] >= 0.0
    assert m["bandwidth_hz"] >= 0.0
    assert np.isfinite(m["snr_db"])
    assert np.isfinite(m["loudness_lufs"])


def test_bandwidth_detects_low_rolloff() -> None:
    # A pure low tone rolls off far below 8 kHz; a broadband noise reaches higher.
    sr = 24000
    tone = synthesize.sine_tone(180.0, 3.0, sr)
    bw_tone = s1.bandwidth_hz(tone, sr)
    bw_noise = s1.bandwidth_hz(synthesize.noise(3.0, sr, amplitude=0.3), sr)
    assert bw_tone < bw_noise


def _row(**over) -> S1AcousticsRow:
    base = dict(clip_id="c", shard="s", aes_pq=7.5, aes_pc=1.9, aes_ce=7.0, aes_cu=7.0,
                dnsmos_sig=3.9, dnsmos_bak=4.1, dnsmos_ovrl=3.7, snr_db=25.0,
                clipping_ratio=0.0, bandwidth_hz=10000.0, loudness_lufs=-19.0,
                passed=True, reject_reason=None)
    base.update(over)
    return S1AcousticsRow(**base)


def test_pass_predicate_all_gates_ok(base_config) -> None:
    assert s1.s1_pass(_row(), base_config) is True
    assert s1.s1_reject_reason(_row(), base_config) is None


def test_pass_predicate_boundary_pq(base_config) -> None:
    # min_aes_pq == 7.0: exactly 7.0 passes, just below fails on that gate only.
    assert s1.s1_pass(_row(aes_pq=7.0), base_config) is True
    reason = s1.s1_reject_reason(_row(aes_pq=6.999), base_config)
    assert reason is not None and reason.startswith("aes_pq<")


def test_pass_predicate_accepts_mapping_and_model(base_config) -> None:
    row = _row(bandwidth_hz=1000.0)  # fails bandwidth gate
    assert s1.s1_pass(row, base_config) == s1.s1_pass(row.model_dump(), base_config)
    assert s1.s1_pass(row, base_config) is False


def test_compute_s1_rows_order_and_determinism(relaxed_config) -> None:
    clips = _clips_from_shard(relaxed_config)
    models = s1.get_s1_models(relaxed_config)
    assert models.is_mock
    rows1 = s1.compute_s1_rows(clips, models, relaxed_config, shard="00000")
    rows2 = s1.compute_s1_rows(clips, models, relaxed_config, shard="00000")
    assert [r.clip_id for r in rows1] == [c[0] for c in clips]
    assert all(isinstance(r, S1AcousticsRow) for r in rows1)
    assert [r.model_dump() for r in rows1] == [r.model_dump() for r in rows2]


def test_compute_s1_rows_empty_batch(base_config) -> None:
    models = s1.get_s1_models(base_config)
    assert s1.compute_s1_rows([], models, base_config, shard="00000") == []


def test_write_s1_rows_atomic_and_queryable(relaxed_config) -> None:
    clips = _clips_from_shard(relaxed_config)
    models = s1.get_s1_models(relaxed_config)
    rows = s1.compute_s1_rows(clips, models, relaxed_config, shard="00000")
    path = s1.write_s1_rows(rows, relaxed_config, "00000")
    assert path.exists()
    assert not list(Path(relaxed_config.paths.s1_acoustics).glob("*.tmp"))
    got = io_utils.query_parquet(
        "SELECT count(*) FROM s1", s1=s1.s1_parquet_glob(relaxed_config)
    ).fetchall()[0][0]
    assert got == len(rows)
