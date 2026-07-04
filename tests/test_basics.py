"""Basic functional checks (fast, fully mocked).

Deliberately minimal: the deep per-stage behavior is exercised by the full
mock pipeline in ``test_e2e_mock_pipeline.py``; this file only pins the
fundamentals that everything else builds on -- config loading, the Phase-1
worker's core invariants on a synthetic shard, and the S1 pass predicate.
"""

from __future__ import annotations

from emilia_pipeline.common import io_utils
from emilia_pipeline.phase1 import worker as worker_mod
from emilia_pipeline.stages.s1_acoustics import s1_pass, s1_reject_reason


def test_config_loads(base_config) -> None:
    cfg = base_config
    assert cfg.s0.min_duration_s < cfg.s0.max_duration_s
    assert cfg.s1.min_aes_pq > 0
    assert abs(sum(cfg.s2.z_weights.model_dump().values()) - 1.0) < 1e-6
    assert cfg.s3.embedding_dim == 192
    assert cfg.runtime.use_mocks is True


def test_phase1_worker_invariants(synth_shard) -> None:
    cfg, tar, clips = synth_shard
    res = worker_mod.run_shard(tar, cfg, parallel=False)

    # Every clip gets an S0 row; done marker lands after all writes.
    assert res is not None
    assert len(res.s0_rows) == len(clips)
    assert io_utils.is_done("phase1", "00000", cfg.paths.done)
    for path in res.paths.values():
        assert path.exists()

    # S1 rows exist only for S0 survivors; S2/S3 only for S1 survivors.
    n_s0_pass = sum(r.passed for r in res.s0_rows)
    n_s1_pass = sum(r.passed for r in res.s1_rows)
    assert len(res.s1_rows) == n_s0_pass
    assert len(res.s2_rows) == n_s1_pass
    assert len(res.s3_rows) == n_s1_pass
    assert res.embeddings.shape == (n_s1_pass, cfg.s3.embedding_dim)

    # DNSMOS is retired from S1: columns exist for schema stability but are
    # always None, and no gate ever rejects on them.
    for row in res.s1_rows:
        assert row.dnsmos_ovrl is None
        assert "dnsmos" not in (row.reject_reason or "")

    # Re-running a completed shard is a no-op (idempotency via done marker).
    assert worker_mod.run_shard(tar, cfg, parallel=False) is None


def test_s1_pass_predicate(base_config) -> None:
    cfg = base_config
    good = {
        "aes_pq": 8.0, "aes_pc": 1.5,
        "snr_db": 30.0, "clipping_ratio": 0.0, "bandwidth_hz": 11000.0,
    }
    assert s1_pass(good, cfg)
    assert s1_reject_reason({**good, "aes_pq": 5.0}, cfg) == f"aes_pq<{cfg.s1.min_aes_pq}"
    assert not s1_pass({**good, "snr_db": 3.0}, cfg)
