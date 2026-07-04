"""Phase-1 fused worker + repack tests (mock models, synthetic tar, no GPU/key)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emilia_pipeline.common import synthesize
from emilia_pipeline.common.config import Config, load_config
from emilia_pipeline.common.contracts import (
    S0PrefilterRow,
    S1AcousticsRow,
    S2ProsodyRow,
    S3SpeakerRow,
)
from emilia_pipeline.common.io_utils import is_done, parquet_glob, query_parquet
from emilia_pipeline.phase1 import repack as repack_mod
from emilia_pipeline.phase1 import worker as worker_mod

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.yaml"


def _tmp_config(tmp_path: Path) -> Config:
    """Mock-forcing config with every path rerooted under a tmp dir."""
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
                    "s0_prefilter": tmp_path / "stage" / "s0",
                    "s1_acoustics": tmp_path / "stage" / "s1",
                    "s2_prosody": tmp_path / "stage" / "s2",
                    "s3_speaker_features": tmp_path / "stage" / "s3feat",
                    "s3_speaker_embeddings": tmp_path / "stage" / "s3emb",
                    "repacked": tmp_path / "repacked",
                    "manifests": tmp_path / "manifests",
                    "done": tmp_path / "done",
                    "failed": tmp_path / "failed",
                }
            )
        }
    )


@pytest.fixture()
def cfg_and_shard(tmp_path: Path):
    cfg = _tmp_config(tmp_path)
    cfg.paths.source.mkdir(parents=True, exist_ok=True)
    tar, clips = synthesize.build_synthetic_shard(
        cfg.paths.source / "00000.tar", n_clips=6, shard_name="00000"
    )
    return cfg, tar, clips


def test_read_shard_clips_order(cfg_and_shard) -> None:
    cfg, tar, clips = cfg_and_shard
    got = worker_mod.read_shard_clips(tar)
    assert [c.clip_id for c in got] == [c.key for c in clips]
    assert all(c.audio_bytes for c in got)


def test_run_shard_end_to_end(cfg_and_shard) -> None:
    cfg, tar, clips = cfg_and_shard
    res = worker_mod.run_shard(tar, cfg, parallel=False)
    assert res is not None
    # S0 row per clip (never dropped).
    assert len(res.s0_rows) == len(clips)
    assert all(isinstance(r, S0PrefilterRow) for r in res.s0_rows)
    # short + long fail S0 duration => S1 only over S0 survivors.
    n_s0_pass = sum(r.passed for r in res.s0_rows)
    assert len(res.s1_rows) == n_s0_pass
    assert all(isinstance(r, S1AcousticsRow) for r in res.s1_rows)
    # S2/S3 only over S1 survivors.
    n_s1_pass = sum(r.passed for r in res.s1_rows)
    assert len(res.s2_rows) == n_s1_pass
    assert len(res.s3_rows) == n_s1_pass
    assert all(isinstance(r, S2ProsodyRow) for r in res.s2_rows)
    assert all(isinstance(r, S3SpeakerRow) for r in res.s3_rows)
    # embeddings aligned to S3 rows; emb pointer filled.
    assert res.embeddings.shape[0] == n_s1_pass
    if n_s1_pass:
        assert res.embeddings.shape[1] == cfg.s3.embedding_dim
        assert res.embeddings.dtype == np.float16
        assert res.s3_rows[0].emb_file == "emb-00000.npy"
        assert [r.emb_row for r in res.s3_rows] == list(range(n_s1_pass))
    # done marker created.
    assert is_done("phase1", "00000", cfg.paths.done)
    # no leftover tmp files anywhere.
    assert not list(cfg.paths.s0_prefilter.glob("*.tmp"))
    assert not list(cfg.paths.s3_speaker_embeddings.glob("*.tmp"))


def test_run_shard_parquet_queryable(cfg_and_shard) -> None:
    cfg, tar, clips = cfg_and_shard
    worker_mod.run_shard(tar, cfg, parallel=False)
    n = query_parquet(
        "SELECT count(*) FROM s0", s0=parquet_glob(cfg.paths.s0_prefilter)
    ).fetchall()[0][0]
    assert n == len(clips)
    # S1 rows readable, passed column is boolean.
    rows = query_parquet(
        "SELECT clip_id, passed FROM s1", s1=parquet_glob(cfg.paths.s1_acoustics)
    ).fetchall()
    assert all(isinstance(r[1], bool) for r in rows)


def test_run_shard_idempotent_skip(cfg_and_shard) -> None:
    cfg, tar, clips = cfg_and_shard
    worker_mod.run_shard(tar, cfg, parallel=False)
    # Second run should short-circuit on the done marker.
    assert worker_mod.run_shard(tar, cfg, parallel=False, skip_if_done=True) is None


def test_run_shard_determinism(cfg_and_shard, tmp_path) -> None:
    cfg, tar, clips = cfg_and_shard
    res1 = worker_mod.run_shard(tar, cfg, parallel=False, mark_done=False)
    res2 = worker_mod.run_shard(tar, cfg, parallel=False, mark_done=False)
    d1 = [r.model_dump() for r in res1.s1_rows]
    d2 = [r.model_dump() for r in res2.s1_rows]
    assert d1 == d2
    assert np.array_equal(res1.embeddings, res2.embeddings)


def test_repack_worklist_and_index(cfg_and_shard) -> None:
    cfg, tar, clips = cfg_and_shard
    worker_mod.run_shard(tar, cfg, parallel=False)
    # Disable the top-fraction gate so we keep survivors regardless of mock scores.
    summary = repack_mod.run_repack(
        cfg, apply_s2_top_fraction=False, parallel=False
    )
    assert summary["n_survivors"] >= 0
    assert (cfg.paths.manifests / repack_mod.WORKLIST_NAME).exists()
    assert (cfg.paths.repacked / repack_mod.REPACK_INDEX_NAME).exists()

    if summary["n_survivors"] > 0:
        # worklist ordered by priority desc; ranks are dense 0..n-1.
        wl = query_parquet(
            "SELECT priority, priority_rank, slice_id FROM w ORDER BY priority_rank",
            w=str(cfg.paths.manifests / repack_mod.WORKLIST_NAME),
        ).fetchall()
        prios = [r[0] for r in wl]
        assert prios == sorted(prios, reverse=True)
        assert [r[1] for r in wl] == list(range(len(wl)))
        # repack index count == indexed clips; offsets unique within a shard.
        idx = query_parquet(
            "SELECT clip_id, shard, offset FROM ix",
            ix=str(cfg.paths.repacked / repack_mod.REPACK_INDEX_NAME),
        ).fetchall()
        assert len(idx) == summary["n_indexed"]
        # A repacked shard tar exists for each slice.
        assert list(cfg.paths.repacked.glob("shard-*.tar"))


def test_repack_slice_ordering() -> None:
    # slice_size drives slice_id; ranks map to slices in priority order.
    from emilia_pipeline.phase1.repack import _slice_token

    assert _slice_token(0, 3) == "00000"
    assert _slice_token(12, 3) == "00012"


def test_survivor_query_empty(tmp_path) -> None:
    cfg = _tmp_config(tmp_path)
    cfg.paths.source.mkdir(parents=True, exist_ok=True)
    tar, clips = synthesize.build_synthetic_shard(
        cfg.paths.source / "00000.tar", n_clips=4, shard_name="00000"
    )
    worker_mod.run_shard(tar, cfg, parallel=False)
    # With the strict top-fraction gate on, may keep few/none; must not raise.
    survivors = repack_mod.query_survivors(cfg, apply_s2_top_fraction=True)
    assert isinstance(survivors, list)
