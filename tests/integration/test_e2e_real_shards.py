"""Gated small end-to-end integration test on real Emilia shards.

Markers: ``integration`` + ``slow``.

Runs the real Phase-1 fused scan over a couple of actual Emilia ``*.tar`` shards
supplied via ``$EMILIA_SAMPLE_DIR``, then the repack -> S4 (mock unless a key is
present) -> S5 tail, writing every artifact under a tmp dir. This is the closest
thing to a production dry-run that fits in a test.

Skipped by default: ``$EMILIA_SAMPLE_DIR`` is unset here, so the test reports a
clear skip reason rather than failing. GPU models fall back to mocks when no GPU
is present (design convention), so this test can validate the *plumbing / wiring
/ atomic-write discipline* end-to-end even on a CPU box; on a GPU box with
weights configured it also exercises the real models.

Run with::

    EMILIA_SAMPLE_DIR=/path/to/emilia/shards pytest tests/integration -m integration
"""

from __future__ import annotations

import numpy as np
import pytest

from emilia_pipeline.common.config import Config
from emilia_pipeline.common.io_utils import (
    is_done,
    parquet_glob,
    query_parquet,
)
from emilia_pipeline.phase1 import repack, worker
from emilia_pipeline.phase2 import s4_client
from emilia_pipeline.scoring import s5_score

from .conftest import emilia_sample_shards, emilia_sample_skip_reason

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _require_shards() -> None:
    reason = emilia_sample_skip_reason()
    if reason:
        pytest.skip(reason)


def test_phase1_scan_on_real_shards(real_config_tmp_paths: Config) -> None:
    _require_shards()
    cfg = real_config_tmp_paths
    shards = emilia_sample_shards(limit=2)

    for shard_path in shards:
        result = worker.run_shard(shard_path, cfg, parallel=False, mark_done=True)
        assert result is not None
        assert result.n_clips > 0
        # S0 is a whitelist over ALL clips (never drops rows).
        assert len(result.s0_rows) == result.n_clips
        # Done marker is the sole source of truth for completion (design §3).
        assert is_done(worker.PHASE1_STAGE, shard_path.stem, cfg.paths.done)
        # No leftover *.tmp files anywhere under the stage tree.
        assert not list(cfg.paths.stage.rglob("*.tmp"))

    # Idempotent re-run: already-done shard is skipped.
    assert worker.run_shard(shards[0], cfg, parallel=False, skip_if_done=True) is None

    # Stage parquet is queryable via DuckDB.
    n_s0 = query_parquet(
        "SELECT count(*) FROM s0", s0=parquet_glob(cfg.paths.s0_prefilter)
    ).fetchall()[0][0]
    assert n_s0 > 0


def test_full_tail_repack_s4_s5(real_config_tmp_paths: Config) -> None:
    _require_shards()
    cfg = real_config_tmp_paths
    shards = emilia_sample_shards(limit=2)

    for shard_path in shards:
        worker.run_shard(shard_path, cfg, parallel=False, mark_done=True)

    # --- Repack: build worklist + repacked shards from the survivor set. ---
    repack_stats = repack.run_repack(cfg, parallel=False)
    assert repack_stats["n_survivors"] >= 0
    worklist_path = cfg.paths.manifests / repack.WORKLIST_NAME
    assert worklist_path.exists()

    # If any clip survived S0/S1/S3, label the first slice and run S5.
    if repack_stats["n_survivors"] > 0:
        assert repack_stats["n_slices"] >= 1
        # slice ids are zero-padded rank // slice_size; the first is all-zeros.
        first_slice = f"{0:05d}"
        entries = s4_client.load_slice_worklist(cfg, first_slice)
        if entries:
            source = s4_client.RepackIndexAudioSource(cfg)
            try:
                rows = s4_client.run_slice(
                    first_slice, cfg, audio_source=source, worklist=entries
                )
            finally:
                source.close()
            assert len(rows) == len(entries)
            assert is_done("s4", first_slice, cfg.paths.done)

        # --- S5: join -> score -> tier -> flat parquet + per-clip JSON. ---
        s5_result = s5_score.run_s5(cfg, write=True)
        assert s5_result.n_candidates > 0
        # Every kept clip is assigned a tier; tier counts sum to n_kept.
        assert sum(s5_result.tier_counts.values()) == s5_result.n_kept
        if s5_result.flat_parquet_path is not None:
            assert s5_result.flat_parquet_path.exists()

    # No leftover *.tmp files anywhere under the tmp root (atomic-write discipline).
    assert not list(cfg.paths.root.rglob("*.tmp"))
