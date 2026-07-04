"""End-to-end mock-tier run over one synthetic shard.

Drives the full pipeline with zero GPU / API key / real data:

    S0/S1/S2/S3 fused worker -> repack -> S4 (mock transport) -> S5 scoring ->
    HF packaging -> monitor snapshot

and asserts a non-empty tiered manifest plus a §7-valid published JSON. The S1
funnel is opened (``relaxed_config``) so synthetic sine clips survive to the
export; every stage's own logic is unchanged. Marked ``slow`` but still in the
always-on mock tier (no external dependency).
"""

from __future__ import annotations

import json
import tarfile

import pytest

from emilia_pipeline.common import io_utils, synthesize
from emilia_pipeline.common.contracts import MetaRecord, S4Status
from emilia_pipeline.phase1 import repack as repack_mod
from emilia_pipeline.phase1 import worker as worker_mod
from emilia_pipeline.phase2 import s4_client
from emilia_pipeline.scoring import hf_package, monitor, s5_score


@pytest.mark.slow
def test_full_pipeline_synthetic(relaxed_config, clip_audio_map) -> None:
    cfg = relaxed_config
    cfg.paths.source.mkdir(parents=True, exist_ok=True)
    tar, clips = synthesize.build_synthetic_shard(
        cfg.paths.source / "00000.tar", n_clips=8, shard_name="00000"
    )

    # ---- Phase 1: fused S0..S3 scan over the shard ----
    res = worker_mod.run_shard(tar, cfg, parallel=False)
    assert res is not None
    assert io_utils.is_done("phase1", "00000", cfg.paths.done)
    assert len(res.s0_rows) == len(clips)

    # ---- Repack: survivors -> worklist + repacked shards ----
    summary = repack_mod.run_repack(cfg, apply_s2_top_fraction=False, parallel=False)
    assert summary["n_survivors"] > 0
    slices = io_utils.enumerate_slice_tasks(
        cfg.paths.manifests / repack_mod.WORKLIST_NAME
    )
    assert slices

    # ---- Phase 2: S4 labeling via the mock transport ----
    # Inject synthetic audio by clip_id (DictAudioSource) so we do not depend on
    # the repacked-tar reader path; the mock S4 client is deterministic.
    audio_source = s4_client.DictAudioSource(clip_audio_map(clips))
    total_ok = 0
    for sid in slices:
        entries = s4_client.load_slice_worklist(cfg, sid)
        rows = s4_client.run_slice(sid, cfg, audio_source=audio_source, worklist=entries)
        total_ok += sum(r.s4_status == S4Status.OK for r in rows)
        assert io_utils.is_done("s4", sid, cfg.paths.done)
    assert total_ok > 0

    # ---- S5: scoring + tiering + published meta ----
    s5_result = s5_score.run_s5(cfg, write=True)
    assert s5_result.n_kept > 0
    assert sum(s5_result.tier_counts.values()) == s5_result.n_kept
    assert s5_result.flat_parquet_path.exists()
    assert io_utils.is_done("s5", s5_score.S5_TASK_ID, cfg.paths.done)

    # Every published clip's JSON validates back into a MetaRecord (§7).
    json_files = sorted(s5_result.json_dir.glob("*.json"))
    assert len(json_files) == s5_result.n_kept
    rec = MetaRecord.model_validate(json.loads(json_files[0].read_text()))
    assert rec.selection is not None and rec.selection.tier in {"S", "A", "B"}
    assert rec.omni_labels is not None

    # Flat parquet is queryable and its tiered count matches.
    tiered = io_utils.query_parquet(
        "SELECT count(*) FROM m WHERE selection_tier IS NOT NULL",
        m=io_utils.parquet_glob(s5_score.s5_flat_parquet_path(cfg).parent),
    ).fetchall()[0][0]
    assert tiered == s5_result.n_kept

    # ---- HF packaging: non-empty tiered manifest + shard tars ----
    pack = hf_package.package_export(cfg, target_shard_bytes=10_000_000)
    assert pack.n_clips == s5_result.n_kept
    assert pack.manifest_path.exists()
    assert pack.shard_paths
    names = set()
    for sp in pack.shard_paths:
        with tarfile.open(sp) as t:
            names.update(t.getnames())
    # Each kept clip contributes a JSON member to the export shards.
    assert any(n.endswith(".json") for n in names)
    assert io_utils.is_done("pack", hf_package.PACK_TASK_ID, cfg.paths.done)

    # Upload degrades gracefully with no HF token.
    up = hf_package.upload_to_hf(cfg, "org/emilia-expressive-zh")
    assert up.uploaded is False

    # ---- Monitor: snapshot reflects the run ----
    snap = monitor.build_snapshot(cfg)
    stage_names = {s.stage for s in snap.stages}
    assert {"s0", "s1", "s2", "s3", "s4", "s5", "pack"} <= stage_names
    assert snap.s4_cost is not None and snap.s4_cost.labels_ok == total_ok
    assert sum(snap.tier_distribution.values()) == s5_result.n_kept
