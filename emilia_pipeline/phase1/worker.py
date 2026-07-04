"""Phase-1 fused scan worker: one task == one Emilia source tar shard.

Design §6.2. A single GPU worker (one process per ``CUDA_VISIBLE_DEVICES``)
sequentially reads a source shard, runs the inline S0 metadata whitelist, then a
CPU pool (``mp.Pool``, spawn, size ``total_cores // n_gpus``) that decodes each
S0-surviving clip and computes S2 prosody DSP features + the S1 CPU metrics. The
decoded audio is gathered back into the main process where the two GPU models
(Audiobox-Aesthetics + DNSMOS) score the whole batch at once. ``s1_pass`` then
short-circuits: clips that fail the strict acoustic gates keep their fully
computed S1 metrics but are excluded from S2/S3 accounting. Finally CAM++ runs a
sliding-window purity pass over the S1-surviving clips only, verdicts are
decided, and every stage's rows are buffered and written *once* at shard end:

    stage/s0_prefilter/part-{shard}.parquet      (all clips)
    stage/s1_acoustics/part-{shard}.parquet      (S0-pass clips)
    stage/s2_prosody/part-{shard}.parquet         (S1-pass clips)
    stage/s3_speaker/features/part-{shard}.parquet(S1-pass clips)
    stage/s3_speaker/embeddings/emb-{shard}.npy   (S1-pass clip mean embeddings)
    done/phase1/{shard}                           (marker; only after all writes)

Every GPU model sits behind the mock-aware factory (:mod:`common.models`), so
the whole worker runs on synthetic audio with zero GPU/key/data when
``config.runtime.use_mocks`` is set (or weights/GPU are absent). All writes are
atomic (``*.tmp`` then rename); the done marker is created only after every data
file lands. Re-running a completed shard is idempotent (atomic overwrite).
"""

from __future__ import annotations

import argparse
import enum
import io
import json
import multiprocessing as mp
import os
import tarfile
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
from pydantic import BaseModel

from ..common import audio as audio_utils
from ..common.config import Config, load_config
from ..common.contracts import (
    S0PrefilterRow,
    S1AcousticsRow,
    S2ProsodyRow,
    S3SpeakerRow,
)
from ..common.io_utils import (
    atomic_write_npy,
    atomic_write_parquet,
    is_done,
    write_done_marker,
    write_failed,
)
from ..stages import s1_acoustics, s2_prosody, s3_speaker
from ..stages.s0_prefilter import prefilter_shard

PHASE1_STAGE = "phase1"
# Audio member extensions recognised inside an Emilia/WebDataset tar.
_AUDIO_EXTS = (".flac", ".wav", ".ogg", ".mp3", ".opus")

__all__ = [
    "PHASE1_STAGE",
    "ClipInput",
    "CpuResult",
    "ShardResult",
    "read_shard_clips",
    "run_shard",
    "main",
]


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass
class ClipInput:
    """One raw clip read from the source tar (audio still encoded)."""

    clip_id: str
    audio_bytes: bytes
    meta: dict[str, Any]


@dataclass
class CpuResult:
    """Output of the per-clip CPU stage (decode + S2 features + S1 CPU metrics)."""

    clip_id: str
    audio: np.ndarray
    sr: int
    cpu_metrics: dict[str, float]
    features: s2_prosody.ProsodyFeatures


@dataclass
class ShardResult:
    """Everything produced for one shard (also what tests assert against)."""

    shard: str
    n_clips: int
    s0_rows: list[S0PrefilterRow]
    s1_rows: list[S1AcousticsRow]
    s2_rows: list[S2ProsodyRow]
    s3_rows: list[S3SpeakerRow]
    embeddings: np.ndarray
    paths: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tar reading (sequential, design §6.2 "tar 顺序读")
# ---------------------------------------------------------------------------


def _is_audio_member(name: str) -> bool:
    return name.lower().endswith(_AUDIO_EXTS)


def read_shard_clips(shard_path: str | os.PathLike[str]) -> list[ClipInput]:
    """Read a source tar into ordered :class:`ClipInput` records.

    Members are paired by stem: each audio member (``{key}.flac``) is matched to
    its sidecar ``{key}.json``. Audio-member order is preserved so the shard is
    processed in the order it is laid out on disk (sequential read).

    Args:
        shard_path: Path to the source ``*.tar`` shard.

    Returns:
        One :class:`ClipInput` per audio member that has a sidecar JSON, in tar
        order. Audio members without metadata are skipped.
    """
    clips: list[ClipInput] = []
    with tarfile.open(shard_path, "r") as tar:
        members = tar.getmembers()
        # Index the JSON sidecars by stem for O(1) pairing.
        json_by_stem: dict[str, tarfile.TarInfo] = {}
        for m in members:
            if m.isfile() and m.name.lower().endswith(".json"):
                json_by_stem[Path(m.name).stem] = m
        for m in members:
            if not (m.isfile() and _is_audio_member(m.name)):
                continue
            stem = Path(m.name).stem
            meta_info = json_by_stem.get(stem)
            if meta_info is None:
                continue
            audio_handle = tar.extractfile(m)
            meta_handle = tar.extractfile(meta_info)
            if audio_handle is None or meta_handle is None:
                continue
            meta = json.loads(meta_handle.read().decode("utf-8"))
            clips.append(
                ClipInput(clip_id=stem, audio_bytes=audio_handle.read(), meta=meta)
            )
    return clips


# ---------------------------------------------------------------------------
# CPU-pool stage: decode + S2 DSP + S1 CPU metrics (per clip, spawn-safe)
# ---------------------------------------------------------------------------

# Per-process worker state, populated by :func:`_pool_init`. Kept module-global
# so the mapped function needs no per-call tracker/VAD construction and remains
# picklable under the spawn start method.
_WORKER_STATE: dict[str, Any] = {}


def _pool_init(config: Config) -> None:
    """Pool initializer: build the (mock-aware) F0 tracker + VAD once per proc."""
    _WORKER_STATE["config"] = config
    _WORKER_STATE["f0"] = s2_prosody.get_f0_tracker(config)
    _WORKER_STATE["vad"] = s2_prosody.get_vad(config)


def _cpu_clip_task(item: tuple[int, str, bytes, str]) -> tuple[int, CpuResult]:
    """Decode one clip and compute its S2 features + S1 CPU metrics.

    Runs inside a CPU-pool process (or inline). Returns the original index so the
    caller can restore input order regardless of completion order.
    """
    index, clip_id, audio_bytes, text = item
    config: Config = _WORKER_STATE["config"]
    arr, sr = audio_utils.decode_bytes(audio_bytes)
    cpu_metrics = s1_acoustics.compute_cpu_metrics(arr, sr)
    features = s2_prosody.extract_prosody_features(
        arr,
        sr,
        text,
        config,
        f0_tracker=_WORKER_STATE["f0"],
        vad=_WORKER_STATE["vad"],
    )
    return index, CpuResult(
        clip_id=clip_id, audio=arr, sr=sr, cpu_metrics=cpu_metrics, features=features
    )


def _text_of(meta: Mapping[str, Any]) -> str:
    """Extract the reference text from an Emilia sidecar dict (tolerant keys)."""
    for key in ("original_text", "text", "transcript"):
        val = meta.get(key)
        if val:
            return str(val)
    return ""


def _run_cpu_stage(
    items: Sequence[tuple[int, str, bytes, str]],
    config: Config,
    *,
    parallel: bool,
) -> list[CpuResult]:
    """Map the CPU stage over ``items``, restoring input order.

    Uses an ``mp.Pool`` (spawn context, size ``total_cores // n_gpus``) when
    ``parallel`` and there is more than one item; otherwise runs inline (the
    order-restoring, deterministic path used by unit tests).
    """
    if not items:
        return []
    pool_size = max(1, config.runtime.resolved_cpu_per_gpu())
    if not parallel or len(items) == 1 or pool_size == 1:
        _pool_init(config)
        try:
            paired = [_cpu_clip_task(it) for it in items]
        finally:
            _WORKER_STATE.clear()
        paired.sort(key=lambda p: p[0])
        return [res for _, res in paired]

    ctx = mp.get_context(config.runtime.mp_start_method)
    workers = min(pool_size, len(items))
    with ctx.Pool(
        processes=workers, initializer=_pool_init, initargs=(config,)
    ) as pool:
        paired = list(pool.imap_unordered(_cpu_clip_task, items))
    paired.sort(key=lambda p: p[0])
    return [res for _, res in paired]


# ---------------------------------------------------------------------------
# Schema-safe parquet writing (empty shards still produce a readable file)
# ---------------------------------------------------------------------------


def _pa_type_for(annotation: Any) -> pa.DataType:
    """Map a pydantic field annotation to a pyarrow type (enum -> string)."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _pa_type_for(non_none[0]) if non_none else pa.string()
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return pa.string()
    if annotation is bool:
        return pa.bool_()
    if annotation is int:
        return pa.int64()
    if annotation is float:
        return pa.float64()
    if annotation is str:
        return pa.string()
    return pa.string()


def _schema_for(model_cls: type[BaseModel]) -> pa.Schema:
    """Build a pyarrow schema from a pydantic row model's fields."""
    return pa.schema(
        [(name, _pa_type_for(f.annotation)) for name, f in model_cls.model_fields.items()]
    )


def _write_rows(
    rows: Sequence[BaseModel], model_cls: type[BaseModel], path: Path
) -> Path:
    """Atomically write typed rows to parquet with an explicit schema.

    An explicit schema guarantees that even an empty shard writes a file with
    columns (a zero-column parquet breaks DuckDB ``union_by_name`` reads) and
    keeps Optional/enum columns correctly typed.
    """
    schema = _schema_for(model_cls)
    payload = [r.model_dump() for r in rows]
    table = pa.Table.from_pylist(payload, schema=schema)
    return atomic_write_parquet(table, path)


# ---------------------------------------------------------------------------
# Shard entrypoint
# ---------------------------------------------------------------------------


def run_shard(
    shard_path: str | os.PathLike[str],
    config: Config,
    *,
    parallel: bool = True,
    mark_done: bool = True,
    skip_if_done: bool = True,
) -> Optional[ShardResult]:
    """Process one source shard end-to-end (design §6.2 fused scan).

    Args:
        shard_path: Path to the source ``*.tar`` shard. Its stem is the shard
            token / Phase-1 task id.
        config: Pipeline config. ``runtime.use_mocks`` (or absent GPU/weights)
            routes every model through the deterministic mocks.
        parallel: Use the ``mp.Pool`` spawn CPU stage when True; run the CPU
            stage inline (deterministic, test-friendly) when False.
        mark_done: Write the ``done/phase1/{shard}`` marker after all data files
            land. The marker is the sole source of truth for completion.
        skip_if_done: Return ``None`` immediately if the done marker already
            exists (dispatch double-check, idempotent re-runs).

    Returns:
        A :class:`ShardResult`, or ``None`` when skipped as already-done.
    """
    shard = Path(shard_path).stem
    if skip_if_done and is_done(PHASE1_STAGE, shard, config.paths.done):
        return None

    clips = read_shard_clips(shard_path)

    # --- S0: inline metadata whitelist over every clip (never drops rows). ---
    s0_rows = prefilter_shard(
        [(c.clip_id, c.meta) for c in clips], shard=shard, config=config
    )
    passed_by_id = {r.clip_id: r.passed for r in s0_rows}
    s0_pass_clips = [c for c in clips if passed_by_id.get(c.clip_id, False)]

    # --- CPU pool: decode + S2 features + S1 CPU metrics for S0 survivors. ---
    cpu_items = [
        (i, c.clip_id, c.audio_bytes, _text_of(c.meta))
        for i, c in enumerate(s0_pass_clips)
    ]
    cpu_results = _run_cpu_stage(cpu_items, config, parallel=parallel)

    # --- GPU: Audiobox-Aesthetics + DNSMOS batch inference -> S1 rows. ---
    # The CPU pool already computed SNR / bandwidth / loudness during decode;
    # thread them through so compute_s1_rows does not recompute the DSP.
    s1_models = s1_acoustics.get_s1_models(config)
    try:
        gpu_clips: list[s1_acoustics.Clip] = [
            (r.clip_id, r.audio, r.sr) for r in cpu_results
        ]
        s1_rows = s1_acoustics.compute_s1_rows(
            gpu_clips,
            s1_models,
            config,
            shard=shard,
            cpu_metrics=[r.cpu_metrics for r in cpu_results],
        )
    finally:
        s1_models.close()

    # --- s1_pass short-circuit: only survivors continue to S2/S3 accounting. ---
    cpu_by_id = {r.clip_id: r for r in cpu_results}
    s1_survivors = [r for r in s1_rows if r.passed]

    # --- S2: store RAW richness metrics only. prosody_dsp_score is a
    # population-wide z-score, so it is NOT computed per-shard here (that would
    # make a clip's score depend on its shard); it is materialized globally in
    # DuckDB at repack / S5 time from these raw columns (design §4 S2). ---
    s2_rows: list[S2ProsodyRow] = []
    for s1_row in s1_survivors:
        feat = cpu_by_id[s1_row.clip_id].features
        s2_rows.append(
            S2ProsodyRow(
                clip_id=s1_row.clip_id,
                shard=shard,
                f0_mean_hz=feat.f0_mean_hz,
                f0_std_st=feat.f0_std_st,
                f0_range_st=feat.f0_range_st,
                energy_std_db=feat.energy_std_db,
                speech_rate_cps=feat.speech_rate_cps,
                rate_var_cv=feat.rate_var_cv,
                pause_count=feat.pause_count,
                pause_total_ms=feat.pause_total_ms,
                f0_tracker_confidence=feat.f0_tracker_confidence,
            )
        )

    # --- S3: CAM++ sliding-window purity over S1 survivors only. ---
    s3_rows: list[S3SpeakerRow] = []
    embeddings_list: list[np.ndarray] = []
    if s1_survivors:
        campp = s3_speaker.get_model(s3_speaker.MODEL_CAMPP, config)
        try:
            batch_audio = [
                (cpu_by_id[r.clip_id].audio, cpu_by_id[r.clip_id].sr)
                for r in s1_survivors
            ]
            original_speakers = [
                _speaker_of(s0_rows, r.clip_id) for r in s1_survivors
            ]
            f0_confs = [
                cpu_by_id[r.clip_id].features.f0_tracker_confidence
                for r in s1_survivors
            ]
            s3_results = s3_speaker.process_batch(
                batch_audio,
                clip_ids=[r.clip_id for r in s1_survivors],
                shard=shard,
                original_speakers=original_speakers,
                f0_tracker_confidences=f0_confs,
                cfg=config,
                model=campp,
            )
        finally:
            campp.close()
        emb_file = f"emb-{shard}.npy"
        for emb_row, res in enumerate(s3_results):
            s3_rows.append(res.with_embedding_ref(emb_file, emb_row))
            embeddings_list.append(np.asarray(res.embedding, dtype=np.float16).reshape(-1))

    embeddings = (
        np.stack(embeddings_list).astype(np.float16)
        if embeddings_list
        else np.zeros((0, config.s3.embedding_dim), dtype=np.float16)
    )

    paths = _persist_shard(
        shard=shard,
        config=config,
        s0_rows=s0_rows,
        s1_rows=s1_rows,
        s2_rows=s2_rows,
        s3_rows=s3_rows,
        embeddings=embeddings,
        write_emb=bool(embeddings_list),
        mark_done=mark_done,
    )

    return ShardResult(
        shard=shard,
        n_clips=len(clips),
        s0_rows=s0_rows,
        s1_rows=s1_rows,
        s2_rows=s2_rows,
        s3_rows=s3_rows,
        embeddings=embeddings,
        paths=paths,
    )


def _speaker_of(s0_rows: Sequence[S0PrefilterRow], clip_id: str) -> str:
    """Return the Emilia ``original_speaker`` for a clip from its S0 row."""
    for r in s0_rows:
        if r.clip_id == clip_id:
            return r.original_speaker
    return ""


def _persist_shard(
    *,
    shard: str,
    config: Config,
    s0_rows: Sequence[S0PrefilterRow],
    s1_rows: Sequence[S1AcousticsRow],
    s2_rows: Sequence[S2ProsodyRow],
    s3_rows: Sequence[S3SpeakerRow],
    embeddings: np.ndarray,
    write_emb: bool,
    mark_done: bool,
) -> dict[str, Path]:
    """Atomically write all stage outputs, then the done marker (design §3).

    The done marker is created strictly last, only after every data file has
    been renamed into place, so a crashed shard is never seen as complete.
    """
    paths: dict[str, Path] = {}
    paths["s0"] = _write_rows(
        s0_rows, S0PrefilterRow, config.paths.s0_prefilter / f"part-{shard}.parquet"
    )
    paths["s1"] = _write_rows(
        s1_rows, S1AcousticsRow, config.paths.s1_acoustics / f"part-{shard}.parquet"
    )
    paths["s2"] = _write_rows(
        s2_rows, S2ProsodyRow, config.paths.s2_prosody / f"part-{shard}.parquet"
    )
    paths["s3"] = _write_rows(
        s3_rows,
        S3SpeakerRow,
        config.paths.s3_speaker_features / f"part-{shard}.parquet",
    )
    if write_emb:
        paths["emb"] = atomic_write_npy(
            embeddings, config.paths.s3_speaker_embeddings / f"emb-{shard}.npy"
        )
    if mark_done:
        paths["done"] = write_done_marker(PHASE1_STAGE, shard, config.paths.done)
    return paths


# ---------------------------------------------------------------------------
# CLI: one process per GPU (reads CUDA_VISIBLE_DEVICES), pending == all - done
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint: process the shard's pending set for this GPU worker.

    Launched once per GPU (``CUDA_VISIBLE_DEVICES=k python -m
    emilia_pipeline.phase1.worker --config ...``). Enumerates source shards,
    subtracts the done set, and processes each pending shard; failures are
    recorded under ``failed/phase1/`` and never re-queued in-process (the next
    dispatch pass re-enqueues anything lacking a done marker, design §6.1).
    """
    from ..common.io_utils import enumerate_shard_tasks, pending_tasks

    parser = argparse.ArgumentParser(description="Emilia Phase-1 fused scan worker")
    parser.add_argument("--config", required=True, help="path to pipeline_v1.yaml")
    parser.add_argument(
        "--source", default=None, help="override source dir (default: config paths.source)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process at most N pending shards"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="run the CPU stage inline (no mp.Pool)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    source_dir = Path(args.source) if args.source else config.paths.source
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    tasks = enumerate_shard_tasks(source_dir)
    pending = pending_tasks(tasks, PHASE1_STAGE, config.paths.done)
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"[phase1] device={device!r} pending={len(pending)}/{len(tasks)} shards")
    processed = 0
    for shard in pending:
        shard_path = source_dir / f"{shard}.tar"
        try:
            run_shard(shard_path, config, parallel=not args.no_parallel)
            processed += 1
            print(f"[phase1] done shard={shard} ({processed}/{len(pending)})")
        except Exception as exc:  # noqa: BLE001 - record and continue
            write_failed(
                PHASE1_STAGE, shard, config.paths.failed, f"{type(exc).__name__}: {exc}"
            )
            print(f"[phase1] FAILED shard={shard}: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
