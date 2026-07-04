"""Atomic IO discipline, done/failed markers, task enumeration, DuckDB helpers.

Global write discipline (design §3): every parquet / npy is written to a
``*.tmp`` sibling then ``os.rename``'d into place (atomic on POSIX); a done
marker is created only AFTER the data file rename succeeds. Readers ignore
``*.tmp`` files. Repeating a task is idempotent (same-name atomic overwrite).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TMP_SUFFIX = ".tmp"


# ---------------------------------------------------------------------------
# Low-level atomic primitives
# ---------------------------------------------------------------------------


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """fsync ``tmp_path`` then atomically rename it to ``final_path``."""
    with open(tmp_path, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)


def atomic_write_bytes(data: bytes, path: str | os.PathLike[str]) -> Path:
    """Write raw bytes atomically (``*.tmp`` then rename)."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + TMP_SUFFIX)
    with open(tmp_path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def atomic_write_json(obj: Any, path: str | os.PathLike[str]) -> Path:
    """Serialize ``obj`` to JSON and write it atomically (UTF-8, unescaped)."""
    payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    return atomic_write_bytes(payload, path)


# ---------------------------------------------------------------------------
# Parquet / npy atomic writers
# ---------------------------------------------------------------------------


def atomic_write_parquet(
    rows: Sequence[Mapping[str, Any]] | pa.Table,
    path: str | os.PathLike[str],
    *,
    compression: str = "zstd",
) -> Path:
    """Write rows to a parquet file atomically.

    Args:
        rows: A sequence of flat dict rows, or a pre-built ``pyarrow.Table``.
        path: Destination ``.parquet`` path.
        compression: Parquet codec (default ``zstd``).

    Returns:
        The final path written.
    """
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + TMP_SUFFIX)

    table = rows if isinstance(rows, pa.Table) else pa.Table.from_pylist(list(rows))
    pq.write_table(table, tmp_path, compression=compression)
    _atomic_replace(tmp_path, final_path)
    return final_path


def atomic_write_npy(arr: np.ndarray, path: str | os.PathLike[str]) -> Path:
    """Write a numpy array to ``.npy`` atomically.

    Args:
        arr: Array to persist (e.g. ``(n_clips, 192)`` fp16 embeddings).
        path: Destination ``.npy`` path.

    Returns:
        The final path written.
    """
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + TMP_SUFFIX)
    with open(tmp_path, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def read_npy(path: str | os.PathLike[str], *, mmap: bool = False) -> np.ndarray:
    """Load a ``.npy`` array, optionally memory-mapped for random-row access."""
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


# ---------------------------------------------------------------------------
# Done / failed markers
# ---------------------------------------------------------------------------


def _marker_path(done_dir: str | os.PathLike[str], stage: str, task_id: str) -> Path:
    return Path(done_dir) / stage / task_id


def write_done_marker(
    stage: str, task_id: str, done_dir: str | os.PathLike[str]
) -> Path:
    """Create the done marker ``{done_dir}/{stage}/{task_id}``.

    Call this ONLY after the stage's data files have been renamed into place.
    """
    marker = _marker_path(done_dir, stage, task_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return marker


def is_done(stage: str, task_id: str, done_dir: str | os.PathLike[str]) -> bool:
    """Return whether the done marker for ``(stage, task_id)`` exists."""
    return _marker_path(done_dir, stage, task_id).exists()


def write_failed(
    stage: str,
    task_id: str,
    failed_dir: str | os.PathLike[str],
    error: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Record a task failure as ``{failed_dir}/{stage}/{task_id}.json``.

    The task is NOT re-queued here; the next dispatch pass re-enqueues anything
    that lacks a done marker (design §6.1).
    """
    payload: dict[str, Any] = {
        "stage": stage,
        "task_id": task_id,
        "err": error,
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    return atomic_write_json(payload, Path(failed_dir) / stage / f"{task_id}.json")


# ---------------------------------------------------------------------------
# Task enumeration / pending-set (dispatch)
# ---------------------------------------------------------------------------


def list_done_task_ids(stage: str, done_dir: str | os.PathLike[str]) -> set[str]:
    """Return the set of completed task IDs for ``stage`` (marker file names)."""
    stage_dir = Path(done_dir) / stage
    if not stage_dir.is_dir():
        return set()
    return {p.name for p in stage_dir.iterdir() if not p.name.endswith(TMP_SUFFIX)}


def pending_tasks(
    all_tasks: Iterable[str], stage: str, done_dir: str | os.PathLike[str]
) -> list[str]:
    """Compute the pending set = all_tasks - done, sorted (design §6.1).

    Sorting matters for Phase 2: slice IDs are numbered in priority order, so
    sorted pending == labeling order == the anytime-optimality guarantee.
    """
    done = list_done_task_ids(stage, done_dir)
    return sorted(set(all_tasks) - done)


def enumerate_shard_tasks(source_dir: str | os.PathLike[str]) -> list[str]:
    """List Phase-1 task IDs: one per Emilia source ``*.tar`` shard (stems)."""
    src = Path(source_dir)
    if not src.is_dir():
        return []
    return sorted(p.stem for p in src.glob("*.tar"))


def enumerate_slice_tasks(worklist_parquet: str | os.PathLike[str]) -> list[str]:
    """List Phase-2 task IDs: distinct ``slice_id`` values in the worklist."""
    path = Path(worklist_parquet)
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["slice_id"])
    slices = table.column("slice_id").to_pylist()
    return sorted({str(s) for s in slices})


# ---------------------------------------------------------------------------
# DuckDB query helpers (connect over a glob of non-tmp parquet)
# ---------------------------------------------------------------------------


def parquet_glob(stage_dir: str | os.PathLike[str], pattern: str = "part-*.parquet") -> str:
    """Return a glob string matching a stage's part files (excludes ``*.tmp``)."""
    return str(Path(stage_dir) / pattern)


def duckdb_connect(read_only: bool = False):  # noqa: ANN201 - duckdb type is dynamic
    """Open an in-memory DuckDB connection.

    DuckDB is the pipeline's query engine over append-only parquet (design §1).
    Imported lazily so importing this module never requires duckdb.
    """
    import duckdb

    return duckdb.connect(database=":memory:", read_only=read_only)


def query_parquet(sql: str, **relations: str):  # noqa: ANN201 - duckdb dynamic
    """Run a SQL query, binding named parquet globs as views.

    Example:
        ``query_parquet("SELECT count(*) FROM s1", s1=parquet_glob(dir))``
        registers a view ``s1`` over the glob and runs the query.

    Args:
        sql: A DuckDB SQL statement referencing the bound relation names.
        **relations: name -> parquet glob string. Each is exposed as a view via
            ``read_parquet``. ``*.tmp`` files never match the standard glob.

    Returns:
        A ``duckdb.DuckDBPyRelation`` result; call ``.fetchall()`` / ``.df()``.
    """
    con = duckdb_connect()
    for name, glob in relations.items():
        # read_parquet cannot be a bound parameter inside a prepared statement,
        # so inline the glob with single-quote escaping. Names are internal.
        safe = str(glob).replace("'", "''")
        con.execute(
            f"CREATE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{safe}', union_by_name=true)"
        )
    return con.execute(sql)


__all__ = [
    "TMP_SUFFIX",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_npy",
    "read_npy",
    "write_done_marker",
    "is_done",
    "write_failed",
    "list_done_task_ids",
    "pending_tasks",
    "enumerate_shard_tasks",
    "enumerate_slice_tasks",
    "parquet_glob",
    "duckdb_connect",
    "query_parquet",
]
