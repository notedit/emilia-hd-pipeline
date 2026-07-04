"""Generic idempotent task dispatch (design §6.1).

The filesystem done-markers are the single source of truth: ``pending == all
tasks − done``. A queue backend (Redis or ``multiprocessing.Queue``, one
interface) is only a mailbox for handing task ids to workers -- if it is lost it
is rebuilt from the done-markers, so no state is authoritative except the disk.

Idempotency guarantees (§6.1):
  * ``dispatch(...)`` computes ``pending`` and enqueues it; re-running re-enqueues
    anything still lacking a done marker (so failures are naturally retried).
  * A worker double-checks the done marker before processing (``is_done``) and
    skips already-completed tasks, so an occasional duplicate delivery is a
    harmless no-op -- no heartbeats or leases needed.
  * On task failure the worker records ``failed/<stage>/<task>.json`` and does
    NOT re-enqueue; the next ``dispatch`` pass re-enqueues it.

This is stage-agnostic: it drives Phase-1 shard tasks and Phase-2 slice tasks
alike. The caller supplies ``all_tasks`` (e.g. via
``io_utils.enumerate_shard_tasks`` / ``enumerate_slice_tasks``) and a
``process`` callable ``(task_id) -> None`` that does the atomic writes.
"""

from __future__ import annotations

import abc
import multiprocessing as mp
import queue as _queue
from typing import Callable, Iterable, Optional

from ..common.config import Config
from ..common.io_utils import is_done, pending_tasks, write_done_marker, write_failed

# A stage's work function: given a task id, do the work + atomic writes. The
# done marker is written by the worker loop AFTER this returns successfully.
ProcessFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# Queue backend: one interface, Redis or multiprocessing.Queue
# ---------------------------------------------------------------------------


class TaskQueue(abc.ABC):
    """Minimal FIFO task mailbox. Rebuildable from done-markers, never truth."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all pending items for this stage (dispatch re-fills it)."""

    @abc.abstractmethod
    def push(self, task_ids: Iterable[str]) -> int:
        """Enqueue task ids in order. Returns the count pushed."""

    @abc.abstractmethod
    def pop(self, timeout: float = 0.0) -> Optional[str]:
        """Dequeue the next task id, or None when empty."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any backend resources."""


class MpTaskQueue(TaskQueue):
    """``multiprocessing.Queue``-backed mailbox for single-host runs.

    The queue is shared across worker processes (pass the same instance to each,
    or use a :class:`multiprocessing.Manager`). ``pop`` returns None once the
    queue is empty (and stays empty for ``timeout`` seconds).
    """

    # Bounded wait (s) used by a non-blocking pop when items are known to be
    # queued, to absorb the mp.Queue feeder-thread latency (get_nowait can raise
    # Empty immediately after put even though the item is in flight).
    _FEEDER_GRACE_S = 1.0

    def __init__(self, mp_queue: Optional["mp.Queue"] = None) -> None:
        self._q: "mp.Queue" = mp_queue if mp_queue is not None else mp.Queue()
        # Best-effort in-process count of enqueued-but-unpopped items. Accurate
        # within a single process (the dispatch_and_run path); cross-process
        # workers should pass poll_timeout > 0 to pop instead.
        self._pending = 0

    def reset(self) -> None:
        # Drain any residual items (fresh dispatch owns the contents).
        try:
            while True:
                self._q.get_nowait()
        except _queue.Empty:
            pass
        self._pending = 0

    def push(self, task_ids: Iterable[str]) -> int:
        count = 0
        for task_id in task_ids:
            self._q.put(str(task_id))
            count += 1
        self._pending += count
        return count

    def pop(self, timeout: float = 0.0) -> Optional[str]:
        effective = timeout
        if timeout <= 0 and self._pending > 0:
            effective = self._FEEDER_GRACE_S
        try:
            if effective > 0:
                item = self._q.get(timeout=effective)
            else:
                item = self._q.get_nowait()
        except _queue.Empty:
            return None
        self._pending = max(0, self._pending - 1)
        return item

    def close(self) -> None:
        self._q.close()


class RedisTaskQueue(TaskQueue):
    """Redis-list-backed mailbox (``rpush`` / ``lpop``) for multi-host runs.

    The key is ``q:{stage}``. Redis is only a mailbox: losing it costs nothing
    since ``dispatch`` rebuilds it from the filesystem done-markers (§6.1).
    """

    def __init__(self, stage: str, redis_url: str, *, client: object | None = None):
        self.stage = stage
        self.key = f"q:{stage}"
        if client is not None:
            self._r = client
        else:
            import redis  # lazy: only needed for the real multi-host path

            self._r = redis.Redis.from_url(redis_url)

    def reset(self) -> None:
        self._r.delete(self.key)

    def push(self, task_ids: Iterable[str]) -> int:
        items = [str(t) for t in task_ids]
        if not items:
            return 0
        self._r.rpush(self.key, *items)
        return len(items)

    def pop(self, timeout: float = 0.0) -> Optional[str]:
        if timeout > 0:
            result = self._r.blpop([self.key], timeout=int(timeout))
            value = None if result is None else result[1]
        else:
            value = self._r.lpop(self.key)
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def make_task_queue(stage: str, config: Config) -> TaskQueue:
    """Return a Redis-backed queue when ``runtime.redis_url`` is set, else mp.

    Args:
        stage: Stage token (``"s4"``, a Phase-1 stage name, ...); namespaces the
            Redis key.
        config: Pipeline config (consults ``runtime.redis_url``).

    Returns:
        A :class:`TaskQueue` with the uniform push/pop interface.
    """
    if config.runtime.redis_url:
        return RedisTaskQueue(stage, config.runtime.redis_url)
    return MpTaskQueue()


# ---------------------------------------------------------------------------
# Dispatch + worker loop
# ---------------------------------------------------------------------------


def compute_pending(
    all_tasks: Iterable[str], stage: str, config: Config
) -> list[str]:
    """Return ``sorted(all_tasks − done)`` for ``stage`` (design §6.1).

    Sorted order matters for Phase 2: slice ids are numbered by priority, so the
    pending order == labeling order == the anytime-optimality guarantee.
    """
    return pending_tasks(all_tasks, stage, config.paths.done)


def dispatch(
    all_tasks: Iterable[str],
    stage: str,
    config: Config,
    *,
    task_queue: Optional[TaskQueue] = None,
) -> tuple[TaskQueue, list[str]]:
    """Rebuild the queue from the done-markers and enqueue the pending set.

    Idempotent: resets the mailbox then pushes ``all_tasks − done`` in sorted
    (priority) order. Anything without a done marker -- including prior failures
    -- is re-enqueued, which is exactly the desired retry-on-rerun behavior.

    Args:
        all_tasks: The full task universe for the stage.
        stage: Stage token.
        config: Pipeline config (done dir + backend selection).
        task_queue: Reuse an existing queue; a new one is created when None.

    Returns:
        ``(task_queue, pending)`` -- the mailbox and the enqueued task-id list.
    """
    tq = task_queue or make_task_queue(stage, config)
    pending = compute_pending(all_tasks, stage, config)
    tq.reset()
    tq.push(pending)
    return tq, pending


def run_worker(
    stage: str,
    config: Config,
    process: ProcessFn,
    *,
    task_queue: TaskQueue,
    mark_done: bool = True,
    max_tasks: Optional[int] = None,
    poll_timeout: float = 0.0,
) -> list[str]:
    """Generic worker loop over a task queue (design §6.1 skeleton).

    For each dequeued task id: double-check the done marker (skip if present),
    call ``process(task_id)``, then write the done marker on success; on any
    exception record a failed marker and DO NOT re-enqueue (the next dispatch
    pass handles retries). This loop is identical for Phase-1 shard tasks and
    Phase-2 slice tasks.

    Args:
        stage: Stage token (done/failed namespacing).
        config: Pipeline config.
        process: ``(task_id) -> None`` performing the atomic stage work.
        task_queue: The shared mailbox to drain.
        mark_done: Write the done marker after ``process`` succeeds. Set False if
            ``process`` writes its own done marker (e.g. ``write_s4_rows`` does).
        max_tasks: Stop after this many tasks (for tests / bounded runs).
        poll_timeout: Blocking-pop timeout in seconds (0 = non-blocking, stop
            when the queue drains).

    Returns:
        The list of task ids this worker completed (excluding skips / failures).
    """
    completed: list[str] = []
    processed = 0
    while True:
        if max_tasks is not None and processed >= max_tasks:
            break
        task_id = task_queue.pop(timeout=poll_timeout)
        if task_id is None:
            break
        processed += 1
        # Double-check: an idempotent duplicate delivery is a harmless skip.
        if is_done(stage, task_id, config.paths.done):
            continue
        try:
            process(task_id)
            if mark_done:
                write_done_marker(stage, task_id, config.paths.done)
            completed.append(task_id)
        except Exception as exc:  # failure is recorded, not re-queued (§6.1)
            write_failed(
                stage,
                task_id,
                config.paths.failed,
                f"{type(exc).__name__}: {exc}",
            )
    return completed


def dispatch_and_run(
    all_tasks: Iterable[str],
    stage: str,
    config: Config,
    process: ProcessFn,
    *,
    mark_done: bool = True,
    max_tasks: Optional[int] = None,
) -> list[str]:
    """Single-process convenience: :func:`dispatch` then :func:`run_worker`.

    Used by tests and simple single-worker runs. Multi-worker / multi-host runs
    call :func:`dispatch` once (from the launcher) then spawn several processes
    each running :func:`run_worker` against the shared queue.
    """
    tq, _ = dispatch(all_tasks, stage, config)
    try:
        return run_worker(
            stage,
            config,
            process,
            task_queue=tq,
            mark_done=mark_done,
            max_tasks=max_tasks,
        )
    finally:
        tq.close()


__all__ = [
    "ProcessFn",
    "TaskQueue",
    "MpTaskQueue",
    "RedisTaskQueue",
    "make_task_queue",
    "compute_pending",
    "dispatch",
    "run_worker",
    "dispatch_and_run",
]
