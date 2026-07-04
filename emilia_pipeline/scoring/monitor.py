"""Pipeline monitoring snapshot (design §6.4).

A cron-style, dependency-light snapshot intended to run every ~30 minutes. It
reports, without standing up any monitoring platform:

  * per-stage done / failed / pending counts (from done markers + failed JSONs);
  * an S4 token / cost estimate and the observed HTTP-429 (rate-limit) rate,
    derived from the failed-task records and the S4 label parquet;
  * rolling distributions over S3 ``verdict`` / S5 reject reasons and over the
    S4 ``emotion.primary`` labels.

A sudden distribution shift (e.g. an hour where ``emotion=neutral`` spikes) is
the design's flagged signal for prompt regression or a data-source anomaly.

Everything reads append-only parquet + marker files, so the snapshot is safe to
run concurrently with the workers and needs no GPU / API / lock.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..common.config import Config
from ..common.io_utils import (
    enumerate_shard_tasks,
    enumerate_slice_tasks,
    list_done_task_ids,
    parquet_glob,
    query_parquet,
)

# --- S4 cost model (approximate, tunable) -----------------------------------
# DashScope Qwen3-Omni is billed per token; audio input tokens dominate. These
# are order-of-magnitude estimates used only for the running cost readout.
# Approx tokens per second of 16k audio input (multimodal encoder).
S4_AUDIO_TOKENS_PER_SEC = 25.0
# Approx output tokens per full guided-JSON label.
S4_OUTPUT_TOKENS_PER_LABEL = 220.0
# Approx text-input tokens per request (system + few-shot + reference text).
S4_TEXT_INPUT_TOKENS_PER_REQ = 900.0
# USD per 1K tokens (input / output). Placeholder rates; adjust to the account.
S4_USD_PER_1K_INPUT = 0.0008
S4_USD_PER_1K_OUTPUT = 0.0020
# Substring markers used to detect a rate-limit failure in a failed-task record.
_RATE_LIMIT_MARKERS = ("429", "throttl", "rate limit", "ratelimit", "too many requests")


# ---------------------------------------------------------------------------
# Carriers
# ---------------------------------------------------------------------------


@dataclass
class StageProgress:
    """done / failed / pending counts for one stage."""

    stage: str
    total: int
    done: int
    failed: int
    pending: int

    @property
    def done_fraction(self) -> float:
        return (self.done / self.total) if self.total else 0.0


@dataclass
class S4CostSnapshot:
    """S4 token / cost estimate and rate-limit rate."""

    labels_ok: int
    labels_failed: int
    rate_limit_failures: int
    rate_limit_rate: float
    est_input_tokens: float
    est_output_tokens: float
    est_cost_usd: float


@dataclass
class MonitorSnapshot:
    """A full monitoring snapshot (JSON-serializable via :meth:`to_dict`)."""

    timestamp: str
    stages: list[StageProgress] = field(default_factory=list)
    s4_cost: Optional[S4CostSnapshot] = None
    verdict_distribution: dict[str, int] = field(default_factory=dict)
    emotion_distribution: dict[str, int] = field(default_factory=dict)
    reject_distribution: dict[str, int] = field(default_factory=dict)
    tier_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict form suitable for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "stages": [asdict(s) for s in self.stages],
            "s4_cost": asdict(self.s4_cost) if self.s4_cost else None,
            "verdict_distribution": self.verdict_distribution,
            "emotion_distribution": self.emotion_distribution,
            "reject_distribution": self.reject_distribution,
            "tier_distribution": self.tier_distribution,
        }


# ---------------------------------------------------------------------------
# Stage progress
# ---------------------------------------------------------------------------


def _count_failed(stage: str, failed_dir: Path) -> int:
    """Count failed-task JSON records for a stage."""
    d = failed_dir / stage
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.json"))


def stage_progress(config: Config) -> list[StageProgress]:
    """Compute done/failed/pending for every stage.

    Phase-1 stages (s0..s3) are keyed by source shard; s4 by worklist slice; s5
    and pack are single one-shot tasks. When the total is unknown (e.g. no
    source dir mounted) ``total`` falls back to the done count so the readout is
    still coherent.
    """
    done_dir = Path(config.paths.done)
    failed_dir = Path(config.paths.failed)

    shard_tasks = enumerate_shard_tasks(config.paths.source)
    slice_tasks = enumerate_slice_tasks(
        Path(config.paths.manifests) / "s4_worklist_v1.parquet"
    )

    plan: list[tuple[str, list[str]]] = [
        ("s0", shard_tasks),
        ("s1", shard_tasks),
        ("s2", shard_tasks),
        ("s3", shard_tasks),
        ("s4", slice_tasks),
        ("s5", ["all"]),
        ("pack", ["all"]),
    ]

    out: list[StageProgress] = []
    for stage, tasks in plan:
        done = list_done_task_ids(stage, done_dir)
        failed = _count_failed(stage, failed_dir)
        total = len(tasks) if tasks else len(done)
        pending = max(0, total - len(done))
        out.append(
            StageProgress(
                stage=stage,
                total=total,
                done=len(done),
                failed=failed,
                pending=pending,
            )
        )
    return out


# ---------------------------------------------------------------------------
# S4 cost / rate-limit
# ---------------------------------------------------------------------------


def _is_rate_limit(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _RATE_LIMIT_MARKERS)


def s4_cost_snapshot(config: Config) -> S4CostSnapshot:
    """Estimate S4 tokens / cost and the observed rate-limit failure rate.

    OK/failed label counts come from the s4 label parquet (``s4_status``); the
    rate-limit share is inferred from the per-slice failed JSON error strings
    and from any failed rows carrying a 429-flavored ``error``. Token/cost are
    coarse estimates from the module's cost model over labeled clips (using each
    clip's duration when available, else a nominal 8s).
    """
    s4_glob = parquet_glob(config.paths.s4_labels)
    labels_ok = 0
    labels_failed = 0
    rate_limit_rows = 0
    est_audio_seconds = 0.0

    if _glob_has_files(s4_glob):
        # Total audio seconds for labeled clips: join s4 clip_ids to s0 duration
        # when available; otherwise fall back to a nominal duration.
        s0_glob = parquet_glob(config.paths.s0_prefilter)
        try:
            if _glob_has_files(s0_glob):
                res = query_parquet(
                    """
                    SELECT s4.s4_status AS status, s4.error AS error,
                           COALESCE(s0.duration_s, 8.0) AS dur
                    FROM s4 LEFT JOIN s0 USING (clip_id)
                    """,
                    s4=s4_glob,
                    s0=s0_glob,
                )
            else:
                res = query_parquet(
                    "SELECT s4_status AS status, error AS error, 8.0 AS dur FROM s4",
                    s4=s4_glob,
                )
            for status, error, dur in res.fetchall():
                if str(status) == "ok":
                    labels_ok += 1
                    est_audio_seconds += float(dur or 8.0)
                else:
                    labels_failed += 1
                    if error and _is_rate_limit(str(error)):
                        rate_limit_rows += 1
        except Exception:  # pragma: no cover - malformed parquet during a write
            pass

    # Rate-limit signals also live in failed-task JSON records.
    failed_dir = Path(config.paths.failed) / "s4"
    if failed_dir.is_dir():
        for p in failed_dir.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if _is_rate_limit(str(rec.get("err", "")) + str(rec.get("error", ""))):
                rate_limit_rows += 1

    total_labeled = labels_ok + labels_failed
    rate = (rate_limit_rows / total_labeled) if total_labeled else 0.0

    est_input = (
        est_audio_seconds * S4_AUDIO_TOKENS_PER_SEC
        + labels_ok * S4_TEXT_INPUT_TOKENS_PER_REQ
    )
    est_output = labels_ok * S4_OUTPUT_TOKENS_PER_LABEL
    est_cost = (
        est_input / 1000.0 * S4_USD_PER_1K_INPUT
        + est_output / 1000.0 * S4_USD_PER_1K_OUTPUT
    )

    return S4CostSnapshot(
        labels_ok=labels_ok,
        labels_failed=labels_failed,
        rate_limit_failures=rate_limit_rows,
        rate_limit_rate=round(rate, 4),
        est_input_tokens=round(est_input, 1),
        est_output_tokens=round(est_output, 1),
        est_cost_usd=round(est_cost, 4),
    )


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def _distribution(glob: str, column: str) -> dict[str, int]:
    """Return a ``value -> count`` histogram for a column over a parquet glob."""
    if not _glob_has_files(glob):
        return {}
    try:
        res = query_parquet(
            f"SELECT {column} AS v, count(*) AS n FROM t "
            f"WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY n DESC",
            t=glob,
        )
    except Exception:  # pragma: no cover
        return {}
    return {str(v): int(n) for v, n in res.fetchall()}


def verdict_distribution(config: Config) -> dict[str, int]:
    """S3 ``verdict`` histogram over the speaker-features parquet."""
    return _distribution(parquet_glob(config.paths.s3_speaker_features), "verdict")


def emotion_distribution(config: Config) -> dict[str, int]:
    """S4 ``emotion.primary`` histogram over the label parquet (OK rows only)."""
    glob = parquet_glob(config.paths.s4_labels)
    if not _glob_has_files(glob):
        return {}
    try:
        res = query_parquet(
            "SELECT labels.emotion.primary AS v, count(*) AS n FROM t "
            "WHERE labels IS NOT NULL GROUP BY v ORDER BY n DESC",
            t=glob,
        )
    except Exception:  # pragma: no cover
        return {}
    return {str(v): int(n) for v, n in res.fetchall() if v is not None}


def reject_distribution(config: Config) -> dict[str, int]:
    """S5 ``selection.reject_reason`` histogram over the flat meta parquet."""
    from .s5_score import s5_flat_parquet_path

    glob = parquet_glob(s5_flat_parquet_path(config).parent)
    return _distribution(glob, "selection_reject_reason")


def tier_distribution(config: Config) -> dict[str, int]:
    """S5 ``selection.tier`` histogram over the flat meta parquet."""
    from .s5_score import s5_flat_parquet_path

    glob = parquet_glob(s5_flat_parquet_path(config).parent)
    return _distribution(glob, "selection_tier")


# ---------------------------------------------------------------------------
# Snapshot orchestration
# ---------------------------------------------------------------------------


def build_snapshot(config: Config) -> MonitorSnapshot:
    """Assemble a full :class:`MonitorSnapshot` from on-disk state.

    Args:
        config: Pipeline config.

    Returns:
        A populated snapshot (every sub-query degrades to empty on missing data).
    """
    return MonitorSnapshot(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        stages=stage_progress(config),
        s4_cost=s4_cost_snapshot(config),
        verdict_distribution=verdict_distribution(config),
        emotion_distribution=emotion_distribution(config),
        reject_distribution=reject_distribution(config),
        tier_distribution=tier_distribution(config),
    )


def write_snapshot(config: Config, snapshot: Optional[MonitorSnapshot] = None) -> Path:
    """Write a timestamped snapshot JSON under ``{root}/monitor/`` (atomic).

    Args:
        config: Pipeline config.
        snapshot: A pre-built snapshot; built from ``config`` when None.

    Returns:
        The path to the written snapshot JSON.
    """
    from ..common.io_utils import atomic_write_json

    snap = snapshot or build_snapshot(config)
    stamp = snap.timestamp.replace(":", "").replace("-", "")
    out_dir = Path(config.paths.root) / "monitor"
    out_path = out_dir / f"snapshot-{stamp}.json"
    atomic_write_json(snap.to_dict(), out_path)
    # Also refresh a stable ``latest.json`` pointer for dashboards / cron greps.
    atomic_write_json(snap.to_dict(), out_dir / "latest.json")
    return out_path


def format_snapshot(snapshot: MonitorSnapshot) -> str:
    """Render a compact human-readable text report of a snapshot."""
    lines = [f"=== pipeline snapshot {snapshot.timestamp} ==="]
    lines.append("stages:")
    for s in snapshot.stages:
        lines.append(
            f"  {s.stage:<5} done={s.done}/{s.total} "
            f"({s.done_fraction:5.1%}) failed={s.failed} pending={s.pending}"
        )
    if snapshot.s4_cost:
        c = snapshot.s4_cost
        lines.append(
            f"s4: ok={c.labels_ok} failed={c.labels_failed} "
            f"429_rate={c.rate_limit_rate:.1%} "
            f"~tokens(in/out)={c.est_input_tokens:.0f}/{c.est_output_tokens:.0f} "
            f"~cost=${c.est_cost_usd:.2f}"
        )
    if snapshot.verdict_distribution:
        lines.append(f"verdict: {snapshot.verdict_distribution}")
    if snapshot.emotion_distribution:
        lines.append(f"emotion: {snapshot.emotion_distribution}")
    if snapshot.tier_distribution:
        lines.append(f"tier: {snapshot.tier_distribution}")
    if snapshot.reject_distribution:
        lines.append(f"reject: {snapshot.reject_distribution}")
    return "\n".join(lines)


def _glob_has_files(glob: str) -> bool:
    """Return whether a parquet glob resolves to at least one file."""
    parent = Path(glob).parent
    pattern = Path(glob).name
    return parent.is_dir() and any(parent.glob(pattern))


__all__ = [
    "S4_AUDIO_TOKENS_PER_SEC",
    "S4_OUTPUT_TOKENS_PER_LABEL",
    "S4_TEXT_INPUT_TOKENS_PER_REQ",
    "S4_USD_PER_1K_INPUT",
    "S4_USD_PER_1K_OUTPUT",
    "StageProgress",
    "S4CostSnapshot",
    "MonitorSnapshot",
    "stage_progress",
    "s4_cost_snapshot",
    "verdict_distribution",
    "emotion_distribution",
    "reject_distribution",
    "tier_distribution",
    "build_snapshot",
    "write_snapshot",
    "format_snapshot",
]
