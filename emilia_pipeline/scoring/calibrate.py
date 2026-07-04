"""Phase-1 threshold calibration analysis (design §10).

Reads the Phase-1 stage parquet a real run produced and answers the question the
pilot exists to answer: *are the §4 initial thresholds right for real Emilia-ZH
data, and if not, what should they be?*

For each S1 acoustic gate and each S2 richness metric it reports:
  * the real quantile distribution (p1/p5/p10/p25/p50/p75/p90/p95/p99),
  * the kill-rate of the current config threshold (how many S0-survivors it
    rejects), broken down per gate and jointly,
  * a *suggested* threshold derived from a target survival rate (so you can pick
    "keep the cleanest X%") rather than guessing,
  * for DNSMOS, the correlation between our recomputed OVRL and Emilia's own
    ``dnsmos`` field (sanity check that our model agrees with the source).

It also dumps the S3 verdict distribution and the prosody_dsp_score distribution
(computed globally via the shared SQL, matching repack/S5).

This is read-only: it never changes the config. It prints a Markdown report and,
with ``--out``, writes it to a file. Threshold changes remain a human decision
(and, per the design, a config edit — never a pipeline re-run).

Usage:
    python -m emilia_pipeline.scoring.calibrate --config <yaml> [--target-survival 0.30] [--out report.md]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..common.config import Config, load_config
from ..common.io_utils import parquet_glob, query_parquet
from ..common.prosody_sql import prosody_dsp_score_sql

# S1 gates: (metric column, direction, config attribute). direction "min" means
# the gate keeps rows >= threshold; "max" keeps rows <= threshold.
_S1_GATES = [
    ("aes_pq", "min", "min_aes_pq"),
    ("aes_pc", "max", "max_aes_pc"),
    ("aes_ce", "min", "min_aes_ce"),
    ("snr_db", "min", "min_snr_db"),
    ("clipping_ratio", "max", "max_clipping_ratio"),
    ("bandwidth_hz", "min", "min_bandwidth_hz"),
]
# S1 metrics stored but not gated (kept for S5 scoring); report distribution
# only. dnsmos_* are NULL on rows scanned after DNSMOS was retired from S1.
_S1_UNGATED = ["aes_cu", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "loudness_lufs"]
_S2_METRICS = [
    "f0_std_st", "f0_range_st", "energy_std_db", "speech_rate_cps",
    "rate_var_cv", "pause_count", "pause_total_ms", "f0_tracker_confidence",
]
_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


@dataclass
class MetricDist:
    """Quantile distribution + basic stats for one numeric metric."""

    name: str
    n: int
    mean: float
    quantiles: dict[float, float]


def _dist(rel: str, column: str, where: str = "") -> Optional[MetricDist]:
    """Compute quantiles + mean for one column over a parquet glob."""
    clause = f"WHERE {where}" if where else ""
    q_list = ", ".join(str(q) for q in _QUANTILES)
    sql = (
        f"SELECT count(*) n, avg({column}) m, "
        f"quantile_cont({column}, [{q_list}]) qs FROM data {clause}"
    )
    row = query_parquet(sql, data=rel).fetchone()
    if not row or row[0] == 0 or row[2] is None:
        return None
    return MetricDist(
        name=column, n=int(row[0]), mean=float(row[1]),
        quantiles={q: float(v) for q, v in zip(_QUANTILES, row[2])},
    )


def _kill_rate(rel: str, column: str, direction: str, thr: float, where: str) -> tuple[int, int]:
    """Return (killed, total) for a gate at threshold ``thr`` over S0-survivors."""
    op = "<" if direction == "min" else ">"  # gate keeps >=(min)/<=(max); kill is the complement
    total = query_parquet(f"SELECT count(*) FROM data WHERE {where}", data=rel).fetchone()[0]
    killed = query_parquet(
        f"SELECT count(*) FROM data WHERE {where} AND {column} {op} {thr}", data=rel
    ).fetchone()[0]
    return int(killed), int(total)


def _suggest_threshold(dist: MetricDist, direction: str, target_survival: float) -> float:
    """Suggest a threshold keeping ~``target_survival`` fraction of the metric.

    For a "min" gate we keep the top fraction, so the cut is the (1-target)
    quantile; for a "max" gate we keep the bottom fraction, so the cut is the
    ``target`` quantile. Interpolates between the computed quantile points.
    """
    frac = 1.0 - target_survival if direction == "min" else target_survival
    return _interp_quantile(dist, frac)


def _interp_quantile(dist: MetricDist, frac: float) -> float:
    """Linearly interpolate the metric value at cumulative fraction ``frac``."""
    qs = sorted(dist.quantiles)
    if frac <= qs[0]:
        return dist.quantiles[qs[0]]
    if frac >= qs[-1]:
        return dist.quantiles[qs[-1]]
    for i in range(1, len(qs)):
        if frac <= qs[i]:
            lo, hi = qs[i - 1], qs[i]
            t = (frac - lo) / (hi - lo)
            return dist.quantiles[lo] + t * (dist.quantiles[hi] - dist.quantiles[lo])
    return dist.quantiles[qs[-1]]


def _fmt_dist(d: MetricDist) -> str:
    """One-line quantile summary."""
    q = d.quantiles
    return (
        f"n={d.n} mean={d.mean:.3g} | "
        f"p1={q[0.01]:.3g} p5={q[0.05]:.3g} p10={q[0.10]:.3g} p25={q[0.25]:.3g} "
        f"p50={q[0.50]:.3g} p75={q[0.75]:.3g} p90={q[0.90]:.3g} p95={q[0.95]:.3g} p99={q[0.99]:.3g}"
    )


def build_report(config: Config, target_survival: float) -> str:
    """Build the Markdown calibration report from the Phase-1 parquet."""
    p = config.paths
    s0 = parquet_glob(p.s0_prefilter)
    s1 = parquet_glob(p.s1_acoustics)
    s2 = parquet_glob(p.s2_prosody)
    s3 = parquet_glob(p.s3_speaker_features)

    lines: list[str] = []
    add = lines.append
    add("# Phase-1 阈值校准报告\n")
    add(f"目标存活率参考: **{target_survival:.0%}** (用于反推建议阈值)\n")

    # ---- S0 survival ----
    n0 = query_parquet("SELECT count(*) FROM d", d=s0).fetchone()[0]
    n0p = query_parquet("SELECT count(*) FROM d WHERE passed", d=s0).fetchone()[0]
    add("## S0 元数据预筛\n")
    add(f"- 总 clip: **{n0}**, S0 存活: **{n0p}** ({100*n0p/max(1,n0):.1f}%)\n")
    add("- S0 拒绝原因:\n")
    for r in query_parquet(
        "SELECT reject_reason, count(*) c FROM d WHERE NOT passed GROUP BY 1 ORDER BY c DESC", d=s0
    ).fetchall():
        add(f"  - `{r[0]}`: {r[1]}")
    add("")

    # ---- S1 gates ----
    add("## S1 声学门 (真实分布 vs §4 初值)\n")
    add("| 指标 | 方向 | 当前阈值 | 卡掉/总 | 卡掉率 | 建议阈值(目标存活) | 真实分位 |")
    add("|---|---|---|---|---|---|---|")
    s1_where = "passed IS NOT NULL"  # all S1 rows == S0 survivors
    for col, direction, attr in _S1_GATES:
        thr = getattr(config.s1, attr)
        d = _dist(s1, col)
        if d is None:
            continue
        killed, total = _kill_rate(s1, col, direction, thr, s1_where)
        suggest = _suggest_threshold(d, direction, target_survival)
        q = d.quantiles
        add(
            f"| {col} | {direction} | {thr:g} | {killed}/{total} | "
            f"{100*killed/max(1,total):.0f}% | **{suggest:.3g}** | "
            f"p10={q[0.10]:.3g} p50={q[0.50]:.3g} p90={q[0.90]:.3g} |"
        )
    add("")

    # joint survival under current thresholds
    joint = " AND ".join(
        f"{col} {'>=' if d=='min' else '<='} {getattr(config.s1, a):g}"
        for col, d, a in _S1_GATES
    )
    n1_joint = query_parquet(f"SELECT count(*) FROM d WHERE {joint}", d=s1).fetchone()[0]
    add(f"- **当前所有 S1 门联合存活: {n1_joint}/{n0p} ({100*n1_joint/max(1,n0p):.1f}% of S0-pass)**\n")

    # ---- S1 ungated metrics ----
    add("## S1 未设门的指标 (入库供 S5 评分)\n")
    for col in _S1_UNGATED:
        d = _dist(s1, col)
        if d:
            add(f"- **{col}**: {_fmt_dist(d)}")
    add("")

    # ---- DNSMOS agreement with Emilia's own dnsmos ----
    add("## DNSMOS 一致性 (我们的 OVRL vs Emilia 自带 dnsmos)\n")
    corr_row = query_parquet(
        "SELECT corr(s1.dnsmos_ovrl, s0.original_dnsmos), "
        "avg(s1.dnsmos_ovrl - s0.original_dnsmos) "
        "FROM s1 JOIN s0 USING(clip_id) WHERE s0.original_dnsmos IS NOT NULL",
        s1=s1, s0=s0,
    ).fetchone()
    if corr_row and corr_row[0] is not None:
        add(f"- 皮尔逊相关: **{corr_row[0]:.3f}**, 平均差(ours-emilia): {corr_row[1]:+.3f}\n")

    # ---- S2 richness ----
    add("## S2 韵律指标分布\n")
    for col in _S2_METRICS:
        d = _dist(s2, col)
        if d:
            add(f"- **{col}**: {_fmt_dist(d)}")
    add("")

    # prosody_dsp_score (global, matches repack/S5)
    score_expr = prosody_dsp_score_sql(config.s2.z_weights, column_prefix="s2.")
    try:
        score_rel = query_parquet(
            f"SELECT {score_expr} AS score FROM s2", s2=s2
        )
        import tempfile
        # materialize then quantile (window fn already global over s2)
        rows = [r[0] for r in score_rel.fetchall()]
        if rows:
            rows.sort()
            import statistics
            def qtile(frac):
                i = min(len(rows) - 1, int(frac * len(rows)))
                return rows[i]
            add("## prosody_dsp_score (全局 z-score, 匹配 repack/S5)\n")
            add(
                f"- n={len(rows)} mean={statistics.mean(rows):.3g} | "
                f"p10={qtile(0.1):.3g} p50={qtile(0.5):.3g} p90={qtile(0.9):.3g}"
            )
            keep = 1.0 - config.s2.top_fraction
            add(f"- top_fraction={config.s2.top_fraction:g} → 分位阈值(保留top): {qtile(keep):.3g}\n")
    except Exception as exc:  # pragma: no cover - defensive
        add(f"- (prosody score 计算跳过: {exc})\n")

    # ---- S3 verdicts ----
    add("## S3 声纹 verdict 分布\n")
    for r in query_parquet(
        "SELECT verdict, count(*) c FROM d GROUP BY 1 ORDER BY c DESC", d=s3
    ).fetchall():
        add(f"- `{r[0]}`: {r[1]}")
    add("")

    add("---\n")
    add("> 阈值变更是人工决策 + config 编辑,不重跑管线(数值已全部入库,改 SQL 即可重放)。")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-1 threshold calibration analysis")
    parser.add_argument("--config", required=True, help="pipeline yaml (points at the run's paths)")
    parser.add_argument(
        "--target-survival", type=float, default=0.30,
        help="target survival fraction used to suggest per-gate thresholds",
    )
    parser.add_argument("--out", default=None, help="write the Markdown report to this path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = build_report(config, args.target_survival)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n[calibrate] report written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
