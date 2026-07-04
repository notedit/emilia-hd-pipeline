"""Global ``prosody_dsp_score`` as a DuckDB SQL expression (design §4 S2).

The prosody richness score is a z-score-weighted sum of six raw metrics. The
z-scores MUST be taken over the whole surviving population, not per-shard, so a
clip's score does not depend on which shard it happened to land in. The raw
metrics are the only thing persisted (see
:class:`~emilia_pipeline.common.contracts.S2ProsodyRow`); this module builds the
window-function SQL that materializes the score globally at point of use.

Both :mod:`emilia_pipeline.phase1.repack` (for the top-fraction gate + labeling
priority) and :mod:`emilia_pipeline.scoring.s5_score` (for ``selection_score``)
import :func:`prosody_dsp_score_sql` so the two can never drift apart.

The per-metric z-score matches the Python reference in
:func:`emilia_pipeline.stages.s2_prosody._zscore`: a metric whose population
standard deviation is zero (or a single-row population) contributes 0, via
``COALESCE(.../NULLIF(stddev_pop, 0), 0)``. ``stddev_pop`` (population std,
divide-by-N) is used to match numpy's ``ndarray.std`` default.
"""

from __future__ import annotations

from .config import ProsodyZWeights

# The six richness metrics feeding the score, in ProsodyZWeights field order.
# Kept in sync with ``emilia_pipeline.stages.s2_prosody._SCORE_METRICS``.
_SCORE_METRICS = (
    "f0_std_st",
    "f0_range_st",
    "energy_std_db",
    "speech_rate_cps",
    "rate_var_cv",
    "pause_count",
)


def prosody_dsp_score_sql(
    weights: ProsodyZWeights, *, column_prefix: str = "", partition: str = ""
) -> str:
    """Build the SQL expression computing the global ``prosody_dsp_score``.

    Args:
        weights: The six z-score weights (:class:`ProsodyZWeights`).
        column_prefix: Optional table/CTE alias prefix for the metric columns,
            e.g. ``"s2."`` yields ``s2.f0_std_st``. Empty for bare column names.
        partition: Optional ``PARTITION BY`` body for the window (rarely needed;
            empty means the whole result set is one population -- "全体存活样本").

    Returns:
        A SQL scalar expression (no trailing alias) that must be evaluated in a
        SELECT whose row set is exactly the population to normalize over.
    """
    over = f"OVER (PARTITION BY {partition})" if partition else "OVER ()"
    weight_map = weights.model_dump()
    terms: list[str] = []
    for metric in _SCORE_METRICS:
        col = f"{column_prefix}{metric}"
        w = float(weight_map[metric])
        # (x - mean) / std, with a zero-variance / single-row metric -> 0.
        z = (
            f"COALESCE(({col} - avg({col}) {over}) "
            f"/ NULLIF(stddev_pop({col}) {over}, 0), 0)"
        )
        terms.append(f"({w!r} * {z})")
    return "(" + " + ".join(terms) + ")"


__all__ = ["prosody_dsp_score_sql", "_SCORE_METRICS"]
