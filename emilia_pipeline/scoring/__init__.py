"""S5 selection scoring, HuggingFace packaging and monitoring (design §7 / §6.4).

Modules:
  * :mod:`s5_score`   -- DuckDB join over S0..S4 parquet, selection scoring,
    tiering, stratified sampling, and final §7 meta-record assembly (flat
    parquet + per-clip published JSON).
  * :mod:`hf_package` -- shuffle + WebDataset tar/parquet packaging under
    ``export/`` and optional ``upload_large_folder`` to the Hub.
  * :mod:`phase1_hf`  -- short-circuit release of the Phase-1 *filtered* subset
    (S0-S3 survivors) straight to the same Hub repo on its own revision, without
    waiting for Phase-2 labeling / S5 scoring.
  * :mod:`monitor`    -- cron-style per-stage progress / S4 cost / distribution
    snapshot.
"""

from __future__ import annotations

from . import hf_package, monitor, phase1_hf, s5_score

__all__ = ["s5_score", "hf_package", "phase1_hf", "monitor"]
