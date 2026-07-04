"""Emilia 高表现力子集抽取 Pipeline (VoxSift / Emilia-ZH).

Foundation package: shared contracts, config, IO discipline, audio helpers,
mock-aware model / API factories, and synthetic-fixture generation.

Stage logic (S0-S5) and the workers live in the ``stages``, ``phase1``,
``phase2`` and ``scoring`` subpackages and are intentionally NOT implemented
here -- this module only ships the scaffolding every other component depends on.
"""

from __future__ import annotations

__version__ = "1.3.0"
SCHEMA_VERSION = "1.3"
PIPELINE_VERSION = "voxsift-emilia-v1.3"

__all__ = ["__version__", "SCHEMA_VERSION", "PIPELINE_VERSION"]
