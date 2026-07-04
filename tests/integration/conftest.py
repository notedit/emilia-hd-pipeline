"""Shared fixtures + skip guards for the opt-in real-integration test tier.

These tests hit *real* GPU models, a *real* cloud API, or *real* Emilia data.
They are SKIPPED by default (plain ``pytest``) and only run when the required
resource is actually available. Every guard below returns a human-readable skip
reason so a plain run shows the tests as *skipped*, never as *failed / errored*.

Selection knobs (all read at collection time):
  * ``torch.cuda.is_available()`` + configured on-disk model weights  -> ``gpu``
  * ``$DASHSCOPE_API_KEY`` (or whatever ``config.s4.api_key_env`` names) -> ``api``
  * ``$EMILIA_SAMPLE_DIR`` pointing at a dir with ``*.tar`` shards      -> ``integration`` e2e

The unit tier (``tests/*.py``) is fully mock-based and unaffected by any of this.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from emilia_pipeline.common.config import Config, load_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipeline_v1.yaml"


# ---------------------------------------------------------------------------
# Capability probes (each returns None when available, else a skip reason)
# ---------------------------------------------------------------------------


def cuda_skip_reason() -> Optional[str]:
    """Return a skip reason if a usable CUDA GPU is unavailable, else None."""
    try:
        import torch
    except Exception as exc:  # torch not importable
        return f"torch not importable: {exc}"
    try:
        if not torch.cuda.is_available():
            return "torch.cuda.is_available() is False (no usable GPU / driver)"
    except Exception as exc:  # driver mismatch etc.
        return f"CUDA probe failed: {exc}"
    return None


def weights_skip_reason(config: Config, *, kind: str) -> Optional[str]:
    """Return a skip reason if the real weights for ``kind`` are not on disk.

    Args:
        config: Pipeline config (``config.models.*`` holds the weight paths).
        kind: One of ``"aesthetics"``, ``"dnsmos"``, ``"campplus"``.
    """
    path = {
        "aesthetics": config.models.aesthetics_weights,
        "dnsmos": config.models.dnsmos_onnx,
        "campplus": config.models.campplus_weights,
    }[kind]
    if not path:
        return f"config.models weights for {kind!r} not configured (null)"
    if not os.path.exists(str(path)):
        return f"weights for {kind!r} not found on disk: {path}"
    return None


def api_key_skip_reason(config: Config) -> Optional[str]:
    """Return a skip reason if the S4 API key env var is unset, else None."""
    if config.api_key():
        return None
    return (
        f"env var {config.s4.api_key_env} is not set "
        "(no live DashScope / OpenAI-compatible key)"
    )


def emilia_sample_skip_reason() -> Optional[str]:
    """Return a skip reason if ``$EMILIA_SAMPLE_DIR`` is unusable, else None."""
    raw = os.environ.get("EMILIA_SAMPLE_DIR")
    if not raw:
        return "env var EMILIA_SAMPLE_DIR is not set (no real Emilia shards)"
    directory = Path(raw)
    if not directory.is_dir():
        return f"EMILIA_SAMPLE_DIR is not a directory: {raw}"
    if not sorted(directory.glob("*.tar")):
        return f"EMILIA_SAMPLE_DIR contains no *.tar shards: {raw}"
    return None


def emilia_sample_shards(limit: int = 2) -> list[Path]:
    """Return up to ``limit`` real Emilia ``*.tar`` shards from the sample dir."""
    directory = Path(os.environ["EMILIA_SAMPLE_DIR"])
    return sorted(directory.glob("*.tar"))[:limit]


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_config() -> Config:
    """Load the pipeline config with ``use_mocks=False`` (real code paths)."""
    cfg = load_config(CONFIG_PATH)
    return cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"use_mocks": False})}
    )


@pytest.fixture()
def real_config_tmp_paths(real_config: Config, tmp_path: Path) -> Config:
    """``real_config`` with every disk path redirected under a tmp dir.

    Lets the gated end-to-end test write stage parquet / npy / done markers
    without touching the production ``/data/emilia-expressive`` tree.
    """
    p = real_config.paths
    return real_config.model_copy(
        update={
            "paths": p.model_copy(
                update={
                    "root": tmp_path,
                    "stage": tmp_path / "stage",
                    "s0_prefilter": tmp_path / "stage" / "s0_prefilter",
                    "s1_acoustics": tmp_path / "stage" / "s1_acoustics",
                    "s2_prosody": tmp_path / "stage" / "s2_prosody",
                    "s3_speaker_features": tmp_path / "stage" / "s3_speaker" / "features",
                    "s3_speaker_embeddings": tmp_path / "stage" / "s3_speaker" / "embeddings",
                    "s4_labels": tmp_path / "stage" / "s4_labels",
                    "repacked": tmp_path / "repacked",
                    "manifests": tmp_path / "manifests",
                    "done": tmp_path / "done",
                    "failed": tmp_path / "failed",
                    "export": tmp_path / "export",
                }
            )
        }
    )
