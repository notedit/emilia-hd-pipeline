"""Shared fixtures for the mock/CI test tier (zero GPU / key / real data).

Everything here runs offline: audio is synthesized via
``emilia_pipeline.common.synthesize``, models/clients are forced to their
deterministic mock implementations, and every pipeline path is rerooted under a
per-test ``tmp_path`` so nothing touches the real data root.

Fixtures
--------
* ``config_path``    -- absolute path to ``configs/pipeline_v1.yaml``.
* ``base_config``    -- a mock-forcing :class:`Config` (``runtime.use_mocks=True``).
* ``tmp_config``     -- ``base_config`` with all data paths rerooted under ``tmp_path``.
* ``relaxed_config`` -- ``tmp_config`` with S1 gates opened and S3 verdicts
  widened so synthetic sine clips survive end-to-end (the funnel is calibrated
  for real audio, not tones).
* ``synth_shard``    -- ``(cfg, tar_path, clips)`` for a written synthetic shard.
* ``clip_audio_map`` -- ``{clip_id: (audio, sr)}`` for the synthetic shard, for
  injecting into the S4 :class:`DictAudioSource`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emilia_pipeline.common import synthesize
from emilia_pipeline.common.config import Config, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline_v1.yaml"

# Path fields on PathsConfig that tests reroot under a tmp data root.
_PATH_FIELDS = (
    "root",
    "source",
    "stage",
    "s0_prefilter",
    "s1_acoustics",
    "s2_prosody",
    "s3_speaker_features",
    "s3_speaker_embeddings",
    "s4_labels",
    "repacked",
    "manifests",
    "done",
    "failed",
    "export",
)


@pytest.fixture()
def config_path() -> Path:
    """Absolute path to the pipeline config YAML."""
    return CONFIG_PATH


@pytest.fixture()
def base_config() -> Config:
    """A mock-forcing :class:`Config` (no path rerooting)."""
    cfg = load_config(CONFIG_PATH)
    return cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"use_mocks": True})}
    )


def reroot_paths(cfg: Config, root: Path) -> Config:
    """Return a copy of ``cfg`` with every data path placed under ``root``."""
    updates = {field: root / field for field in _PATH_FIELDS}
    updates["root"] = root
    return cfg.model_copy(update={"paths": cfg.paths.model_copy(update=updates)})


@pytest.fixture()
def tmp_config(base_config: Config, tmp_path: Path) -> Config:
    """Mock config with all data paths rerooted under the test's ``tmp_path``."""
    return reroot_paths(base_config, tmp_path)


@pytest.fixture()
def relaxed_config(tmp_config: Config) -> Config:
    """Mock+rerooted config with the S1/S3 funnel opened for synthetic tones.

    The production thresholds are tuned for real 24 kHz speech; synthetic sine
    clips legitimately fail S1 bandwidth/SNR gates. Opening the gates lets the
    end-to-end mock test carry clips all the way to the export without changing
    any stage logic (pass/reject stays a query condition).
    """
    s1 = tmp_config.s1.model_copy(
        update={
            "min_aes_pq": 0.0,
            "max_aes_pc": 100.0,
            "min_snr_db": -100.0,
            "max_clipping_ratio": 1.0,
            "min_bandwidth_hz": 0.0,
        }
    )
    s5 = tmp_config.s5.model_copy(
        update={"allowed_verdicts": ["single", "intruded_trimmed", "degraded_pass"]}
    )
    return tmp_config.model_copy(update={"s1": s1, "s5": s5})


@pytest.fixture()
def synth_shard(tmp_config: Config):
    """Write a small synthetic Emilia shard; yield ``(cfg, tar_path, clips)``."""
    tmp_config.paths.source.mkdir(parents=True, exist_ok=True)
    tar, clips = synthesize.build_synthetic_shard(
        tmp_config.paths.source / "00000.tar", n_clips=6, shard_name="00000"
    )
    return tmp_config, tar, clips


@pytest.fixture()
def clip_audio_map():
    """Factory: ``clips -> {clip_id: (audio, sr)}`` for S4 DictAudioSource."""

    def _build(clips):
        return {c.key: (c.audio, c.sr) for c in clips}

    return _build
