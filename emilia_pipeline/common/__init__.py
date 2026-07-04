"""Foundation: shared config, contracts, IO discipline, audio, model factories.

Re-exports the most-used names so stage code can do
``from emilia_pipeline.common import Config, load_config, MetaRecord, get_model``.
"""

from __future__ import annotations

from .config import Config, load_config
from .contracts import (
    MetaRecord,
    S0PrefilterRow,
    S1AcousticsRow,
    S2ProsodyRow,
    S3SpeakerRow,
    S4GuidedJSON,
    S4LabelRow,
    SpeakerVerdict,
    Tier,
    flatten_meta,
    unflatten_meta,
)
from .models import (
    BaseAudioModel,
    BaseS4Client,
    content_hash,
    get_model,
    get_s4_client,
)

__all__ = [
    "Config",
    "load_config",
    "MetaRecord",
    "S0PrefilterRow",
    "S1AcousticsRow",
    "S2ProsodyRow",
    "S3SpeakerRow",
    "S4GuidedJSON",
    "S4LabelRow",
    "SpeakerVerdict",
    "Tier",
    "flatten_meta",
    "unflatten_meta",
    "BaseAudioModel",
    "BaseS4Client",
    "get_model",
    "get_s4_client",
    "content_hash",
]
