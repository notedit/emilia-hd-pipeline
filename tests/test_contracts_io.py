"""Contract + IO-atomicity tests for the shared Foundation surface.

* Closed-vocabulary enums expose exactly the design §5.3 / §7 vocab and reject
  out-of-vocab values.
* ``flatten_meta`` / ``unflatten_meta`` round-trip a full :class:`MetaRecord`,
  including the prefix-collision-prone nested blocks and list leaves.
* A published-§7 JSON validates back into a :class:`MetaRecord`.
* Atomic writers leave no non-tmp file on an interrupted write, readers skip
  ``*.tmp``, and the done marker is only visible after the data rename.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from emilia_pipeline.common import contracts as C
from emilia_pipeline.common import io_utils
from emilia_pipeline.common.contracts import (
    Accent,
    Defect,
    EmotionPrimary,
    MetaRecord,
    Paralinguistic,
    Rhythm,
    Scenario,
    SpeakerVerdict,
    SpeakingStyle,
    TextVerdict,
    Tier,
    flatten_meta,
    unflatten_meta,
)

# ---------------------------------------------------------------------------
# Closed-vocabulary enums
# ---------------------------------------------------------------------------


def test_enum_vocab_exact() -> None:
    assert [e.value for e in EmotionPrimary] == [
        "neutral", "happy", "excited", "sad", "angry", "fearful",
        "surprised", "disgusted", "affectionate", "serious",
    ]
    assert [e.value for e in TextVerdict] == ["match", "fixable", "broken"]
    assert [e.value for e in SpeakerVerdict] == [
        "single", "intruded_trimmed", "intruded_rejected",
        "overlap_rejected", "degraded_pass",
    ]
    assert [e.value for e in Tier] == ["S", "A", "B"]
    assert [e.value for e in Rhythm] == ["steady", "varied", "dramatic"]
    assert [e.value for e in Accent] == ["standard", "accented", "dialect"]
    assert [e.value for e in Defect] == ["truncated_head", "truncated_tail", "artifact", "other"]
    assert [e.value for e in Scenario] == [
        "podcast", "audiobook", "drama", "interview", "lecture",
        "vlog", "customer_service", "other",
    ]
    assert [e.value for e in SpeakingStyle] == [
        "narration", "conversational", "storytelling", "speech",
        "broadcast", "acting", "vlog", "interview",
    ]
    assert [e.value for e in Paralinguistic] == [
        "laughter", "sigh", "crying", "breath_prominent", "filler_heavy", "disfluent",
    ]


def test_str_mixin_enums_compare_as_strings() -> None:
    assert EmotionPrimary.SAD == "sad"
    assert Tier.S == "S"


def test_guided_json_rejects_out_of_vocab() -> None:
    good = _guided_json()
    assert good.emotion.primary == "sad"
    with pytest.raises(Exception):
        _guided_json(emotion_primary="ecstatic")  # not in the closed vocab
    with pytest.raises(Exception):
        _guided_json(intensity=6)  # 1..5 only


# ---------------------------------------------------------------------------
# flatten / unflatten round-trip + published JSON
# ---------------------------------------------------------------------------


def test_flatten_unflatten_roundtrip() -> None:
    record = MetaRecord.model_validate(_META_EXAMPLE)
    flat = flatten_meta(record)
    # nested "_"-joined keys.
    assert flat["omni_labels_emotion_primary"] == "sad"
    assert flat["acoustics_dnsmos_p835_ovrl"] == 3.7
    assert flat["speaker_purity_check_verdict"] == "single"
    # list leaves stay lists.
    assert flat["pipeline_stages_passed"] == ["s0", "s1", "s2", "s3", "s4"]
    assert flat["omni_labels_paralinguistic"] == ["breath_prominent"]
    rebuilt = MetaRecord.model_validate(unflatten_meta(flat))
    assert rebuilt == record


def test_published_json_validates() -> None:
    # A §7 published record survives model_dump -> model_validate unchanged.
    record = MetaRecord.model_validate(_META_EXAMPLE)
    dumped = record.model_dump()
    assert MetaRecord.model_validate(dumped) == record
    assert dumped["selection"]["tier"] == "S"  # enum dumped as plain string


# ---------------------------------------------------------------------------
# IO atomicity / write discipline
# ---------------------------------------------------------------------------


def test_interrupted_write_leaves_no_final_file(tmp_path: Path, monkeypatch) -> None:
    # Simulate a crash between writing *.tmp and the atomic rename.
    target = tmp_path / "s1" / "part-0.parquet"

    def boom(_tmp, _final):
        raise RuntimeError("crash before rename")

    monkeypatch.setattr(io_utils, "_atomic_replace", boom)
    with pytest.raises(RuntimeError):
        io_utils.atomic_write_parquet([{"clip_id": "a"}], target)
    # No final (non-tmp) file exists; readers would see nothing.
    assert not target.exists()


def test_reader_skips_tmp_files(tmp_path: Path) -> None:
    stage = tmp_path / "s1"
    io_utils.atomic_write_parquet([{"clip_id": "a"}, {"clip_id": "b"}], stage / "part-0.parquet")
    # Drop a stray *.tmp that must be ignored by the standard glob.
    (stage / "part-9.parquet.tmp").write_bytes(b"garbage-not-parquet")
    n = io_utils.query_parquet(
        "SELECT count(*) FROM s1", s1=io_utils.parquet_glob(stage)
    ).fetchall()[0][0]
    assert n == 2


def test_done_marker_only_after_data(tmp_path: Path) -> None:
    stage = tmp_path / "s1"
    done = tmp_path / "done"
    io_utils.atomic_write_parquet([{"clip_id": "a"}], stage / "part-0.parquet")
    # Correct discipline: marker written only after the data file lands.
    assert not io_utils.is_done("s1", "0", done)
    io_utils.write_done_marker("s1", "0", done)
    assert io_utils.is_done("s1", "0", done)


def test_pending_is_all_minus_done(tmp_path: Path) -> None:
    done = tmp_path / "done"
    io_utils.write_done_marker("phase1", "00000", done)
    assert io_utils.pending_tasks(["00000", "00001", "00002"], "phase1", done) == ["00001", "00002"]


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def _guided_json(*, emotion_primary="sad", intensity=4):
    return C.S4GuidedJSON(
        text_verdict="fixable",
        text_fixed="a",
        text_punctuated="b",
        emotion=C.EmotionLabel(primary=emotion_primary, secondary=None,
                               intensity=intensity, confidence=0.8),
        prosody=C.ProsodyLabel(expressiveness=4, speaking_style="storytelling",
                               rhythm="dramatic", prominent_stress=True),
        context=C.ContextLabel(scenario="podcast", register="intimate", summary="s"),
        language=C.LanguageLabel(primary="zh", code_switch=False, accent="standard"),
        paralinguistic=[Paralinguistic.BREATH_PROMINENT],
        defects=[],
        usable=True,
    )


_META_EXAMPLE = {
    "clip_id": "c1",
    "schema_version": "1.3",
    "audio": {"path": "x", "duration_s": 8.4, "sample_rate": 24000,
              "channels": 1, "loudness_lufs": -19.3},
    "source": {"dataset": "emilia", "dataset_version": "v1", "original_id": "O",
               "original_text": "t", "original_speaker": "SP",
               "original_dnsmos": 3.4, "original_language": "zh"},
    "acoustics": {"aesthetics": {"pq": 7.8, "pc": 1.9, "ce": 7.1, "cu": 7.4},
                  "dnsmos_p835": {"sig": 3.9, "bak": 4.1, "ovrl": 3.7},
                  "snr_db": 28.4, "clipping_ratio": 0.0, "bandwidth_hz": 11200},
    "prosody_dsp": {"f0_mean_hz": 218.5, "f0_std_st": 4.2, "f0_range_st": 14.1,
                    "energy_std_db": 6.8, "speech_rate_cps": 4.9,
                    "rate_var_cv": 0.31, "pause_count": 2, "pause_total_ms": 610,
                    "f0_tracker_confidence": 0.94, "prosody_dsp_score": 0.81},
    "speaker": {"original_speaker": "SP",
                "embedding_ref": {"emb_file": "emb-1.npy", "emb_row": 187},
                "gender_pred": "female",
                "purity_check": {"n_windows": 9, "mean_win_cos": 0.91,
                                 "min_win_cos": 0.83, "f0_stability": 0.94,
                                 "verdict": "single", "intrusion_span_ms": None,
                                 "trimmed": False}},
    "omni_labels": {"model": "qwen", "prompt_version": "v3.1",
                    "text_verdict": "fixable", "text_fixed": "a",
                    "text_punctuated": "b", "cer_vs_original": 0.0,
                    "emotion": {"primary": "sad", "secondary": "affectionate",
                                "intensity": 4, "confidence": 0.86},
                    "prosody": {"expressiveness": 4, "speaking_style": "storytelling",
                                "rhythm": "dramatic", "prominent_stress": True},
                    "context": {"scenario": "podcast", "register": "intimate", "summary": "s"},
                    "language": {"primary": "zh", "code_switch": False, "accent": "standard"},
                    "paralinguistic": ["breath_prominent"], "defects": [], "usable": True},
    "selection": {"selection_score": 0.87, "tier": "S", "reject_reason": None},
    "pipeline": {"version": "v1.3", "stages_passed": ["s0", "s1", "s2", "s3", "s4"],
                 "processed_at": "2026-07-05T03:12:44Z"},
}
