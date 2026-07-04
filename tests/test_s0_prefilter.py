"""S0 metadata-prefilter unit tests (pure metadata, no audio decode).

Covers the four §4-S0 gates, reason-token wiring, tolerant key normalization,
never-drop batch semantics, and a parquet round-trip. All deterministic; no
GPU / API / audio needed.
"""

from __future__ import annotations

from pathlib import Path

from emilia_pipeline.common import io_utils, synthesize
from emilia_pipeline.common.contracts import S0PrefilterRow
from emilia_pipeline.stages import s0_prefilter as s0


def _meta(**over):
    base = {"id": "o0", "text": "一二三四", "speaker": "SPK", "language": "zh",
            "duration": 6.0, "dnsmos": 3.5}
    base.update(over)
    return base


def test_all_gates_pass(base_config) -> None:
    row = s0.evaluate_s0(_meta(), clip_id="c0", shard="00000", config=base_config)
    assert isinstance(row, S0PrefilterRow)
    assert row.passed is True
    assert row.reject_reason is None
    assert (row.dur_ok, row.lang_ok, row.dnsmos_ok, row.text_ok) == (True, True, True, True)
    assert row.clip_id == "c0" and row.shard == "00000"


def test_duration_boundaries(base_config) -> None:
    # exactly at the min / max is inclusive; just outside fails.
    assert s0.evaluate_s0(_meta(duration=3.0), clip_id="c", shard="s", config=base_config).dur_ok
    assert s0.evaluate_s0(_meta(duration=20.0), clip_id="c", shard="s", config=base_config).dur_ok
    short = s0.evaluate_s0(_meta(duration=2.99), clip_id="c", shard="s", config=base_config)
    assert not short.dur_ok and short.reject_reason == s0.REASON_DURATION_SHORT
    long = s0.evaluate_s0(_meta(duration=20.01), clip_id="c", shard="s", config=base_config)
    assert not long.dur_ok and long.reject_reason == s0.REASON_DURATION_LONG


def test_missing_numeric_fields_fail_closed(base_config) -> None:
    row = s0.evaluate_s0(
        {"text": "一二三四", "language": "zh"}, clip_id="c", shard="s", config=base_config
    )
    # duration + dnsmos absent -> both recorded as missing, row not passing.
    assert not row.passed
    assert row.duration_s == 0.0  # default when duration absent
    reasons = set(row.reject_reason.split(";"))
    assert s0.REASON_DURATION_MISSING in reasons
    assert s0.REASON_DNSMOS_MISSING in reasons


def test_language_and_dnsmos_gates(base_config) -> None:
    lang = s0.evaluate_s0(_meta(language="en"), clip_id="c", shard="s", config=base_config)
    assert not lang.lang_ok and s0.REASON_LANGUAGE in lang.reject_reason
    low = s0.evaluate_s0(_meta(dnsmos=3.19), clip_id="c", shard="s", config=base_config)
    assert not low.dnsmos_ok and s0.REASON_DNSMOS_LOW in low.reject_reason


def test_text_char_count_ignores_whitespace(base_config) -> None:
    # 3 non-whitespace chars < min_text_chars(4) even with padding spaces.
    row = s0.evaluate_s0(_meta(text="  一 二 三 "), clip_id="c", shard="s", config=base_config)
    assert not row.text_ok and s0.REASON_TEXT_SHORT in row.reject_reason


def test_multiple_failing_gates_joined(base_config) -> None:
    row = s0.evaluate_s0(
        _meta(duration=1.0, language="en", dnsmos=1.0, text="x"),
        clip_id="c", shard="s", config=base_config,
    )
    tokens = row.reject_reason.split(";")
    assert s0.REASON_DURATION_SHORT in tokens
    assert s0.REASON_LANGUAGE in tokens
    assert s0.REASON_DNSMOS_LOW in tokens
    assert s0.REASON_TEXT_SHORT in tokens


def test_normalize_accepts_qualified_keys() -> None:
    norm = s0.normalize_emilia_meta(
        {"original_id": "X", "original_text": "hello", "original_speaker": "S",
         "original_language": "ZH", "duration_s": 5.5, "original_dnsmos": 3.9}
    )
    assert norm.original_id == "X"
    assert norm.original_language == "zh"  # lowercased
    assert norm.duration_s == 5.5 and norm.original_dnsmos == 3.9


def test_prefilter_shard_preserves_order_and_keeps_failures(synth_shard) -> None:
    cfg, _, clips = synth_shard
    rows = s0.prefilter_shard(
        [(c.key, c.meta) for c in clips], shard="00000", config=cfg
    )
    # never-drop: one row per input clip, order preserved.
    assert [r.clip_id for r in rows] == [c.key for c in clips]
    # the _short and _long edge clips must fail on duration.
    by_id = {r.clip_id: r for r in rows}
    assert not by_id["emilia_zh_00000_short"].passed
    assert not by_id["emilia_zh_00000_long"].passed


def test_s0_parquet_roundtrip(synth_shard, tmp_path: Path) -> None:
    cfg, _, clips = synth_shard
    rows = s0.prefilter_shard(
        [(c.key, c.meta) for c in clips], shard="00000", config=cfg
    )
    out = tmp_path / "s0" / "part-00000.parquet"
    io_utils.atomic_write_parquet([r.model_dump() for r in rows], out)
    n_pass = io_utils.query_parquet(
        "SELECT count(*) FROM s0 WHERE passed", s0=io_utils.parquet_glob(out.parent)
    ).fetchall()[0][0]
    assert n_pass == sum(r.passed for r in rows)
