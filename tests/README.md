# Test tiers

This repo has two clearly separated test tiers.

## 1. Unit tier (default) — `tests/*.py`

Fully mock-based. Every GPU model and every network/API dependency sits behind a
lazy factory with a deterministic MOCK fallback, so these tests need **zero GPU,
zero API key, and zero real data**. This is what runs in CI and on any laptop.

Deliberately minimal (basic functional verification only):

- `test_basics.py`            — config load, Phase-1 worker core invariants on a
  synthetic shard (row counts / done marker / DNSMOS short-circuit / idempotency),
  S1 pass predicate.
- `test_e2e_mock_pipeline.py` — full pipeline on one synthetic shard:
  S0..S3 worker -> repack -> S4 (mock transport) -> S5 -> HF packaging -> monitor.

Run it:

```bash
pytest -m "not gpu and not api and not integration"
```

## 2. Real-integration tier (opt-in) — `tests/integration/`

These hit **real** GPU models, a **real** cloud API, or **real** Emilia shards.
They are **SKIPPED by default**: a plain `pytest` shows them as *skipped* (with a
human-readable reason), never as *failed / errored*. Each test probes its
required resource at runtime and calls `pytest.skip(reason)` when it is absent.

Markers (registered in `pytest.ini`): `gpu`, `api`, `integration`, `slow`.

### `gpu` — real Audiobox-Aesthetics / DNSMOS-onnx / CAM++

File: `tests/integration/test_gpu_models.py`

Runs the real models that `get_model(...)` / `s1_acoustics.get_s1_models(...)`
select when `use_mocks=False` AND CUDA is available AND the configured weights
exist on disk. Asserts outputs are finite and in their documented ranges
(Aesthetics `[0,10]`, DNSMOS P.835 `[1,5]`, CAM++ cosine `[-1,1]`, embedding dim
`192`) and that the **real row schema is identical to the mock row schema** so
downstream parquet / DuckDB code is agnostic to which tier produced the row.

Skip guards: `torch` importable + `torch.cuda.is_available()` +
`config.models.{aesthetics_weights,dnsmos_onnx,campplus_weights}` present on disk.

Requirements to actually run:
- A CUDA GPU with a compatible driver.
- The weight paths wired into `configs/pipeline_v1.yaml` under `models.*`.

```bash
pytest tests/integration -m gpu -v
```

### `api` — real Qwen3-Omni via DashScope (OpenAI-compatible)

File: `tests/integration/test_api_qwen_omni.py`

Drives the real DashScope transport (base64 audio, guided-JSON, retry/backoff)
end-to-end with ONE tiny 2 s clip. Asserts the response validates against the
§5.3 guided-JSON schema (`S4GuidedJSON`) and that `temperature=0` decoding is
deterministic (two identical requests yield identical labels).

These make **real, billable API calls** — kept intentionally tiny and minimal.

Skip guard: the env var named by `config.s4.api_key_env` (default
`DASHSCOPE_API_KEY`) must be set. Absent key -> skipped with a clear reason.

Required env var:
- `DASHSCOPE_API_KEY` — a live DashScope / OpenAI-compatible key.

```bash
export DASHSCOPE_API_KEY=sk-...          # your live key
pytest tests/integration -m api -v
```

### `integration` (+ `slow`) — small end-to-end on real Emilia shards

File: `tests/integration/test_e2e_real_shards.py`

Runs the real Phase-1 fused scan over a couple of actual Emilia `*.tar` shards,
then the repack -> S4 -> S5 tail, writing every artifact under a tmp dir (never
touching production). Verifies the atomic-write discipline (no leftover `*.tmp`),
done-marker idempotency, and DuckDB queryability. GPU models fall back to mocks
when no GPU is present, so this validates the plumbing even on a CPU box; on a
GPU box with weights configured it also exercises the real models. S4 uses the
mock client unless `DASHSCOPE_API_KEY` is also set.

Skip guard: `$EMILIA_SAMPLE_DIR` must point at a directory containing `*.tar`
shards. Absent / empty -> skipped with a clear reason.

Required env var:
- `EMILIA_SAMPLE_DIR` — directory holding a few real Emilia `*.tar` shards.

```bash
EMILIA_SAMPLE_DIR=/path/to/emilia/shards pytest tests/integration -m integration -v
```

### Running the whole real tier at once

```bash
# with all resources available (GPU + key + shards)
export DASHSCOPE_API_KEY=sk-...
EMILIA_SAMPLE_DIR=/path/to/emilia/shards \
  pytest tests/integration -m "gpu or api or integration" -v
```

Any resource that is missing simply produces skips, so this command is safe to
run anywhere.

## Other env vars

- `HF_TOKEN` — used by `emilia_pipeline.scoring.hf_package.upload_to_hf` for the
  final HuggingFace dataset upload. Not required by any test; `upload_to_hf`
  skips gracefully when it is unset. Set it only when doing a real publish.

## Quick reference

| Tier / marker | Command | Needs |
|---|---|---|
| unit (default) | `pytest -m "not gpu and not api and not integration"` | nothing |
| gpu | `pytest tests/integration -m gpu` | CUDA + weights in `configs/pipeline_v1.yaml` |
| api | `pytest tests/integration -m api` | `DASHSCOPE_API_KEY` |
| e2e | `EMILIA_SAMPLE_DIR=... pytest tests/integration -m integration` | `EMILIA_SAMPLE_DIR` with `*.tar` |
