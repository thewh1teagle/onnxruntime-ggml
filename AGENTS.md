# Development notes

## Tooling

- Tasks: `chore` (see `chore list`). CI runs the same tasks.
- Rust workspace at the root, crates under `crates/`. Python wrapper under `python/`, tests under `tests/` as standalone `uv` scripts.
- Python: `uv` only (`uv run tests/test_ops.py`).

## Layout

- `crates/ggml-sys`: bindgen over `libs/include`, links `libs/lib` (prebuilt ggml; `chore fetch-libs`).
- `crates/ort-ep-sys`: bindgen over `libs/ort/include`; nothing linked, onnxruntime hands us its API table.
- `crates/onnxruntime-ggml`: the provider. `ep/` (C tables), `ort/` (API wrappers), `ir.rs`, `host/` (host interpreter), `exec/` (compiler + runtime + ggml emitters).
- `docs/ARCHITECTURE.md` explains the placement rules; read it before touching `exec/runtime.rs`.

## Iterating

- `ORT_GGML_LOG=trace uv run tests/test_ops.py <case>` shows per-node placement, uploads, readbacks and shapes. Read the log before changing code.
- Every new op needs: a host implementation (`host/eval_*.rs`), a ggml emitter or an explicit decline (`exec/ops_*.rs`), and a case in `tests/test_ops.py`.
- `chore compare` is the acceptance test for pocket-tts.

## Execution mindset

Parallel moves, instant iteration. Split heavy work by module. Estimate by output size.

## ETA rule

Quote minutes, never days: single tasks 1–10 min, multi-agent work tens of minutes.

```
minutes ≈ (LOC × 40) / (6000 × N_agents) + ~2 min per stage
```

## File size

700 lines max. On hitting it, split by responsibility into halves: find where the file does two jobs and move one out whole.
