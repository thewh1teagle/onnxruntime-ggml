# Building

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) stable
- [chore](https://github.com/getchore/chore)
- [uv](https://docs.astral.sh/uv/) for the Python tests
- clang (bindgen needs libclang; Xcode command line tools on macOS, `libclang-dev` on Debian/Ubuntu)

Neither onnxruntime nor ggml is compiled. `libs/` holds the pinned headers; `chore fetch-libs` downloads the prebuilt ggml static libraries for this machine.

## Quick start

```console
chore fetch-libs
chore build            # target/release/libonnxruntime_ggml.{dylib,so,dll}
chore test             # cargo test + tests/test_ops.py against the CPU provider
chore model            # models/pocket-tts-english-fp32.onnx
chore compare          # every pocket-tts output vs the CPU provider, then timing
```

The Python tests find the freshly built library automatically (`tests/_common.py` sets `ONNXRUNTIME_GGML_LIBRARY` to `target/release/...`).

## Native inputs

| file | meaning |
|---|---|
| `libs/ggml-version` | ggml tag the headers and libraries come from |
| `libs/libs-repo`, `libs/libs-tag` | GitHub release the prebuilt bundle is downloaded from |
| `libs/ort-version` | onnxruntime tag the C headers come from |
| `libs/include/` | ggml headers (checked in; `chore fetch-headers`) |
| `libs/ort/include/` | onnxruntime headers (checked in; `chore fetch-ort-headers`) |
| `libs/lib/` | prebuilt ggml static libraries (ignored; `chore fetch-libs`) |

Bumping ggml: change `libs/ggml-version` and `libs/libs-tag` together, run `chore fetch-headers`, commit. Bumping onnxruntime: change `libs/ort-version`, run `chore fetch-ort-headers`, check `ORT_API_VERSION` still matches what the wheel the tests use provides.

## Tracing

Everything logs through `tracing` to stderr, filtered by `ORT_GGML_LOG`:

```console
ORT_GGML_LOG=debug uv run tests/test_ops.py gather
ORT_GGML_LOG=trace,onnxruntime_ggml::exec::runtime=trace uv run tests/compare_pocket_tts.py
```

`info` is one line per session event, `debug` per run summaries and flushes, `trace` per node placement with shapes and per transfer.

## Windows

Rust's MSVC toolchain and the `windows-amd64-msvc` ggml bundle. The Vulkan loader is opened at runtime by ggml, so no SDK is needed to build or run.
