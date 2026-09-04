# Building

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) stable
- [chore](https://github.com/getchore/chore)
- [uv](https://docs.astral.sh/uv/) for the Python tests
- clang (bindgen needs libclang; Xcode command line tools on macOS, `libclang-dev` on Debian/Ubuntu)

Neither onnxruntime nor ggml is compiled to build the provider. `libs/` holds the pinned headers; `chore fetch-libs` downloads the prebuilt ggml static libraries for this machine from this repository's own library release. Building that bundle is a separate job (`chore build-libs`, and the Libraries workflow) and needs cmake and a C/C++ toolchain.

## Quick start

```console
chore fetch-libs
chore build            # target/release/libonnxruntime_ggml.{dylib,so,dll}
chore test             # cargo test + tests/test_ops.py against the CPU provider
chore model            # models/pocket-tts-english-fp32.onnx
chore compare          # every pocket-tts output vs the CPU provider, then timing
chore lint             # cargo fmt --check + clippy, the way CI runs them
```

The Python tests find the freshly built library automatically (`tests/_common.py` sets `ONNXRUNTIME_GGML_LIBRARY` to `target/release/...`). The exception is `tests/test_wheel.py`, which checks an *installed* wheel and so clears that variable; see [Verifying a wheel](RELEASING.md#verifying-a-wheel).

## Native inputs

| file | meaning |
|---|---|
| `libs/ggml-version` | ggml tag the headers and libraries are built from |
| `libs/revision` | bundle revision; bumped by hand with any other change under `libs/` |
| `libs/patches/*.patch` | fixes applied to the ggml checkout before it is built; each file's header says what it is for |
| `libs/ort-version` | onnxruntime tag the C headers come from |
| `libs/include/` | ggml headers (ignored; `chore headers` fetches them at the pinned tag) |
| `libs/ort/include/` | onnxruntime headers (ignored; same) |
| `libs/lib/` | prebuilt ggml static libraries (ignored; `chore fetch-libs`) |

The libraries are ours. `chore build-libs` clones ggml at `libs/ggml-version`, applies `libs/patches/`, builds it and writes `target/ggml-libs-<platform>.tar.gz` with `lib/` and `include/` at the archive root; `chore upload-libs` does that and attaches the archive to the release named by `chore libs-tag`, `libraries-ggml-<ggml-version>-r<revision>` on this repository. `chore fetch-libs` computes the same tag and downloads from it. The `.github/workflows/libs.yml` matrix runs `chore upload-libs` on every platform; the first job to finish creates the release, the rest upload into it.

Everything is compiled with `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`. The provider is a cdylib, and static libraries built without `-fPIC` cannot be linked into a shared object on Linux x86_64 (`relocation R_X86_64_PC32 ... against '__libc_single_threaded'`). macOS gets Metal and Accelerate, Linux and Windows get Vulkan with the loader opened at runtime. x86_64 bundles carry the CPU backend twice — a Haswell build and an AVX baseline build, symbols suffixed `_hsw` and `_x64` — and `crates/ggml-sys/src/cpu_variant.rs` picks one at startup, so the release runner's CPU does not decide what users can run.

To test a bundle before it is published, point `fetch-libs` at a local archive:

```console
chore build-libs
ORT_GGML_LIBS_ARCHIVE=target/ggml-libs-darwin-arm64.tar.gz chore fetch-libs
chore test
```

Bumping ggml: change `libs/ggml-version`, reset `libs/revision` to `1`, run `chore fetch-headers`, re-check the patches still apply (`chore build-libs` fails if one does not; drop the ones ggml has taken), commit, and let the Libraries workflow publish the new bundle. Changing anything else under `libs/` — a patch, the recipe — means bumping `libs/revision` instead, so the new bundle cannot replace the one an older release still links against; `upload-libs` refuses a tag that already exists for a different `libs/` tree. Bumping onnxruntime: change `libs/ort-version`, run `chore fetch-ort-headers`, check `ORT_API_VERSION` still matches what the wheel the tests use provides.

## Tracing

Everything logs through `tracing` to stderr, filtered by `ORT_GGML_LOG`:

```console
ORT_GGML_LOG=debug uv run tests/test_ops.py gather
ORT_GGML_LOG=trace,onnxruntime_ggml::exec::runtime=trace uv run tests/compare_pocket_tts.py
```

`info` is one line per session event, `debug` per run summaries and flushes, `trace` per node placement with shapes and per transfer.

## Windows

Rust's MSVC toolchain and the `windows-amd64-msvc` ggml bundle. The Vulkan loader is opened at runtime by ggml, so no SDK is needed to build or run.
