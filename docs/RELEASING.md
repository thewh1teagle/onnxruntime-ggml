# Releasing

One command, from a clean tree on `main`:

```console
chore release 0.1.0
```

It sets the version in `Cargo.toml`, `python/pyproject.toml` and the Python package, commits, tags `v0.1.0`, pushes, and creates the GitHub release. The `Release` workflow then:

1. builds the provider and wheel on macOS arm64, Linux x86_64, Linux aarch64 and Windows x64; an additional job builds the Intel macOS native library;
2. repairs Linux wheels with auditwheel, installs each wheel into a fresh virtual environment on its target platform, and runs inference with CPU EP fallback disabled;
3. attaches the tested native archives and wheels to the GitHub release;
4. publishes all four wheels to PyPI when every platform passes and the `PYPI_API_TOKEN` secret is set.

## By hand

```console
chore package-lib 0.1.0             # this machine's archive into dist/
chore wheels 0.1.0                  # download all four tested release wheels
chore publish 0.1.0                 # publish the tested release wheels to PyPI
```

`chore wheels` downloads the four tested wheels from the release and fails if the set is incomplete. Linux wheels must be built and repaired on their target architecture; the release workflow handles that automatically.

To build a single wheel from one archive:

```console
chore wheel-for 0.1.0 darwin-arm64 dist/onnxruntime-ggml-0.1.0-darwin-arm64.tar.gz dist/wheels
```

The library is staged into `python/onnxruntime_ggml/lib/` for the build and removed again afterwards, leaving the `.gitkeep`. That directory is in `.gitignore`; hatchling still packs it because `artifacts` in `python/pyproject.toml` force-includes VCS-ignored paths. Hatchling builds `py3-none-any`, and `uvx --from wheel wheel tags` retags it for the platform. Linux additionally runs `auditwheel repair` to bundle non-policy runtime libraries and validate the claimed symbol baseline; install `patchelf` first. Build Linux wheels on the matching architecture.

## Verifying a wheel

```console
uv venv /tmp/ggml-wheel-test
uv pip install --python /tmp/ggml-wheel-test/bin/python dist/wheels/*.whl
/tmp/ggml-wheel-test/bin/python -m onnxruntime_ggml
uv run tests/test_wheel.py /tmp/ggml-wheel-test/bin/python
```

`python -m onnxruntime_ggml` prints the library path and one `device ggml ...` line; the path must be inside the venv's `site-packages`, not `target/release`. `tests/test_wheel.py` builds a two-node graph (Add then Mul) with `onnx.helper` and runs it in that interpreter twice, through `onnxruntime_ggml.InferenceSession` and on the CPU provider, then compares the outputs. It clears `ONNXRUNTIME_GGML_LIBRARY` for the child, so it always exercises the library bundled in the wheel rather than a development build.

## Wheel tags

| platform | wheel tag |
|---|---|
| darwin-arm64 | `macosx_14_0_arm64` |
| linux-x86_64 | `manylinux_2_35_x86_64` |
| linux-aarch64 | `manylinux_2_39_aarch64` |
| windows-amd64 | `win_amd64` |

Intel macOS has a native library asset but no Python wheel because ONNX Runtime 1.29 does not distribute an Intel macOS wheel. Python installation on macOS requires macOS 14+ on Apple silicon.

The wheel is pure Python plus one native library, tagged `py3-none-<platform>`. It requires Python 3.11+ and pins `onnxruntime>=1.29,<1.31`: the plugin API version is checked at load and a mismatch fails early with a clear message.

## Models

`chore model` fetches `pocket-tts-english-fp32.onnx` from the `models-v2` release of this repository. The v2 model preserves the FP32 synthesis graph and corrects the bundled voice encoder and conditioning. To publish a new model set: `gh release create models-v3 --prerelease` and upload, then update `models_tag` in the chorefile.
