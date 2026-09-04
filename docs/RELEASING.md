# Releasing

One command, from a clean tree on `main`:

```console
chore release 0.1.0
```

It sets the version in `Cargo.toml`, `python/pyproject.toml` and the Python package, commits, tags `v0.1.0`, pushes, and creates the GitHub release. The `Release` workflow then:

1. builds the provider on macOS arm64, macOS x86_64, Linux x86_64, Linux aarch64 and Windows x64, and attaches `onnxruntime-ggml-<version>-<platform>.tar.gz` to the release;
2. builds one wheel per platform from those archives (`chore wheels <version>`) and attaches them;
3. publishes the wheels to PyPI when the `PYPI_API_TOKEN` secret is set.

## By hand

```console
chore package-lib 0.1.0             # this machine's archive into dist/
chore wheels 0.1.0                  # wheels for every platform whose archive is on the release
chore publish 0.1.0                 # upload wheels to the release and to PyPI
```

`chore wheels` skips platforms whose archive is missing, so a partial matrix still yields wheels for what was built. It looks in `dist/` first and only then tries the release, so an archive left there by `chore package-lib <version>` is enough to build that platform's wheel with no release at all.

To build a single wheel from one archive:

```console
chore wheel-for 0.1.0 darwin-arm64 dist/onnxruntime-ggml-0.1.0-darwin-arm64.tar.gz dist/wheels
```

The library is staged into `python/onnxruntime_ggml/lib/` for the build and removed again afterwards, leaving the `.gitkeep`. That directory is in `.gitignore`; hatchling still packs it because `artifacts` in `python/pyproject.toml` force-includes VCS-ignored paths. Hatchling builds `py3-none-any`, and `uvx --from wheel wheel tags` retags it for the platform.

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
| darwin-arm64 | `macosx_11_0_arm64` |
| darwin-x86_64 | `macosx_10_15_x86_64` |
| linux-x86_64 | `manylinux_2_28_x86_64` |
| linux-aarch64 | `manylinux_2_28_aarch64` |
| windows-amd64 | `win_amd64` |

The wheel is pure Python plus one native library, tagged `py3-none-<platform>`. It pins `onnxruntime>=1.29,<1.31`: the plugin API version is checked at load and a mismatch fails early with a clear message.

## Models

`chore model` fetches `pocket-tts-english-fp32.onnx` from the `models-v1` release of this repository. To publish a new model set: `gh release create models-v2 --prerelease` and upload, then update `models_tag` in the chorefile.
