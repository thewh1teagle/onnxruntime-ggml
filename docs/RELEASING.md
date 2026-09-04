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

`chore wheels` skips platforms whose archive is missing, so a partial matrix still yields wheels for what was built.

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
