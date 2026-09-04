<h1 align="center">onnxruntime-ggml</h1>

<p align="center">
  <strong>Fast GPU inference for ONNX speech models on macOS, Linux and Windows, backed by ggml.</strong>
</p>

<p align="center">
  <a href="https://github.com/thewh1teagle/onnxruntime-ggml/releases">Releases</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/BUILDING.md">Building</a> ·
  <a href="docs/OPTIONS.md">Options</a>
</p>

---

onnxruntime-ggml is a drop-in [onnxruntime](https://onnxruntime.ai) execution provider. It takes the ONNX graph onnxruntime already loaded and runs it on [ggml](https://github.com/ggml-org/ggml)'s Metal, Vulkan and CPU kernels: the same kernels behind llama.cpp and whisper.cpp. Your models, your onnxruntime code, one extra line.

```python
import onnxruntime_ggml as ggml

session = ggml.InferenceSession("pocket-tts-english.onnx")
```

## Why

- **One provider, three platforms.** Metal on macOS, Vulkan on Linux and Windows, CPU everywhere. No CUDA, no per-vendor builds.
- **Built for speech.** Autoregressive TTS and ASR graphs are thousands of small ops; shape math stays on the host, tensor math goes to the GPU, and data crosses only where it must.
- **Nothing to compile.** The wheel bundles the provider; onnxruntime comes from PyPI; ggml comes prebuilt.
- **Honest.** A graph is claimed whole or not at all, and every decision is traceable with `ORT_GGML_LOG=debug`.

## Install

```console
pip install onnxruntime-ggml
```

Wheels ship for macOS arm64 and x86_64, Linux x86_64 and aarch64, and Windows x64, against onnxruntime 1.29 and later.

## Use

```python
import onnxruntime as ort, onnxruntime_ggml as ggml

so = ggml.session_options({"device": "gpu", "threads": 4})   # or ggml.InferenceSession(path, {...})
session = ort.InferenceSession("model.onnx", so)
```

`python -m onnxruntime_ggml` prints what got registered. See [docs/OPTIONS.md](docs/OPTIONS.md) for every option and environment variable.

## Status

Targets [pocket-tts](https://github.com/thewh1teagle/pocket-tts-onnx) today: 38 ONNX op types, opset 17, fp32. Quantized (`MatMulInteger`) models and further models come next. `chore compare` checks every output against onnxruntime's CPU provider.

## Develop

```console
chore fetch-libs   # prebuilt ggml for this machine
chore test         # Rust unit tests + per-op comparisons against the CPU provider
chore compare      # pocket-tts end to end
chore release 0.1.0
```

`chore list` names every task. Details in [docs/BUILDING.md](docs/BUILDING.md) and [docs/RELEASING.md](docs/RELEASING.md).

## License

MIT. ggml is MIT; onnxruntime is MIT.
