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

Install the package and its bundled native provider into a virtual environment:

```console
uv pip install onnxruntime-ggml
```

In a uv project, use `uv add onnxruntime-ggml`. The matching wheel and compatible
onnxruntime dependency are selected automatically; no Rust build or separate
ggml download is required.

Wheels support macOS 14+ on Apple silicon, Linux with glibc 2.35+ on
x86_64 or glibc 2.39+ on aarch64, and Windows x64. Linux wheels bundle additional
runtime libraries and are checked with auditwheel. GPU execution needs a working
Metal or Vulkan device; CPU execution is also available. Intel macOS has a native
library release asset for applications providing a compatible ONNX Runtime;
there is no Intel macOS Python wheel.

From a checkout instead (needs Rust and [chore](https://github.com/getchore/chore); nothing else is compiled, ggml comes prebuilt):

```console
git clone https://github.com/thewh1teagle/onnxruntime-ggml
cd onnxruntime-ggml && chore dev          # builds the provider and puts it inside python/
uv pip install ./python                   # from any project
```

## Use

```python
import onnxruntime as ort, onnxruntime_ggml as ggml

so = ggml.session_options({"device": "gpu", "threads": 4})   # or ggml.InferenceSession(path, {...})
session = ort.InferenceSession("model.onnx", so)
```

`python -m onnxruntime_ggml` prints what got registered. See [docs/OPTIONS.md](docs/OPTIONS.md) for every option and environment variable.

## Status

Validated with Pocket-TTS, Whisper and Kokoro. Generic recurrent, control-flow,
sequence, resize and indexing operators extend the provider beyond the original
Pocket-TTS graph; [architecture notes](docs/ARCHITECTURE.md) describe supported
subsets and host placement.

On Apple M4 Pro, FP32 Kokoro inference measured 0.577 s median with ggml versus
1.161 s with ONNX Runtime CPU, using seven threads and seven measured runs after
warmup. All 2,138 optimized nodes were claimed with CPU EP fallback disabled;
some operations still run inside ggml's CPU backend or the provider's host
interpreter. Performance depends on the model and hardware.

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
