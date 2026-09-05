# onnxruntime-ggml

An onnxruntime execution provider that runs ONNX models on ggml's Metal, Vulkan and CPU kernels.

```console
uv pip install onnxruntime-ggml
```

The wheel includes the native provider and installs a compatible onnxruntime.
Supports macOS 12+ (Apple silicon/Intel), Linux x86_64 (glibc 2.35+) and aarch64
(glibc 2.39+), and Windows x64. No compiler or separate ggml installation is needed.

```python
import onnxruntime_ggml as ggml

session = ggml.InferenceSession("model.onnx")            # ggml first, CPU provider as fallback
session = ggml.InferenceSession("model.onnx", {"device": "cpu", "threads": 4})
```

Or keep your own session options:

```python
import onnxruntime as ort, onnxruntime_ggml as ggml

so = ggml.session_options({"device": "gpu"})
session = ort.InferenceSession("model.onnx", so)
```

Set `ORT_GGML_LOG=debug` to see what the provider does with a graph. Full docs: https://github.com/thewh1teagle/onnxruntime-ggml
