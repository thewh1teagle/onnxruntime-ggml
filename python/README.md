# onnxruntime-ggml

An onnxruntime execution provider that runs ONNX models on ggml's Metal, Vulkan and CPU kernels.

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
