# Options

Provider options are strings. Pass them where you select the provider:

```python
ggml.InferenceSession("model.onnx", {"device": "cpu", "threads": "4"})
ggml.session_options({"device": "gpu"})
so.add_provider_for_devices(ggml.devices(), {"device": "gpu"})   # plain onnxruntime API
```

onnxruntime stores them as session config entries `ep.ggml.<key>`, which is also where the provider reads them.

| key | values | default | meaning |
|---|---|---|---|
| `device` | `auto`, `gpu`, `cpu` | `auto` | `gpu` fails if no Metal/Vulkan device exists; `auto` falls back to CPU |
| `threads` | int | half the cores, max 8 | ggml CPU backend threads |
| `partial` | `0`, `1` | `0` | claim supported nodes even when some ops are unsupported (copies at every boundary) |
| `dump` | `0`, `1` | `0` | log every intermediate value at trace level |

## Environment

Lower precedence than session options.

| variable | meaning |
|---|---|
| `ORT_GGML_DEVICE`, `ORT_GGML_THREADS`, `ORT_GGML_PARTIAL`, `ORT_GGML_DUMP` | the options above |
| `ORT_GGML_LOG` | tracing filter: `info` (default), `debug`, `trace`, or a full `EnvFilter` directive |
| `ORT_GGML_CPU_VARIANT` | x86_64 only: `avx2` or `baseline` to override CPU feature detection |
| `ONNXRUNTIME_GGML_LIBRARY` | path of the provider library, instead of the one bundled in the wheel |

## Reading the log

```
INFO  CreateEpFactories registration_name=ggml version=0.1.0 ort_api_version=29
INFO  ggml device index=0 name=Metal description="Apple M4 Pro" ...
INFO  backend ready primary=Metal n_backends=3 gpu=true threads=6
INFO  nodes claimed for fusion claimed=2288 total=2288
INFO  compiled nodes_in=2288 nodes_out=1417 folded=871 gelu_fused=8 weights=213 weight_bytes=399.1 MiB ms=812.4
DEBUG flush reason="graph outputs" ggml_nodes=1902 readbacks=7 readback=1.2 MiB alloc_ms=0.41 compute_ms=9.83
DEBUG run host_ops=402 device_ops=1015 fallbacks=3 flushes=1 uploads=31 upload=2.9 MiB ...
```

- `claimed < total`: some op is unsupported; the warning above names it.
- `fallbacks`: device emitters that declined a shape; `debug` says which and why.
- `flushes > 1` per run: a host op needed a device value mid-graph; `trace` shows the node.
