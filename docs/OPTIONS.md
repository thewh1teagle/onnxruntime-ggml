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
| `attention` | `auto`, `matmul`, `flash`, `flash-f32` | `auto` | exported attention is fused into one node; `auto` runs flash attention (f32 K/V) when the query has 32+ positions and a three-kernel matmul path otherwise |
| `gpu_min_weight_mb` | int | `256` | programs with fewer resident weight bytes run on ggml's CPU backend even when a GPU exists: small models are launch-bound and measured faster there (pocket-tts 194 MiB: CPU; whisper decoders 430 MiB: Metal) |
| `conv_transpose_matmul` | `0`, `1` | `1` | ConvTranspose as one matmul plus strided accumulates instead of ggml's kernel |
| `profile` | `0`, `1` | `0` | time every ggml kernel through the scheduler callback and log a per-op summary (serialises the graph) |
| `accel` | `0`, `1` | `0` | add ggml's ACCEL backends (BLAS on macOS) to the scheduler |
| `weights` | `f32`, `f16`, `q8_0` | `f16` | storage type of the resident 2-D matmul weights; `q8_0` halves the bytes again for decode-bound models |
| `sticky` | `0`, `1` | `1` | keep large unchanged float graph inputs resident on the device between runs |

### `weights`

On the CPU backend ggml's f16 matmul kernels convert the activations to f16 as well, so results differ from an f32 run at the 1e-2 level on large-magnitude tensors such as KV caches; Metal keeps activations in f32. `weights=f32` gives bit-for-bit agreement with the fp32 reference at about 30% more time per step on the CPU backend.

`f16` stores every 2-D weight matrix `ggml_mul_mat` reads as src0 (the
pre-transposed MatMul weights and Gemm B operands with `transB=1`) as
`GGML_TYPE_F16` on the device, halving the resident bytes. Biases, norm scales
and anything a name is also used for elsewhere stay `f32`.

`ggml_mul_mat` takes an f16 src0 against an f32 src1 on both Metal and CPU, so
nothing else changes. On the whisper-large-v3-turbo encoder (Metal, M4 Pro) this
takes the resident weights from 2.37 GiB to 1.20 GiB and one window from 688 ms
to 623 ms. Set `f32` if the extra rounding matters: the weights are rounded
once, so the error is that of an f16 weight matrix, not of f16 accumulation.

### `sticky`

A decode step of an encoder-decoder model hands the provider the encoder
cross-attention KV caches as ordinary graph inputs: for
whisper-large-v3-turbo, 61 MiB of f32 that is bit-identical at every step.
With `sticky=1` each float graph input of at least 256 KiB gets a device
buffer owned by the compiled program, and a run re-uses it when the input
looks unchanged.

"Looks unchanged" is a *fingerprint*, not a comparison: the byte length, the
shape, and a hash of the first and last 64 elements plus 512 samples spread
across the tensor. A change that touches none of the sampled elements would be
missed; hashing every byte would cost what the upload costs. An input whose
fingerprint ever changes is marked volatile and goes back to per-run uploads,
so a growing self-attention cache pays one buffer allocation, once. Set
`sticky=0` if a model needs the guarantee.

`ORT_GGML_LOG=debug` prints `sticky_hits`, `sticky_misses` and `sticky_saved`
per run.

## Environment

Lower precedence than session options.

| variable | meaning |
|---|---|
| `ORT_GGML_DEVICE`, `ORT_GGML_THREADS`, `ORT_GGML_PARTIAL`, `ORT_GGML_DUMP`, `ORT_GGML_ACCEL`, `ORT_GGML_WEIGHTS`, `ORT_GGML_STICKY` | the options above |
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
- `sticky_hits` low on a decode loop: the caller is rebuilding the constant
  cross-attention caches (a copy or a cast) between steps.
- `flushes > 1` per run: a host op needed a device value mid-graph; `trace` shows the node.
