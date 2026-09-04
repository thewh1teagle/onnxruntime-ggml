# Architecture

onnxruntime-ggml is a plugin execution provider: a shared library onnxruntime loads with `register_execution_provider_library`. onnxruntime keeps everything it is good at (model loading, graph optimisation, the Python API, the CPU provider as fallback) and hands this provider a graph to run.

```
onnxruntime ──GetCapability──▶ claim all nodes, or none
            ──Compile───────▶ OrtGraph ─▶ ir::Graph ─▶ fold ─▶ rewrite ─▶ upload weights ─▶ Program
            ──Compute───────▶ inputs ─▶ Run (host ops + one ggml graph per flush) ─▶ outputs
```

## Crates

| crate | role |
|---|---|
| `ggml-sys` | bindgen over the pinned ggml headers, links the prebuilt static libraries |
| `ort-ep-sys` | bindgen over the pinned onnxruntime headers; links nothing (the API arrives as a function table) |
| `onnxruntime-ggml` | the provider |

Inside `onnxruntime-ggml`:

| module | role |
|---|---|
| `ep/factory.rs` | `OrtEpFactory`: registers on the CPU hardware device, creates providers |
| `ep/provider.rs` | `OrtEp`: claims nodes, compiles subgraphs, owns the ggml backend |
| `ep/compute.rs` | `OrtNodeComputeInfo`: reads inputs, runs the program, writes outputs |
| `ort/` | safe wrappers: API table, `OrtGraph` import, kernel-context I/O |
| `ir.rs` | the graph the compiler works on |
| `host/` | host tensors and an interpreter for every supported ONNX op |
| `exec/program.rs` | compile: constant folding, GELU fusion, weight pre-transpose, weight upload |
| `exec/runtime.rs` | run: placement, uploads, flushes, readbacks |
| `exec/sticky.rs` | graph inputs kept resident on the device between runs |
| `exec/fold.rs` | logical shapes of any rank on 4-dimensional ggml tensors |
| `exec/fusion.rs` | pattern fusions: GELU, decomposed LayerNorm |
| `exec/ops_*.rs` | ONNX → ggml emitters |
| `exec/backend.rs` | ggml backends and scheduler |

## Memory model

The provider tells onnxruntime it lives on the CPU device. Inputs and outputs are host memory; the provider copies them into ggml buffers itself. This keeps the onnxruntime side trivial (no allocators, no data transfer, no streams) and makes CPU-provider fallback free of device copies. Weights are uploaded once at compile time and stay resident on the primary backend (Metal on macOS).

Large float *graph inputs* can also stay resident between runs: `exec/sticky.rs`
keeps one device buffer per input and re-uses it while a fingerprint of the host
bytes is unchanged, which is what makes a decoder's constant cross-attention KV
cache free after the first step (`sticky` option, `docs/OPTIONS.md`).

## Placement

Every runtime value is one of:

- **Host**: shapes, indices, masks, scalars. Lives on the host, never uploaded unless a device op consumes it as data.
- **Staged**: float data that wants to be on the device but is currently on the host (graph inputs, constants not yet uploaded, readbacks).
- **Device**: a tensor inside the current ggml graph.

For each node, in order:

1. **Forced host** if the op is shape-only (`Shape`, `Range`, comparisons, int `Cast`, ...), there is no ggml emitter, or an input has rank > 4 and the op's emitter is not one of the rank-agnostic ones (`RANK_ANY_OPS`: the structural ops, which fold the shape themselves — see below).
2. **Host** if every input is `Host`.
3. **Device** otherwise. Host inputs used as data are uploaded; inputs used as parameters (a Reshape's shape, Slice bounds) must be host, and a flush happens if one is on the device. If the emitter declines a shape it cannot express, the node runs on the host.

A **flush** computes the ggml graph built so far, reads back every live device value as `Staged`, and starts a new graph. Flushes happen only when a host op needs a device value, and at the end for the outputs. `ORT_GGML_LOG=debug` prints one line per flush with node counts, bytes moved and timings.

## Shape convention

ONNX shape `[d0, .., dn-1]` is ggml `ne = [dn-1, .., d0]` padded with 1s. The ONNX rank is carried alongside every device tensor because ggml cannot tell `[3]` from `[1, 3]`. `exec/ggml.rs` holds every conversion; emitters do not touch `ne` directly.

A device tensor carries a *logical* ONNX shape of any rank up to 8. ggml has
only four dimensions, so above rank 4 the ggml `ne` is a **folding** of the
logical shape: size-1 dims are dropped and adjacent dims merged (`exec/fold.rs`).
The invariant is that the ggml tensor holds the elements in the row-major order
of the logical shape, so Reshape/Unsqueeze/Squeeze are pure metadata, and
Gather-with-a-scalar-index, Slice, Split, Concat and Transpose re-fold to
`[outer, axis, inner]` and become views. A shape that does not fold is declined
and the node runs on the host, as before.

`ggml_mul_mat(a, b)` computes `b · aᵀ`, so weights must be `[N, K]` row-major. Gemm with `transB=1` already is; MatMul weights are transposed once at compile time.

## What pocket-tts needs

38 op types, opset 17, no control flow. 14 are pure shape math and fold or run on the host; 19 map directly to ggml; Conv/ConvTranspose go through im2col and `ggml_conv_transpose_1d`; the GELU subgraph is fused into `ggml_gelu_erf`. The 6-D KV-cache tensors stay on the device throughout, under a folded shape.

## Not yet

- Attention fusion (`ggml_flash_attn_ext`), RoPE tables.
- Zero-copy graph inputs: onnxruntime's input buffers are copied into
  `HostTensor`s before a run, which is ~6 ms per whisper decode step.
- Quantized models (`DynamicQuantizeLinear` / `MatMulInteger` → ggml q8 matmul).
- EPContext: caching the compiled program in the model file.
