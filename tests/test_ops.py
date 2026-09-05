# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "onnx>=1.16", "onnxruntime>=1.29,<1.31"]
# ///
"""Per-op tests: tiny ONNX graphs run on the CPU provider and on ggml, outputs compared.

    uv run tests/test_ops.py            # all
    uv run tests/test_ops.py gather     # cases whose name contains "gather"
    ORT_GGML_DEVICE=cpu uv run tests/test_ops.py

Each case builds a graph with onnx.helper, so it doubles as documentation of
the exact op shapes pocket-tts uses (opset 17, 6-D KV caches, dynamic axes).
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _common as C  # noqa: E402

OPSET = 17
rng = np.random.default_rng(0)


def f32(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def model(nodes, inputs, outputs, inits=(), name="t"):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=list(inits))
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m.SerializeToString()


def tin(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


# Each case returns (model_bytes, feeds, atol, rtol)


@case
def add_broadcast():
    nodes = [helper.make_node("Add", ["a", "b"], ["y"])]
    return model(nodes, [tin("a", [2, 1, 4]), tin("b", [3, 1])], [tin("y", [2, 3, 4])]), {"a": f32(2, 1, 4), "b": f32(3, 1)}, 1e-6, 1e-5


@case
def sub_div_scalar_first():
    # const / x and const - x: the small operand comes first, ggml needs a repeat
    nodes = [helper.make_node("Div", ["c", "x"], ["d"]), helper.make_node("Sub", ["c", "d"], ["y"])]
    return model(nodes, [tin("x", [3, 5])], [tin("y", [3, 5])], [const("c", np.float32(2.0))]), {"x": f32(3, 5) + 3}, 1e-5, 1e-5


@case
def unary_chain():
    nodes = [
        helper.make_node("Sigmoid", ["x"], ["s"]),
        helper.make_node("Elu", ["s"], ["e"], alpha=1.0),
        helper.make_node("Sqrt", ["e"], ["q"]),
        helper.make_node("Exp", ["q"], ["ex"]),
        helper.make_node("Reciprocal", ["ex"], ["r"]),
        helper.make_node("Mul", ["r", "x"], ["y"]),
    ]
    return model(nodes, [tin("x", [4, 8])], [tin("y", [4, 8])]), {"x": f32(4, 8)}, 1e-5, 1e-4


@case
def erf_standalone():
    nodes = [helper.make_node("Erf", ["x"], ["y"])]
    return model(nodes, [tin("x", [64])], [tin("y", [64])]), {"x": f32(64) * 2}, 2e-4, 0


@case
def gelu_pattern():
    nodes = [
        helper.make_node("Div", ["x", "sqrt2"], ["d"]),
        helper.make_node("Erf", ["d"], ["e"]),
        helper.make_node("Add", ["e", "one"], ["a"]),
        helper.make_node("Mul", ["x", "a"], ["m"]),
        helper.make_node("Mul", ["m", "half"], ["y"]),
    ]
    inits = [const("sqrt2", np.float32(1.4142135)), const("one", np.float32(1.0)), const("half", np.float32(0.5))]
    return model(nodes, [tin("x", [2, 16])], [tin("y", [2, 16])], inits), {"x": f32(2, 16) * 3}, 1e-5, 1e-4


@case
def gemm_transb_bias():
    w = f32(6, 4)
    b = f32(6)
    nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["y"], alpha=1.0, beta=1.0, transB=1)]
    # tolerance is f16 weight rounding: `weights=f16` is the default and w is standard normal
    return model(nodes, [tin("x", [3, 4])], [tin("y", [3, 6])], [const("w", w), const("b", b)]), {"x": f32(3, 4)}, 5e-3, 1e-2


@case
def matmul_const_weight():
    w = f32(4, 6)
    nodes = [helper.make_node("MatMul", ["x", "w"], ["y"])]
    # tolerance is f16 weight rounding (see gemm_transb_bias)
    return model(nodes, [tin("x", [1, 3, 4])], [tin("y", [1, 3, 6])], [const("w", w)]), {"x": f32(1, 3, 4)}, 5e-3, 1e-2


@case
def matmul_dynamic_attention():
    # q [1,H,T,D] x k^T [1,H,D,S] -> softmax -> x v
    nodes = [
        helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["q", "kt"], ["s"]),
        helper.make_node("Softmax", ["s"], ["p"], axis=-1),
        helper.make_node("MatMul", ["p", "v"], ["y"]),
    ]
    feeds = {"q": f32(1, 2, 3, 8), "k": f32(1, 2, 5, 8), "v": f32(1, 2, 5, 8)}
    return model(nodes, [tin("q", [1, 2, 3, 8]), tin("k", [1, 2, 5, 8]), tin("v", [1, 2, 5, 8])], [tin("y", [1, 2, 3, 8])]), feeds, 1e-5, 1e-4


@case
def layernorm():
    s = f32(16) + 1
    b = f32(16)
    nodes = [helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"], axis=-1, epsilon=1e-5)]
    return model(nodes, [tin("x", [2, 3, 16])], [tin("y", [2, 3, 16])], [const("s", s), const("b", b)]), {"x": f32(2, 3, 16)}, 1e-4, 1e-3


@case
def reduce_mean_keepdims():
    nodes = [helper.make_node("ReduceMean", ["x"], ["y"], axes=[-1], keepdims=1)]
    return model(nodes, [tin("x", [2, 5, 7])], [tin("y", [2, 5, 1])]), {"x": f32(2, 5, 7)}, 1e-5, 1e-4


@case
def transpose_reshape():
    nodes = [
        helper.make_node("Transpose", ["x"], ["t"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["t", "shape"], ["y"]),
    ]
    inits = [const("shape", np.array([0, 3, -1], dtype=np.int64))]
    return model(nodes, [tin("x", [2, 4, 3, 5])], [tin("y", [2, 3, 20])], inits), {"x": f32(2, 4, 3, 5)}, 0, 0


@case
def slice_concat_split():
    nodes = [
        helper.make_node("Slice", ["x", "st", "en", "ax"], ["a"]),
        helper.make_node("Concat", ["a", "x"], ["c"], axis=1),
        helper.make_node("Split", ["c", "sizes"], ["p", "q"], axis=1),
        helper.make_node("Mul", ["p", "q"], ["y"]),
    ]
    inits = [
        const("st", np.array([1], dtype=np.int64)),
        const("en", np.array([3], dtype=np.int64)),
        const("ax", np.array([1], dtype=np.int64)),
        const("sizes", np.array([3, 3], dtype=np.int64)),
    ]
    return model(nodes, [tin("x", [2, 4, 3])], [tin("y", [2, 3, 3])], inits), {"x": f32(2, 4, 3)}, 1e-6, 1e-5


@case
def gather_embedding():
    table = f32(50, 8)
    nodes = [helper.make_node("Gather", ["table", "ids"], ["y"], axis=0)]
    ids = np.array([[3, 7, 0, 49]], dtype=np.int64)
    return model(nodes, [tin("ids", [1, 4], TensorProto.INT64)], [tin("y", [1, 4, 8])], [const("table", table)]), {"ids": ids}, 0, 0


@case
def gather_single_token_embedding():
    # whisper's decoder_with_past: one token, indices of shape [1, 1]. The
    # Gather output must keep rank 3 ([1, 1, d]) even though it holds a single
    # row, or the following Reshape's 0 entries copy the wrong dims.
    table = f32(50, 8)
    nodes = [
        helper.make_node("Gather", ["table", "ids"], ["e"], axis=0),
        helper.make_node("Reshape", ["e", "sh"], ["y"]),
    ]
    inits = [const("table", table), const("sh", np.array([0, 0, 2, 4], dtype=np.int64))]
    ids = np.array([[7]], dtype=np.int64)
    return (
        model(nodes, [tin("ids", [1, 1], TensorProto.INT64)], [tin("y", [1, 1, 2, 4])], inits),
        {"ids": ids},
        0,
        0,
    )


@case
def gather_select_scalar():
    nodes = [helper.make_node("Gather", ["x", "i"], ["y"], axis=1)]
    return model(nodes, [tin("x", [2, 5, 3])], [tin("y", [2, 3])], [const("i", np.int64(2))]), {"x": f32(2, 5, 3)}, 0, 0


@case
def unsqueeze_squeeze_expand():
    nodes = [
        helper.make_node("Unsqueeze", ["x", "ax"], ["u"]),
        helper.make_node("Expand", ["u", "sh"], ["e"]),
        helper.make_node("Squeeze", ["e", "ax2"], ["y"]),
    ]
    inits = [
        const("ax", np.array([0, 2], dtype=np.int64)),
        const("sh", np.array([1, 3, 4, 5], dtype=np.int64)),
        const("ax2", np.array([0], dtype=np.int64)),
    ]
    return model(nodes, [tin("x", [3, 5])], [tin("y", [3, 4, 5])], inits), {"x": f32(3, 5)}, 0, 0


@case
def where_neg_inf_mask():
    # attention mask pattern: Where(mask, scores, -inf) then softmax
    mask = np.array([[True, True, False, False], [True, True, True, False]])
    nodes = [
        helper.make_node("Where", ["m", "x", "ninf"], ["w"]),
        helper.make_node("Softmax", ["w"], ["y"], axis=-1),
    ]
    inits = [const("m", mask), const("ninf", np.array(-np.inf, dtype=np.float32))]
    return model(nodes, [tin("x", [2, 4])], [tin("y", [2, 4])], inits), {"x": f32(2, 4)}, 1e-6, 1e-5


@case
def where_general():
    nodes = [helper.make_node("Where", ["m", "x", "z"], ["y"])]
    m = np.array([True, False, True], dtype=bool)
    return model(nodes, [tin("x", [2, 3]), tin("z", [2, 3])], [tin("y", [2, 3])], [const("m", m)]), {"x": f32(2, 3), "z": f32(2, 3)}, 0, 0


@case
def conv1d_pad():
    w = f32(6, 4, 3)
    b = f32(6)
    nodes = [helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3], pads=[1, 1], strides=[1], dilations=[1], group=1)]
    return model(nodes, [tin("x", [1, 4, 10])], [tin("y", [1, 6, 10])], [const("w", w), const("b", b)]), {"x": f32(1, 4, 10)}, 1e-4, 1e-3


@case
def conv1d_k1():
    w = f32(8, 4, 1)
    nodes = [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[1], pads=[0, 0], strides=[1], dilations=[1], group=1)]
    return model(nodes, [tin("x", [1, 4, 7])], [tin("y", [1, 8, 7])], [const("w", w)]), {"x": f32(1, 4, 7)}, 1e-4, 1e-3


@case
def conv_transpose1d():
    w = f32(4, 3, 8)
    b = f32(3)
    nodes = [helper.make_node("ConvTranspose", ["x", "w", "b"], ["y"], kernel_shape=[8], strides=[4], pads=[0, 0], dilations=[1], group=1)]
    return model(nodes, [tin("x", [1, 4, 5])], [tin("y", [1, 3, 24])], [const("w", w), const("b", b)]), {"x": f32(1, 4, 5)}, 1e-4, 1e-3


@case
def conv_transpose1d_depthwise():
    w = f32(6, 1, 32)
    nodes = [helper.make_node("ConvTranspose", ["x", "w"], ["y"], kernel_shape=[32], strides=[16], pads=[0, 0], dilations=[1], group=6)]
    return model(nodes, [tin("x", [1, 6, 3])], [tin("y", [1, 6, 64])], [const("w", w)]), {"x": f32(1, 6, 3)}, 1e-4, 1e-3


# --- pocket-tts's decoder convolutions, with a symbolic length so the provider
# --- sees a dynamic shape. Weight shapes are the model's own.


# Metal runs the im2col matmul through its f16 simdgroup path, so a k=7 x 512
# channel dot product (K = 3584) lands ~1e-3 off the CPU provider's f32 sum.
# The CPU backend of the same code is accurate to ~1e-5.
CONV_ATOL, CONV_RTOL = 1e-2, 1e-2


def conv_case(w_shape, kernel, stride, group, cin, lin, out_ch, ltag="L"):
    """One Conv over [1, cin, L] with L symbolic; returns (model, feeds, atol, rtol)."""
    w = f32(*w_shape) * 0.05
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[kernel], pads=[0, 0], strides=[stride], dilations=[1], group=group)
    ]
    inits = [const("w", w), const("b", f32(out_ch))]
    m = model(nodes, [tin("x", [1, cin, ltag])], [tin("y", [1, out_ch, ltag + "o"])], inits)
    return m, {"x": f32(1, cin, lin)}, CONV_ATOL, CONV_RTOL


def conv_transpose_case(w_shape, kernel, stride, group, cin, lin, out_ch):
    w = f32(*w_shape) * 0.05
    nodes = [
        helper.make_node(
            "ConvTranspose", ["x", "w", "b"], ["y"], kernel_shape=[kernel], strides=[stride], pads=[0, 0], dilations=[1], group=group
        )
    ]
    inits = [const("w", w), const("b", f32(out_ch))]
    m = model(nodes, [tin("x", [1, cin, "L"])], [tin("y", [1, out_ch, "Lo"])], inits)
    return m, {"x": f32(1, cin, lin)}, CONV_ATOL, CONV_RTOL


@case
def conv1d_k7_512_dynamic():
    # /conv/Conv: w [512, 512, 7], pads [0, 0], on [1, 512, L]
    return conv_case((512, 512, 7), 7, 1, 1, 512, 9, 512)


@case
def conv1d_k3_256_dynamic():
    # /conv_1/Conv: w [128, 256, 3] on [1, 256, L]
    return conv_case((128, 256, 3), 3, 1, 1, 256, 8, 128)


@case
def conv1d_k1_after_transpose():
    # /output_proj/Conv: fed by a Transpose, and L == 1 at run time, so the input
    # is a permuted view whose ne[0] is 1 -- ggml_is_contiguous says yes while
    # nb[0] is still strided, which the CPU im2col kernel refuses.
    nodes = [
        helper.make_node("Transpose", ["x"], ["xt"], perm=[0, 2, 1]),  # [1, L, 32] -> [1, 32, L]
        helper.make_node("Conv", ["xt", "w"], ["y"], kernel_shape=[1], pads=[0, 0], strides=[1], dilations=[1], group=1),
    ]
    inits = [const("w", f32(512, 32, 1) * 0.05)]
    m = model(nodes, [tin("x", [1, "L", 32])], [tin("y", [1, 512, "L"])], inits)
    return m, {"x": f32(1, 1, 32)}, 1e-4, 1e-3


@case
def conv_transpose1d_k12_s6():
    # /convtr_1/ConvTranspose: w [512, 256, 12], stride 6
    return conv_transpose_case((512, 256, 12), 12, 6, 1, 512, 4, 256)


@case
def conv_transpose1d_k10_s5():
    # /convtr_2/ConvTranspose: w [256, 128, 10], stride 5
    return conv_transpose_case((256, 128, 10), 10, 5, 1, 256, 5, 128)


@case
def conv_transpose1d_k8_s4():
    # /convtr_3/ConvTranspose: w [128, 64, 8], stride 4
    return conv_transpose_case((128, 64, 8), 8, 4, 1, 128, 6, 64)


@case
def conv_transpose1d_depthwise_512():
    # /convtr/ConvTranspose: w [512, 1, 32], group 512, stride 16, on [1, 512, L]
    return conv_transpose_case((512, 1, 32), 32, 16, 512, 512, 3, 512)


@case
def shape_chain_dynamic():
    # positions from a dynamic length: Shape -> Gather -> Range -> Cast -> Cos, then broadcast add
    nodes = [
        helper.make_node("Shape", ["x"], ["sh"]),
        helper.make_node("Gather", ["sh", "one"], ["n"], axis=0),
        helper.make_node("Range", ["zero", "n", "one"], ["pos"]),
        helper.make_node("Cast", ["pos"], ["posf"], to=TensorProto.FLOAT),
        helper.make_node("Cos", ["posf"], ["c"]),
        helper.make_node("Unsqueeze", ["c", "ax"], ["c2"]),
        helper.make_node("Add", ["x", "c2"], ["y"]),
    ]
    inits = [const("one", np.int64(1)), const("zero", np.int64(0)), const("ax", np.array([0, 2], dtype=np.int64))]
    return model(nodes, [tin("x", [1, "seq", 4])], [tin("y", [1, "seq", 4])], inits), {"x": f32(1, 7, 4)}, 1e-5, 1e-5


@case
def mask_from_offsets():
    # GreaterOrEqual over positions -> And -> Where(-inf) exactly like the export's causal mask
    nodes = [
        helper.make_node("Shape", ["x"], ["sh"]),
        helper.make_node("Gather", ["sh", "one"], ["n"], axis=0),
        helper.make_node("Range", ["zero", "n", "one"], ["pos"]),
        helper.make_node("Unsqueeze", ["pos", "a1"], ["row"]),
        helper.make_node("Unsqueeze", ["pos", "a0"], ["col"]),
        helper.make_node("GreaterOrEqual", ["row", "col"], ["ge"]),
        helper.make_node("Less", ["col", "n"], ["lt"]),
        helper.make_node("And", ["ge", "lt"], ["m"]),
        helper.make_node("Where", ["m", "x", "ninf"], ["w"]),
        helper.make_node("Softmax", ["w"], ["y"], axis=-1),
    ]
    inits = [
        const("one", np.int64(1)),
        const("zero", np.int64(0)),
        const("a1", np.array([1], dtype=np.int64)),
        const("a0", np.array([0], dtype=np.int64)),
        const("ninf", np.array(-np.inf, dtype=np.float32)),
    ]
    return model(nodes, [tin("x", ["seq", "seq"])], [tin("y", ["seq", "seq"])], inits), {"x": f32(6, 6)}, 1e-6, 1e-5


@case
def mask_from_constant_of_shape():
    # Same causal mask, but the -inf side of the Where is a whole ConstantOfShape
    # tensor rather than a scalar, and the condition has fewer heads than the
    # scores. Multiplying that fill by a 0/1 mask would give -inf * 0 = NaN.
    nodes = [
        helper.make_node("Shape", ["x"], ["sh"]),
        helper.make_node("ConstantOfShape", ["sh"], ["ninf"], value=helper.make_tensor("v", TensorProto.FLOAT, [1], [-np.inf])),
        helper.make_node("Gather", ["sh", "three"], ["n"], axis=0),
        helper.make_node("Range", ["zero", "n", "one"], ["pos"]),
        helper.make_node("Unsqueeze", ["pos", "a1"], ["row"]),
        helper.make_node("Unsqueeze", ["pos", "a0"], ["col"]),
        helper.make_node("GreaterOrEqual", ["row", "col"], ["ge"]),
        helper.make_node("Unsqueeze", ["ge", "a01"], ["m"]),
        helper.make_node("Where", ["m", "x", "ninf"], ["w"]),
        helper.make_node("Softmax", ["w"], ["y"], axis=-1),
    ]
    inits = [
        const("three", np.int64(3)),
        const("zero", np.int64(0)),
        const("one", np.int64(1)),
        const("a1", np.array([1], dtype=np.int64)),
        const("a0", np.array([0], dtype=np.int64)),
        const("a01", np.array([0, 1], dtype=np.int64)),
    ]
    shape = [1, 4, "seq", "seq"]
    return model(nodes, [tin("x", shape)], [tin("y", shape)], inits), {"x": f32(1, 4, 6, 6)}, 1e-6, 1e-5


@case
def six_dim_kv_cache():
    # [past, L, 2, 1, H, D]: gather a layer, split k/v, concat a new step, write back 6-D
    nodes = [
        helper.make_node("Gather", ["kv", "layer"], ["lkv"], axis=1),  # [past, 2, 1, H, D]
        helper.make_node("Gather", ["lkv", "zero"], ["k"], axis=1),  # [past, 1, H, D]
        helper.make_node("Transpose", ["k"], ["kt"], perm=[1, 2, 0, 3]),  # [1, H, past, D]
        helper.make_node("Concat", ["kt", "new"], ["kcat"], axis=2),  # [1, H, past+1, D]
        helper.make_node("Softmax", ["kcat"], ["s"], axis=-1),
        helper.make_node("Transpose", ["s"], ["st"], perm=[2, 0, 1, 3]),  # [past+1, 1, H, D]
        helper.make_node("Unsqueeze", ["st", "ax"], ["out6"]),  # [past+1, 1, 1, 1, H, D]
        helper.make_node("Concat", ["out6", "out6"], ["y"], axis=1),  # [past+1, 2, 1, 1, H, D]
    ]
    inits = [const("layer", np.int64(1)), const("zero", np.int64(0)), const("ax", np.array([1, 2], dtype=np.int64))]
    feeds = {"kv": f32(5, 3, 2, 1, 2, 4), "new": f32(1, 2, 1, 4)}
    return model(nodes, [tin("kv", ["past", 3, 2, 1, 2, 4]), tin("new", [1, 2, 1, 4])], [tin("y", ["p1", 2, 1, 1, 2, 4])], inits), feeds, 1e-6, 1e-5


@case
def five_dim_slice_gather():
    # [past, 3, 2, H, D]: slice off the first step, select a layer, rebuild 5-D
    nodes = [
        helper.make_node("Slice", ["kv", "s0", "e0", "a0"], ["tail"]),  # [past-1, 3, 2, H, D]
        helper.make_node("Gather", ["tail", "layer"], ["lkv"], axis=1),  # [past-1, 2, H, D]
        helper.make_node("Transpose", ["lkv"], ["t"], perm=[1, 0, 2, 3]),  # [2, past-1, H, D]
        helper.make_node("Transpose", ["t"], ["tb"], perm=[1, 0, 2, 3]),  # [past-1, 2, H, D]
        helper.make_node("Unsqueeze", ["tb", "ax"], ["u"]),  # [past-1, 1, 2, H, D]
        helper.make_node("Concat", ["u", "u"], ["y"], axis=1),  # [past-1, 2, 2, H, D]
    ]
    inits = [
        const("s0", np.array([1], dtype=np.int64)),
        const("e0", np.array([1 << 30], dtype=np.int64)),
        const("a0", np.array([0], dtype=np.int64)),
        const("layer", np.int64(2)),
        const("ax", np.array([1], dtype=np.int64)),
    ]
    feeds = {"kv": f32(5, 3, 2, 4, 8)}
    ins = [tin("kv", ["past", 3, 2, 4, 8])]
    outs = [tin("y", ["p1", 2, 2, 4, 8])]
    return model(nodes, ins, outs, inits), feeds, 1e-6, 1e-5


@case
def six_dim_split_roundtrip():
    # a 6-D cache split along the k/v axis and concatenated back the other way round
    nodes = [
        helper.make_node("Gather", ["kv", "layer"], ["lkv"], axis=1),  # [past, 2, 1, H, D]
        helper.make_node("Unsqueeze", ["lkv", "ax"], ["u6"]),  # [past, 1, 2, 1, H, D]
        helper.make_node("Concat", ["u6", "u6"], ["c6"], axis=1),  # [past, 2, 2, 1, H, D]
        helper.make_node("Split", ["c6", "sizes"], ["p", "q"], axis=2),  # two [past, 2, 1, 1, H, D]
        helper.make_node("Concat", ["q", "p"], ["y"], axis=2),  # [past, 2, 2, 1, H, D]
    ]
    inits = [
        const("layer", np.int64(1)),
        const("ax", np.array([1], dtype=np.int64)),
        const("sizes", np.array([1, 1], dtype=np.int64)),
    ]
    feeds = {"kv": f32(4, 3, 2, 1, 2, 8)}
    ins = [tin("kv", ["past", 3, 2, 1, 2, 8])]
    outs = [tin("y", ["past", 2, 2, 1, 2, 8])]
    return model(nodes, ins, outs, inits), feeds, 1e-6, 1e-5


@case
def int_outputs_pass_through():
    nodes = [
        helper.make_node("Add", ["off", "one"], ["off_out"]),
        helper.make_node("Cast", ["off_out"], ["f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["x", "f"], ["y"]),
    ]
    inits = [const("one", np.int64(1))]
    feeds = {"off": np.array(4, dtype=np.int64), "x": f32(3)}
    return model(nodes, [tin("off", [], TensorProto.INT64), tin("x", [3])], [tin("y", [3]), tin("off_out", [], TensorProto.INT64)], inits), feeds, 1e-6, 1e-6


@case
def sin_cos_scale():
    nodes = [
        helper.make_node("Sin", ["x"], ["s"]),
        helper.make_node("Cos", ["x"], ["c"]),
        helper.make_node("Mul", ["s", "c"], ["m"]),
        helper.make_node("Mul", ["m", "k"], ["y"]),
    ]
    return model(nodes, [tin("x", [3, 9])], [tin("y", [3, 9])], [const("k", np.array([2.0], dtype=np.float32))]), {"x": f32(3, 9)}, 1e-5, 1e-5


@case
def layer_norm_decomposed():
    # what torch exports below opset 17; the compiler fuses it back into one
    # LayerNormalization, so none of Pow/Sqrt/Div ever reaches the host
    nodes = [
        helper.make_node("ReduceMean", ["x"], ["m"], axes=[-1]),
        helper.make_node("Sub", ["x", "m"], ["d"]),
        helper.make_node("Pow", ["d", "two"], ["p"]),
        helper.make_node("ReduceMean", ["p"], ["v"], axes=[-1]),
        helper.make_node("Add", ["v", "eps"], ["ve"]),
        helper.make_node("Sqrt", ["ve"], ["s"]),
        helper.make_node("Div", ["d", "s"], ["n"]),
        helper.make_node("Mul", ["n", "w"], ["nw"]),
        helper.make_node("Add", ["nw", "b"], ["y"]),
    ]
    inits = [
        const("two", np.float32(2.0)),
        const("eps", np.float32(1e-5)),
        const("w", f32(8)),
        const("b", f32(8)),
    ]
    return model(nodes, [tin("x", [2, 5, 8])], [tin("y", [2, 5, 8])], inits), {"x": f32(2, 5, 8)}, 1e-5, 1e-4


@case
def matmul_f16_weights():
    # weights=f16 stores the 2-D matmul weight as GGML_TYPE_F16 on the device;
    # tolerance is that of an f16 weight matrix, the accumulation stays f32.
    # (set through the environment: `weights` is not in ep::options::KEYS yet)
    import os

    os.environ["ORT_GGML_WEIGHTS"] = "f16"
    nodes = [helper.make_node("MatMul", ["x", "w"], ["y"])]
    w = (rng.standard_normal((256, 256)) * 0.05).astype(np.float32)
    m = model(nodes, [tin("x", [4, 256])], [tin("y", [4, 256])], [const("w", w)])
    return m, {"x": f32(4, 256)}, 1e-2, 1e-2


@case
def sticky_input_changed():
    # A float graph input above exec::sticky::MIN_BYTES is kept resident on the
    # device between runs and re-uploaded only when its fingerprint changes.
    # Three runs on one session: same data twice (a hit), then new data.
    nodes = [helper.make_node("MatMul", ["x", "w"], ["y"])]
    w = (rng.standard_normal((64, 8)) * 0.1).astype(np.float32)
    m = model(nodes, [tin("x", [2000, 64])], [tin("y", [2000, 8])], [const("w", w)])
    a = f32(2000, 64)
    feeds = [{"x": a}, {"x": a.copy()}, {"x": f32(2000, 64)}, {"x": a}]
    # the weight goes to the device as f16 by default, hence the tolerance
    return m, feeds, 1e-2, 1e-2


@case
def dynamic_int8_matmul():
    # onnxruntime quantize_dynamic output: DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul(xs*ws)
    k, n = 64, 32
    w = f32(k, n)
    ws = (np.abs(w).max(axis=0) / 127.0).astype(np.float32)
    wq = np.clip(np.round(w / ws), -127, 127).astype(np.int8)
    wzp = np.zeros(n, dtype=np.int8)
    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xzp"]),
        helper.make_node("MatMulInteger", ["xq", "wq", "xzp", "wzp"], ["yi"]),
        helper.make_node("Cast", ["yi"], ["yf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["xs", "ws"], ["scales"]),
        helper.make_node("Mul", ["yf", "scales"], ["y"]),
    ]
    inits = [const("wq", wq), const("wzp", wzp), const("ws", ws)]
    # the reference quantises activations to 8 bits; ggml keeps them f32, so ~1% relative differs
    return model(nodes, [tin("x", [1, 5, k])], [tin("y", [1, 5, n])], inits), {"x": f32(1, 5, k)}, 0.25, 0.05


import ops_extra  # noqa: E402
import ops_control  # noqa: E402

ops_extra.register(case, model, tin, const, f32)
ops_control.register(case, model, tin, const, f32)


def run_case(name, fn) -> tuple[bool, list[str]]:
    lines = []
    try:
        mbytes, feeds, atol, rtol, *rest = fn()
        cpu, g = C.sessions(mbytes, rest[0] if rest else None)

        # a case may give a list of feeds: one session, several runs in order
        # (that is how the sticky-input cache is exercised)
        runs = feeds if isinstance(feeds, list) else [feeds]
        names = [o.name for o in cpu.get_outputs()]
        ok = True
        for r, feed in enumerate(runs):
            ref = cpu.run(None, feed)
            out = g.run(None, feed)
            tag = "" if len(runs) == 1 else f"run{r} "
            for n, a, b in zip(names, ref, out):
                good, line = C.compare(tag + n, np.asarray(a), np.asarray(b), atol, rtol)
                ok = ok and good
                lines.append("   " + line)
        return ok, lines
    except Exception:  # noqa: BLE001
        lines.append("   " + traceback.format_exc().strip().replace("\n", "\n   "))
        return False, lines


def main(argv: list[str]) -> int:
    C.prepare()
    import onnxruntime_ggml as ggml

    print(f"provider library: {ggml.library_path()}")
    selected = {k: v for k, v in CASES.items() if not argv or any(a in k for a in argv)}
    failed = []
    for name, fn in selected.items():
        ok, lines = run_case(name, fn)
        print(f"{'PASS' if ok else 'FAIL'} {name}")
        for line in lines:
            print(line)
        if not ok:
            failed.append(name)
    print(f"\n{len(selected) - len(failed)}/{len(selected)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
