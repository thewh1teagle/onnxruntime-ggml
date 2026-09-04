# /// script
# requires-python = ">=3.10"
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
    return model(nodes, [tin("x", [3, 4])], [tin("y", [3, 6])], [const("w", w), const("b", b)]), {"x": f32(3, 4)}, 1e-5, 1e-4


@case
def matmul_const_weight():
    w = f32(4, 6)
    nodes = [helper.make_node("MatMul", ["x", "w"], ["y"])]
    return model(nodes, [tin("x", [1, 3, 4])], [tin("y", [1, 3, 6])], [const("w", w)]), {"x": f32(1, 3, 4)}, 1e-5, 1e-4


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


def run_case(name, fn) -> tuple[bool, list[str]]:
    lines = []
    try:
        mbytes, feeds, atol, rtol = fn()
        cpu, g = C.sessions(mbytes)
        ref = cpu.run(None, feeds)
        out = g.run(None, feeds)
        names = [o.name for o in cpu.get_outputs()]
        ok = True
        for n, a, b in zip(names, ref, out):
            good, line = C.compare(n, np.asarray(a), np.asarray(b), atol, rtol)
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
