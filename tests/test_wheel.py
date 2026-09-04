# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "onnx>=1.16", "onnxruntime>=1.29,<1.31"]
# ///
"""End-to-end check of an installed onnxruntime-ggml wheel.

    uv run tests/test_wheel.py                          # the current interpreter
    uv run tests/test_wheel.py /tmp/ggml-wheel-test/bin/python

Builds a two-node graph (Add then Mul) with onnx.helper here, then runs it in
the target interpreter twice: once through ``onnxruntime_ggml.InferenceSession``
and once on the plain CPU provider, and compares the outputs. The target only
needs numpy, onnxruntime and the wheel itself; onnx is used on this side.

``ONNXRUNTIME_GGML_LIBRARY`` is cleared for the child, so the check always
exercises the library bundled inside the wheel rather than a dev build.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

OPSET = 17

# Runs inside the target interpreter: no onnx, only numpy + onnxruntime + the wheel.
CHILD = r'''
import sys
import numpy as np
import onnxruntime as ort
import onnxruntime_ggml as ggml

model = sys.argv[1]
print("library", ggml.library_path())

rng = np.random.default_rng(0)
feeds = {n: rng.standard_normal((2, 3), dtype=np.float32) for n in ("a", "b", "c")}

ggml_out = ggml.InferenceSession(model).run(None, feeds)[0]
cpu_out = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(None, feeds)[0]

if ggml_out.shape != cpu_out.shape:
    raise SystemExit(f"shape mismatch: ggml {ggml_out.shape} vs cpu {cpu_out.shape}")
np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-5, atol=1e-6)
print("outputs match", ggml_out.shape, ggml_out.dtype)
'''


def build_model(path: Path) -> None:
    """out = (a + b) * c, all float32 [2, 3]."""
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [2, 3]) for n in ("a", "b", "c")]
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3])
    nodes = [
        helper.make_node("Add", ["a", "b"], ["t"], name="add"),
        helper.make_node("Mul", ["t", "c"], ["out"], name="mul"),
    ]
    graph = helper.make_graph(nodes, "add_mul", inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main(argv: list[str]) -> int:
    python = argv[0] if argv else sys.executable
    if not Path(python).exists():
        print(f"no such interpreter: {python}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env.pop("ONNXRUNTIME_GGML_LIBRARY", None)

    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "add_mul.onnx"
        build_model(model)
        print(f"running {model.name} under {python}")
        result = subprocess.run([python, "-c", CHILD, str(model)], env=env)

    if result.returncode != 0:
        print("FAIL", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
