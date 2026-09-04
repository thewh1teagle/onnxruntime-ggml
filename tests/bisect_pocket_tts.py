# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy>=1.26",
#   "onnx>=1.16",
#   "onnxruntime>=1.29,<1.31",
#   "pocket-tts-onnx @ git+https://github.com/thewh1teagle/pocket-tts-onnx",
# ]
# ///
"""Find the first node in pocket-tts whose output goes NaN (or drifts) on ggml.

Adds intermediate tensors as extra graph outputs, runs the same feeds on the CPU
provider and on ggml, and reports the first one that disagrees.

    uv run tests/bisect_pocket_tts.py [model] --step 50
    uv run tests/bisect_pocket_tts.py --lo 900 --hi 1000 --step 1
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

FEEDS = C.ROOT / "target" / "pocket_feeds.pkl"


def capture_feeds(model: str, text: str, voice: str) -> list[dict]:
    """Run pocket-tts on the CPU provider once and keep the raw feeds."""
    if FEEDS.exists():
        with FEEDS.open("rb") as f:
            return pickle.load(f)
    from pocket_tts_onnx import PocketTTS

    calls: list[dict] = []
    tts = PocketTTS(model, providers=["CPUExecutionProvider"])
    inner = tts.session

    class Rec:
        def __getattr__(self, name):
            return getattr(inner, name)

        def run(self, output_names, feeds, run_options=None):
            if len(calls) < 4:
                calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
            return inner.run(output_names, feeds, run_options)

    tts.session = Rec()
    tts.create(text, voice=voice, temperature=0.0)
    FEEDS.parent.mkdir(parents=True, exist_ok=True)
    with FEEDS.open("wb") as f:
        pickle.dump(calls, f)
    return calls


def candidates(model, lo: int, hi: int, step: int) -> list[tuple[int, str, str]]:
    """(node index, op type, output name) for the float-valued nodes we probe."""
    graph = model.graph
    inferred = onnx.shape_inference.infer_shapes(model)
    ftype = onnx.TensorProto.FLOAT
    floats = {
        v.name
        for v in list(inferred.graph.value_info) + list(inferred.graph.output)
        if v.type.HasField("tensor_type") and v.type.tensor_type.elem_type == ftype
    }
    skip = {"Constant", "Shape", "ConstantOfShape"}
    existing = {o.name for o in graph.output}
    out = []
    for i, n in enumerate(graph.node):
        if i < lo or i > hi or n.op_type in skip or not n.output or not n.output[0]:
            continue
        if n.output[0] in existing or n.output[0] not in floats:
            continue
        out.append((i, n.op_type, n.output[0]))
    return out[:: max(1, step)] if step > 1 else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(C.ROOT / "models" / "pocket-tts-english-fp32.onnx"))
    ap.add_argument("--run", type=int, default=0, help="which recorded run's feeds to replay")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=10**9)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--max", type=int, default=80, help="cap on how many probes per pass")
    ap.add_argument("--text", default="Hello! I am streaming this to you frame by frame, as I generate it.")
    ap.add_argument("--voice", default="alba")
    args = ap.parse_args()
    C.prepare()

    import onnxruntime as ort

    import onnxruntime_ggml as ggml

    feeds = capture_feeds(args.model, args.text, args.voice)[args.run]

    model = onnx.load(args.model)
    probes = candidates(model, args.lo, args.hi, args.step)[: args.max]
    print(f"{len(model.graph.node)} nodes, probing {len(probes)} (indices {probes[0][0]}..{probes[-1][0]})")

    keep = [o.name for o in model.graph.output]
    for _, _, name in probes:
        model.graph.output.extend([onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)])
    blob = model.SerializeToString()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    # ORT will constant-fold away tensors we asked for unless optimisation is off.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
    names = [o.name for o in cpu.get_outputs()]
    ref = cpu.run(None, feeds)

    gs = ggml.InferenceSession(blob)
    got = gs.run(None, feeds)

    index = {name: (i, op) for i, op, name in probes}
    first = None
    for name, a, b in zip(names, ref, got):
        if name in keep:
            continue
        i, op = index[name]
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        bad_nan = bool(np.isnan(b).any() and not np.isnan(a).any())
        if a.shape != b.shape:
            note, bad = f"shape {a.shape} vs {b.shape}", True
        else:
            d = np.abs(a - b)
            m = float(d.max()) if d.size else 0.0
            bad = bad_nan or m > 1e-3
            note = f"max_abs={m:.3e}{' NAN' if bad_nan else ''}"
        mark = "BAD " if bad else "ok  "
        print(f"{mark}node {i:5d} {op:22s} {name[:60]:60s} {note}")
        if bad and first is None:
            first = (i, op, name)
    if first:
        print(f"\nfirst bad: node {first[0]} {first[1]} -> {first[2]}")
    else:
        print("\nno bad probe in this range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
