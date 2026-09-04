# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "onnxruntime>=1.29,<1.31",
# ]
# ///
"""Time one encoder forward pass on a given provider.

    uv run tests/bench_encoder.py --provider ggml
    uv run tests/bench_encoder.py --provider cpu

Loads the whisper encoder (or any model taking a single float input), feeds a
random tensor of the model's input shape three times and prints the minimum
wall time. `ORT_GGML_LOG=debug` adds the provider's per-run stats.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, prepare  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "whisper-large-v3-turbo" / "onnx" / "encoder_model.onnx"


def build(model: str, provider: str, options: dict[str, str] | None):
    import onnxruntime as ort

    if provider == "cpu":
        so = ort.SessionOptions()
        so.log_severity_level = 3
        return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    import onnxruntime_ggml as ggml

    so = ort.SessionOptions()
    so.log_severity_level = 2
    return ggml.InferenceSession(model, options or {}, sess_options=so)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(DEFAULT_MODEL))
    ap.add_argument("--provider", choices=["cpu", "ggml"], default="ggml")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--shape", default="1,128,3000", help="input shape, comma separated")
    ap.add_argument("--option", action="append", default=[], metavar="k=v", help="provider option (ggml only)")
    args = ap.parse_args()

    prepare()
    model = Path(args.model)
    if not model.exists():
        print(f"missing model: {model}")
        return 1

    options = dict(o.split("=", 1) for o in args.option)
    t0 = time.perf_counter()
    sess = build(str(model), args.provider, options)
    load = time.perf_counter() - t0

    inp = sess.get_inputs()[0]
    shape = [int(d) for d in args.shape.split(",")]
    rng = np.random.default_rng(0)
    feed = {inp.name: rng.standard_normal(shape, dtype=np.float32)}
    out_names = [o.name for o in sess.get_outputs()]

    times = []
    outs = None
    for _ in range(args.runs):
        t = time.perf_counter()
        outs = sess.run(out_names, feed)
        times.append(time.perf_counter() - t)

    opt = " ".join(f"{k}={v}" for k, v in options.items())
    print(f"model    {model.name}")
    print(f"provider {args.provider} {opt}".rstrip())
    print(f"load     {load:.2f} s")
    print(f"input    {inp.name} {shape}")
    print(f"output   {out_names[0]} {list(outs[0].shape)}")
    print("times    " + "  ".join(f"{t * 1000:.0f} ms" for t in times))
    print(f"min      {min(times) * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
