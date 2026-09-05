# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "onnxruntime>=1.29,<1.31",
# ]
# ///
"""Time the whisper `decoder_with_past` step on a given provider.

    uv run tests/bench_decoder.py --provider ggml
    uv run tests/bench_decoder.py --provider cpu --past 64

Builds synthetic past key/value caches for a given past length: the encoder
cross-attention caches are random and *constant* across steps (that is what the
real decode loop does), the decoder self-attention caches grow by one token per
step, exactly as onnxruntime's own generation loop feeds them back.

`ORT_GGML_LOG=debug` adds the provider's per-run stats.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, prepare  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "whisper-large-v3-turbo" / "onnx" / "decoder_with_past_model.onnx"


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
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--past", type=int, default=64, help="past decoder sequence length at the first step")
    ap.add_argument("--enc", type=int, default=1500, help="encoder sequence length")
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--dim", type=int, default=64)
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

    rng = np.random.default_rng(0)
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    layers = sorted({n.split(".")[1] for n in in_names if n.startswith("past_key_values.")}, key=int)

    feed: dict[str, np.ndarray] = {"input_ids": np.array([[50363]], dtype=np.int64)}
    for layer in layers:
        for kv in ("key", "value"):
            feed[f"past_key_values.{layer}.encoder.{kv}"] = rng.standard_normal(
                (1, args.heads, args.enc, args.dim), dtype=np.float32
            )
            feed[f"past_key_values.{layer}.decoder.{kv}"] = rng.standard_normal(
                (1, args.heads, args.past, args.dim), dtype=np.float32
            )

    times = []
    for step in range(args.steps):
        t = time.perf_counter()
        outs = sess.run(out_names, feed)
        times.append(time.perf_counter() - t)
        # feed the presents back, exactly as a generation loop does
        by_name = dict(zip(out_names, outs))
        for layer in layers:
            for kv in ("key", "value"):
                feed[f"past_key_values.{layer}.decoder.{kv}"] = by_name[f"present.{layer}.decoder.{kv}"]
        feed["input_ids"] = np.array([[int(np.argmax(by_name["logits"][0, -1]))]], dtype=np.int64)
        del step

    ms = [t * 1000 for t in times]
    steady = ms[1:] or ms
    opt = " ".join(f"{k}={v}" for k, v in options.items())
    print(f"model    {model.name}")
    print(f"provider {args.provider} {opt}".rstrip())
    print(f"load     {load:.2f} s")
    print(f"past     {args.past} -> {args.past + args.steps}  enc {args.enc}  layers {len(layers)}")
    print("steps    " + "  ".join(f"{t:.1f}" for t in ms))
    print(f"mean     {sum(steady) / len(steady):.1f} ms/step (excluding the first)")
    print(f"min      {min(steady):.1f} ms/step   first {ms[0]:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
