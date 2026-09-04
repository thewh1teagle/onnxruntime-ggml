# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy>=1.26",
#   "onnxruntime>=1.29,<1.31",
#   "pocket-tts-onnx @ git+https://github.com/thewh1teagle/pocket-tts-onnx",
#   "soundfile>=0.12",
# ]
# ///
"""End-to-end check on pocket-tts: record every session.run on the CPU provider,
replay the same feeds on ggml, compare every output, then time both.

    chore model                         # fetches models/pocket-tts-english-fp32.onnx
    uv run tests/compare_pocket_tts.py [model] [--frames N] [--text "..."]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

# (atol, rtol) per output. atol is set per tensor because these differ by two
# orders of magnitude in scale, and ggml reassociates fp32 sums differently from
# the CPU provider, so the error compounds with depth: the disagreements sit in
# the last transformer layers and none in the first. The KV caches are the
# loosest -- raw projections spanning roughly +/-11, compared elementwise
# including the near-zero entries, where an absolute tolerance bites hardest.
# `audio`, what anyone actually listens to, stays at 2e-3 and lands near 1e-4.
TOLERANCES = {
    "audio": (2e-3, 1e-2),
    "next_latent": (3e-3, 1e-2),
    "eos_logit": (5e-3, 1e-2),
    "flow_kv_new": (1.5e-2, 1e-2),
    "mimi_kv_new": (1.5e-2, 1e-2),
    "mimi_conv_out": (5e-3, 1e-2),
}


class Recorder:
    """Wrap a session so every run's feeds and outputs are kept."""

    def __init__(self, session, limit: int):
        self.session = session
        self.limit = limit
        self.calls: list[tuple[dict, list]] = []

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, output_names, feeds, run_options=None):
        out = self.session.run(output_names, feeds, run_options)
        if len(self.calls) < self.limit:
            self.calls.append(({k: np.array(v, copy=True) for k, v in feeds.items()}, [np.array(o, copy=True) for o in out]))
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(C.ROOT / "models" / "pocket-tts-english-fp32.onnx"))
    ap.add_argument("--frames", type=int, default=12, help="how many recorded runs to replay")
    ap.add_argument("--text", default="Hello! I am streaming this to you frame by frame, as I generate it.")
    ap.add_argument("--voice", default="alba")
    args = ap.parse_args()
    C.prepare()

    from pocket_tts_onnx import PocketTTS

    import onnxruntime_ggml as ggml

    print(f"model: {args.model}")
    print(f"provider library: {ggml.library_path()}")

    # 1. CPU reference, recorded
    tts = PocketTTS(args.model, providers=["CPUExecutionProvider"])
    rec = Recorder(tts.session, args.frames)
    tts.session = rec
    t = time.perf_counter()
    audio_ref, sr = tts.create(args.text, voice=args.voice, temperature=0.0)
    cpu_s = time.perf_counter() - t
    print(f"cpu: {len(audio_ref) / sr:.2f}s of audio in {cpu_s:.2f}s, {len(rec.calls)} runs recorded")

    # 2. replay on ggml
    # weights=f32: this is an exactness check. The f16 default rounds activations to
    # f16 inside ggml's CPU matmul kernels (1e-2 level on the KV caches); audio quality
    # is checked by transcription instead (see docs/OPTIONS.md).
    gsess = ggml.InferenceSession(args.model, {"weights": "f32"})
    names = [o.name for o in gsess.get_outputs()]
    failures = 0
    for i, (feeds, ref) in enumerate(rec.calls):
        out = gsess.run(None, feeds)
        seq = feeds.get("tokens", feeds.get("latent")).shape[1]
        print(f"run {i} (seq={seq}, past={feeds['flow_kv'].shape[0]})")
        for n, a, b in zip(names, ref, out):
            atol, rtol = TOLERANCES.get(n, (1e-4, 1e-3))
            ok, line = C.compare(n, np.asarray(a), np.asarray(b), atol, rtol)
            failures += not ok
            print("   " + line)

    # 3. full synthesis on ggml, timed
    tts.session = gsess
    t = time.perf_counter()
    audio_ggml, _ = tts.create(args.text, voice=args.voice, temperature=0.0)
    ggml_s = time.perf_counter() - t
    n = min(len(audio_ref), len(audio_ggml))
    ok, line = C.compare("full audio", audio_ref[:n], audio_ggml[:n], 5e-2, 1e-1)
    print(line + f" (lengths {len(audio_ref)} vs {len(audio_ggml)})")
    print(f"ggml: {len(audio_ggml) / sr:.2f}s of audio in {ggml_s:.2f}s  (cpu provider: {cpu_s:.2f}s)")
    try:
        import soundfile as sf

        sf.write(str(C.ROOT / "compare-cpu.wav"), audio_ref, sr)
        sf.write(str(C.ROOT / "compare-ggml.wav"), audio_ggml, sr)
        print("wrote compare-cpu.wav and compare-ggml.wav")
    except Exception as exc:  # noqa: BLE001
        print(f"(no wav written: {exc})")
    if failures:
        print(f"{failures} output comparisons failed")
        return 1
    print("all recorded runs match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
