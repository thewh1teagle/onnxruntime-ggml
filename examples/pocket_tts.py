# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "onnxruntime>=1.29,<1.31",
#   "pocket-tts-onnx @ git+https://github.com/thewh1teagle/pocket-tts-onnx",
#   "soundfile>=0.12",
# ]
# ///
"""Speak a line with pocket-tts on the ggml provider and time it.

    chore model
    uv run examples/pocket_tts.py "Hello from ggml."
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

MODEL = os.environ.get("POCKET_TTS_MODEL", str(ROOT / "models" / "pocket-tts-english-fp32.onnx"))


def main() -> None:
    import soundfile as sf
    from pocket_tts_onnx import PocketTTS

    import onnxruntime_ggml as ggml

    text = " ".join(sys.argv[1:]) or "Hello! This sentence was synthesised on ggml through onnxruntime."
    tts = PocketTTS(MODEL)
    tts.session = ggml.InferenceSession(MODEL)  # swap the CPU session for the ggml one
    for _ in tts.stream("Warming up.", voice="alba"):
        pass
    started = time.perf_counter()
    first = None
    frames = []
    for frame in tts.stream(text, voice="alba"):
        if first is None:
            first = time.perf_counter() - started
        frames.append(frame)
    total = time.perf_counter() - started
    import numpy as np

    audio = np.concatenate(frames)
    sf.write("pocket-tts-ggml.wav", audio, tts.sample_rate)
    print(f"first frame after {first * 1000:.0f} ms, {len(audio) / tts.sample_rate:.2f}s of audio in {total:.2f}s")
    print("wrote pocket-tts-ggml.wav")


if __name__ == "__main__":
    main()
