# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "onnxruntime>=1.29,<1.31",
#   "transformers>=4.45",
#   "soundfile>=0.12",
# ]
# ///
"""Transcribe a wav with the whisper-large-v3-turbo ONNX export, on any onnxruntime provider.

    chore whisper-model
    uv run examples/whisper.py path/to/audio.wav --provider ggml
    uv run examples/whisper.py path/to/audio.wav --provider cpu

Plain onnxruntime sessions only: encoder, decoder (first step) and
decoder-with-past (every other step). Greedy decoding, 30-second windows,
no timestamps. Prints the text and the time spent in each stage.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
# prefer a fresh development build over whatever is bundled in python/
for _name in ("libonnxruntime_ggml.dylib", "libonnxruntime_ggml.so", "onnxruntime_ggml.dll"):
    _p = ROOT / "target" / "release" / _name
    if _p.exists():
        os.environ.setdefault("ONNXRUNTIME_GGML_LIBRARY", str(_p))

SAMPLE_RATE = 16000
WINDOW = 30 * SAMPLE_RATE
MAX_TOKENS = 448


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # linear resample; good enough for a benchmark input
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


class Whisper:
    def __init__(self, model_dir: Path, provider: str, threads: int | None = None):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_dir))
        self.tokenizer = WhisperTokenizerFast.from_pretrained(str(model_dir))
        onnx_dir = model_dir / "onnx"

        def session(name: str):
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if threads:
                so.intra_op_num_threads = threads
            if provider == "ggml":
                import onnxruntime_ggml as ggml

                return ggml.InferenceSession(str(onnx_dir / name), sess_options=so)
            return ort.InferenceSession(str(onnx_dir / name), so, providers=["CPUExecutionProvider"])

        t = time.perf_counter()
        self.encoder = session("encoder_model.onnx")
        self.decoder = session("decoder_model.onnx")
        self.decoder_past = session("decoder_with_past_model.onnx")
        self.load_s = time.perf_counter() - t

        self.past_inputs = [i.name for i in self.decoder_past.get_inputs() if i.name.startswith("past_key_values")]
        self.present_outputs = [o.name for o in self.decoder.get_outputs() if o.name.startswith("present")]
        self.present_past_outputs = [o.name for o in self.decoder_past.get_outputs() if o.name.startswith("present")]
        tok = self.tokenizer.convert_tokens_to_ids
        self.prompt = [tok("<|startoftranscript|>"), tok("<|en|>"), tok("<|transcribe|>"), tok("<|notimestamps|>")]
        self.eos = tok("<|endoftext|>")
        self.stats = {"mel_s": 0.0, "encoder_s": 0.0, "decoder_s": 0.0, "tokens": 0, "windows": 0}

    def transcribe_window(self, audio: np.ndarray) -> list[int]:
        t = time.perf_counter()
        feats = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features.astype(np.float32)
        self.stats["mel_s"] += time.perf_counter() - t

        t = time.perf_counter()
        (hidden,) = self.encoder.run(None, {"input_features": feats})
        self.stats["encoder_s"] += time.perf_counter() - t

        t = time.perf_counter()
        ids = np.array([self.prompt], dtype=np.int64)
        outs = self.decoder.run(None, {"input_ids": ids, "encoder_hidden_states": hidden})
        logits, presents = outs[0], outs[1:]
        past = {f"past_key_values{n[len('present'):]}": v for n, v in zip(self.present_outputs, presents)}
        tokens: list[int] = []
        next_id = int(np.argmax(logits[0, -1]))
        while next_id != self.eos and len(tokens) < MAX_TOKENS:
            tokens.append(next_id)
            feeds = {"input_ids": np.array([[next_id]], dtype=np.int64)}
            feeds.update({k: past[k] for k in self.past_inputs})
            outs = self.decoder_past.run(None, feeds)
            logits, presents = outs[0], outs[1:]
            for n, v in zip(self.present_past_outputs, presents):
                # decoder self-attention caches grow; encoder cross-attention caches are constant
                past[f"past_key_values{n[len('present'):]}"] = v
            next_id = int(np.argmax(logits[0, -1]))
        self.stats["decoder_s"] += time.perf_counter() - t
        self.stats["tokens"] += len(tokens)
        self.stats["windows"] += 1
        return tokens

    def transcribe(self, audio: np.ndarray) -> str:
        pieces = []
        for start in range(0, len(audio), WINDOW):
            chunk = audio[start : start + WINDOW]
            if len(chunk) < SAMPLE_RATE // 2:
                break
            pieces.append(self.tokenizer.decode(self.transcribe_window(chunk), skip_special_tokens=True).strip())
        return " ".join(p for p in pieces if p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--provider", choices=["cpu", "ggml"], default="ggml")
    ap.add_argument("--model", default=str(ROOT / "models" / "whisper-large-v3-turbo"))
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    os.environ.setdefault("ORT_GGML_LOG", "warn")

    audio = load_audio(args.wav)
    w = Whisper(Path(args.model), args.provider, args.threads)
    t = time.perf_counter()
    text = w.transcribe(audio)
    total = time.perf_counter() - t
    s = w.stats
    print(text)
    print(
        f"[{args.provider}] audio {len(audio) / SAMPLE_RATE:.1f}s | load {w.load_s:.2f}s | mel {s['mel_s']:.2f}s | "
        f"encoder {s['encoder_s']:.2f}s ({s['windows']} windows) | decoder {s['decoder_s']:.2f}s ({s['tokens']} tokens, "
        f"{1000 * s['decoder_s'] / max(s['tokens'], 1):.0f} ms/token) | total {total:.2f}s"
    )


if __name__ == "__main__":
    main()
