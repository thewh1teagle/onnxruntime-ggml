# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""whisper-large-v3-turbo, three ways, one wav:

- whisper.cpp through vibe-server (ggml, Metal), the reference
- the ONNX export on onnxruntime's CPU provider
- the ONNX export on onnxruntime-ggml

    chore bench-whisper [wav]

Each run is a subprocess so nothing shares a process or a cache. Prints the
transcript of each and a timing table.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIBE = Path(os.environ.get("VIBE_SERVER", "/Users/yqbqwlny/Documents/projects/audio/vibe/server/target/release/vibe-server"))
GGML_MODEL = Path(
    os.environ.get(
        "WHISPER_GGML_MODEL",
        os.path.expanduser("~/Library/Application Support/github.com.thewh1teagle.vibe/ggml-large-v3-turbo.bin"),
    )
)


def run(cmd: list[str], env: dict | None = None) -> tuple[str, float]:
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    dt = time.perf_counter() - t
    if p.returncode != 0:
        print(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-3000:]}", file=sys.stderr)
    return p.stdout + "\n" + p.stderr, dt


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[len(hyp)] / max(len(ref), 1)


def main() -> int:
    wav = sys.argv[1] if len(sys.argv) > 1 else "/Users/yqbqwlny/Documents/projects/audio/vibe/server/fixtures/multi.wav"
    results = {}

    if VIBE.exists() and GGML_MODEL.exists():
        out, dt = run([str(VIBE), "transcribe", str(GGML_MODEL), wav, "--language", "en"])
        # vibe-server prints segments; keep the text lines, drop timestamps and logs
        text = " ".join(
            re.sub(r"^\[[^\]]*\]\s*", "", line).strip()
            for line in out.splitlines()
            if line.strip() and not line.startswith(("{", "whisper", "ggml", "INFO", "WARN", "DEBUG", "[20"))
        )
        results["whisper.cpp (vibe-server)"] = (text, dt, "")
    else:
        print(f"skipping vibe-server: {VIBE} or {GGML_MODEL} missing")

    for provider in ("cpu", "ggml"):
        out, dt = run(["uv", "run", str(ROOT / "examples" / "whisper.py"), wav, "--provider", provider], {"ORT_GGML_LOG": "warn"})
        lines = [l for l in out.splitlines() if l.strip()]
        stage = next((l for l in lines if l.startswith(f"[{provider}]")), "")
        text = next((l for l in reversed(lines) if not l.startswith("[") and not l.startswith(("INFO", "WARN", "ggml"))), "")
        results[f"onnxruntime {provider}"] = (text, dt, stage)

    ref = words(next(iter(results.values()))[0]) if results else []
    print("\n=== transcripts")
    for name, (text, _, _) in results.items():
        print(f"\n[{name}]\n{text}")
    print("\n=== timing (whole process, includes model load)")
    print(f"{'path':32s} {'seconds':>8s} {'WER vs first':>13s}")
    for name, (text, dt, stage) in results.items():
        print(f"{name:32s} {dt:8.2f} {wer(ref, words(text)):13.3f}")
        if stage:
            print(f"    {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
