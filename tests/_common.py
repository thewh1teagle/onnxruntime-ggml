"""Shared bits for the test scripts: find the provider (a dev build or the
wheel), build sessions, compare outputs."""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def dev_library() -> Path | None:
    name = {"Darwin": "libonnxruntime_ggml.dylib", "Windows": "onnxruntime_ggml.dll"}.get(platform.system(), "libonnxruntime_ggml.so")
    for profile in ("release", "debug"):
        p = ROOT / "target" / profile / name
        if p.exists():
            return p
    return None


def prepare() -> None:
    """Point the wrapper at target/release when no library is bundled."""
    if "ONNXRUNTIME_GGML_LIBRARY" not in os.environ:
        lib = dev_library()
        if lib is not None:
            os.environ["ONNXRUNTIME_GGML_LIBRARY"] = str(lib)
    os.environ.setdefault("ORT_GGML_LOG", "info")


def sessions(model, options=None):
    """(cpu session, ggml session) for the same model."""
    import onnxruntime as ort

    import onnxruntime_ggml as ggml

    so = ort.SessionOptions()
    so.log_severity_level = 3
    cpu = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    gso = ort.SessionOptions()
    gso.log_severity_level = 2
    gso.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    g = ggml.InferenceSession(model, options, sess_options=gso)
    return cpu, g


def compare(name: str, a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"{name}: shape {a.shape} vs {b.shape}"
    if a.dtype.kind in "iub":
        ok = np.array_equal(a, b)
        return ok, f"{name}: exact {'ok' if ok else 'MISMATCH'} ({a.shape})"
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = np.abs(a64 - b64)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(a64), 1e-6)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    ok = bool(np.all(diff <= atol + rtol * np.abs(a64)))
    nan_mismatch = bool(np.any(np.isnan(a64) != np.isnan(b64)))
    ok = ok and not nan_mismatch
    return ok, f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} shape={a.shape} {'ok' if ok else 'FAIL'}"


def timed(fn, repeat: int = 3):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best
