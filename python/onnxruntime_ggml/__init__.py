"""onnxruntime-ggml: register the ggml execution provider with onnxruntime.

The wheel bundles one native library per platform under ``lib/``. onnxruntime
loads it through ``register_execution_provider_library`` and from then on the
provider appears as an ``OrtEpDevice`` named ``ggml``.

    import onnxruntime_ggml as ggml
    session = ggml.InferenceSession("model.onnx", {"device": "gpu"})

Options (``device``: auto|gpu|cpu, ``threads``: int, ``partial``: 0|1, ``dump``:
0|1) can also come from the environment as ``ORT_GGML_DEVICE`` and friends;
``ORT_GGML_LOG=debug`` traces the provider.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import onnxruntime as ort

__all__ = ["EP_NAME", "InferenceSession", "devices", "library_path", "register", "session_options", "version"]
__version__ = "0.1.0"

EP_NAME = "ggml"
_registered: set[str] = set()


def version() -> str:
    return __version__


def _library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libonnxruntime_ggml.dylib"
    if system == "Windows":
        return "onnxruntime_ggml.dll"
    return "libonnxruntime_ggml.so"


def library_path() -> str:
    """Path of the native provider library.

    ``ONNXRUNTIME_GGML_LIBRARY`` overrides the bundled one, which is how a
    development build under ``target/release`` gets picked up.
    """
    override = os.environ.get("ONNXRUNTIME_GGML_LIBRARY")
    if override:
        return override
    bundled = Path(__file__).parent / "lib" / _library_name()
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError(
        f"no bundled provider library at {bundled}; this wheel was built without one for "
        f"{platform.system()} {platform.machine()}, or set ONNXRUNTIME_GGML_LIBRARY"
    )


def register(name: str = EP_NAME) -> None:
    """Load the provider library into onnxruntime, once per process."""
    if name in _registered:
        return
    path = library_path()
    ort.register_execution_provider_library(name, path)
    _registered.add(name)


def devices(name: str = EP_NAME) -> list[Any]:
    """The ``OrtEpDevice`` entries the provider registered (normally one, on the CPU device)."""
    register(name)
    found = [d for d in ort.get_ep_devices() if d.ep_name == name]
    if not found:
        raise RuntimeError(f"provider '{name}' registered but exposes no devices; see stderr with ORT_GGML_LOG=debug")
    return found


def session_options(
    options: Mapping[str, Any] | None = None,
    name: str = EP_NAME,
    base: ort.SessionOptions | None = None,
) -> ort.SessionOptions:
    """Session options with the provider selected first; the CPU provider stays as fallback."""
    so = base if base is not None else ort.SessionOptions()
    normalized = {str(k): str(v) for k, v in (options or {}).items()}
    so.add_provider_for_devices(devices(name), normalized)
    return so


def InferenceSession(  # noqa: N802 - mirrors onnxruntime's name
    path_or_bytes: str | bytes | os.PathLike,
    options: Mapping[str, Any] | None = None,
    sess_options: ort.SessionOptions | None = None,
    name: str = EP_NAME,
    **kwargs: Any,
) -> ort.InferenceSession:
    """``onnxruntime.InferenceSession`` with the ggml provider selected."""
    so = session_options(options, name=name, base=sess_options)
    return ort.InferenceSession(path_or_bytes, so, **kwargs)


def _main(argv: list[str]) -> int:
    """`python -m onnxruntime_ggml` prints where the library is and what it registers."""
    print(f"onnxruntime-ggml {__version__}")
    print(f"onnxruntime {ort.__version__}")
    try:
        print(f"library {library_path()}")
        for d in devices():
            print(f"device {d.ep_name} vendor={d.ep_vendor} hw={d.device.type} metadata={dict(d.ep_metadata)}")
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
