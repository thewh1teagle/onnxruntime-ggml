import sys

from onnxruntime_ggml import _main

raise SystemExit(_main(sys.argv[1:]))
