#!/usr/bin/env bash
set -euo pipefail
extractor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
extractor_python="$extractor_root/.venv/bin/python"
# Set library search paths only for this process; do not alter system CUDA.
extractor_cuda_libs="$($extractor_python - <<'PY'
import site
from pathlib import Path
paths = []
for root in site.getsitepackages():
    paths.extend(str(p) for p in sorted(Path(root).glob('nvidia/*/lib')) if p.is_dir())
    if (Path(root) / 'torch/lib').is_dir():
        paths.append(str(Path(root) / 'torch/lib'))
print(':'.join(paths))
PY
)"
export LD_LIBRARY_PATH="$extractor_cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$extractor_root/.."
exec "$extractor_python" "$@"
