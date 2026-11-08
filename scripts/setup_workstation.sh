#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    cat <<'EOF'
Usage: HALO_PYTHON=/path/to/existing/python bash scripts/setup_workstation.sh [--download-model]
Installs training/data dependencies into an existing CUDA Python environment.
Version snapshots, compatibility dependencies and simulation builds go under .local/.
Reuses installed Blender 5.2 and prepares the required Infinigen Python 3.11 runtime.
--download-model caches the Qwen3.5-2B backbone from Hugging Face.
EOF
    exit 0
fi
download_model=0
if [[ "${1:-}" == --download-model && $# == 1 ]]; then
    download_model=1
elif [[ $# != 0 ]]; then
    echo "Unknown arguments. Use --help." >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
command -v uv >/dev/null
"$HALO_PYTHON" - <<'PY'
import sys, torch
if sys.version_info[:2] != (3, 12):
    raise SystemExit('This workstation profile was configured for Python 3.12.')
if not torch.cuda.is_available():
    raise SystemExit('Select the existing CUDA-enabled training Python using HALO_PYTHON.')
print('Training GPU:', torch.cuda.get_device_name())
PY
mkdir -p "$HALO_LOCAL_ROOT/environment" "$HALO_LOCAL_ROOT/python" "$HALO_DATA_ROOT" "$HALO_RUNS_ROOT"
snapshot="$HALO_LOCAL_ROOT/environment/before-$(date -u +%Y%m%dT%H%M%SZ).txt"
uv pip freeze --python "$HALO_PYTHON" > "$snapshot"
"$HALO_PYTHON" - "$snapshot" "$HALO_LOCAL_ROOT/environment/constraints.txt" <<'PY'
from pathlib import Path
import re, sys
lines = Path(sys.argv[1]).read_text().splitlines()
Path(sys.argv[2]).write_text('\n'.join(x for x in lines if re.match(r'^[A-Za-z0-9_.-]+==', x))+'\n')
PY
uv pip install --python "$HALO_PYTHON" -c "$HALO_LOCAL_ROOT/environment/constraints.txt" \
    -r "$HALO_ROOT/requirements/workstation.txt"
uv pip install --python "$HALO_PYTHON" --target "$HALO_LOCAL_ROOT/python" --no-deps \
    --build-constraint "$HALO_ROOT/requirements/build-constraints.txt" \
    -r "$HALO_ROOT/requirements/runtime-overlay.txt"
uv pip install --python "$HALO_PYTHON" --no-deps \
    -e "$HALO_ROOT/generator" -e "$HALO_ROOT/tracker/training/source/halo_tracker" \
    -e "$HALO_ROOT/system/comm/python" -e "$HALO_ROOT/system/serve/python" -e "$HALO_ROOT/system"
"$HALO_PYTHON" "$HALO_ROOT/tracker/training/scripts/prepare_isaaclab_overlay.py" \
    --root "${HALO_ISAACLAB_ROOT:-$HALO_LOCAL_ROOT/vendor/IsaacLab-2.3.2}"
bash "$HALO_ROOT/scripts/datagen/setup_kimodo.sh"
bash "$HALO_ROOT/scripts/datagen/setup_scene.sh"
bash "$HALO_ROOT/scripts/setup_sim.sh"
uv pip freeze --python "$HALO_PYTHON" > "$HALO_LOCAL_ROOT/environment/installed.txt"
if [[ $download_model == 1 ]]; then
    "$HALO_PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3.5-2B', allow_patterns=[
    '*.json', '*.safetensors', '*.txt', '*.jinja', 'tokenizer*', 'merges*', 'vocab*'
], max_workers=4))
PY
fi
echo "Workstation dependencies installed. Run bash scripts/check_workstation.sh."
