#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help && $# == 1 ]]; then
    echo "Usage: bash scripts/datagen/setup_scene.sh"
    echo "Reuses installed Blender 5.2, installs its bake dependencies, and prepares local Infinigen Python 3.11."
    exit 0
fi
if [[ $# != 0 ]]; then
    echo "Unknown arguments. Use --help." >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
command -v uv >/dev/null
if [[ -z "${BLENDER_DIR:-}" ]]; then
    BLENDER_DIR="$(dirname -- "$(readlink -f -- "$(command -v blender)")")"
fi
export BLENDER_DIR
export INFINIGEN_ROOT="${INFINIGEN_ROOT:-$HALO_LOCAL_ROOT/vendor/infinigen}"
export INFINIGEN_ENV="${INFINIGEN_ENV:-$HALO_LOCAL_ROOT/runtimes/infinigen}"
infinigen_ref=892947be71c4ff102c6a558d088ffe149561c11f
blender_python=("$BLENDER_DIR"/*/python/bin/python3*)
if [[ ! -x "$BLENDER_DIR/blender" || ! -x "${blender_python[0]}" ]]; then
    echo "Set BLENDER_DIR to an installed Blender 5.2 distribution." >&2
    exit 2
fi
blender_version="$("$BLENDER_DIR/blender" --version | head -1)"
if [[ "$blender_version" != Blender\ 5.2.* ]]; then
    echo "This scene profile requires Blender 5.2. Found $blender_version." >&2
    exit 2
fi
echo "$blender_version"
mkdir -p "$HALO_LOCAL_ROOT/environment" "$(dirname -- "$INFINIGEN_ROOT")"
snapshot="$HALO_LOCAL_ROOT/environment/blender-before-scene-$(date -u +%Y%m%dT%H%M%SZ).txt"
uv pip freeze --python "${blender_python[0]}" > "$snapshot"
uv pip install --python "${blender_python[0]}" numpy opencv-python-headless usd-core scipy matplotlib imageio
if [[ ! -d "$INFINIGEN_ROOT/.git" ]]; then
    git clone --filter=blob:none https://github.com/princeton-vl/infinigen.git "$INFINIGEN_ROOT"
    git -C "$INFINIGEN_ROOT" checkout "$infinigen_ref"
fi
if [[ "$(git -C "$INFINIGEN_ROOT" rev-parse HEAD)" != "$infinigen_ref" ]]; then
    echo "Infinigen must use HALO's pinned revision $infinigen_ref. Current checkout is at $INFINIGEN_ROOT." >&2
    exit 2
fi
git -C "$INFINIGEN_ROOT" submodule update --init --depth 1 infinigen/infinigen_gpl infinigen/OcMesher
if [[ ! -x "$INFINIGEN_ENV/bin/python" ]]; then
    uv venv --python 3.11 --seed --relocatable "$INFINIGEN_ENV"
fi
# Existing prefixes can be moved with the workspace. Repair their console scripts,
# then reinstall the editable source below so its .pth points at the current source.
"$HALO_PYTHON" - "$INFINIGEN_ENV" "$HALO_LOCAL_ROOT/environment/infinigen-prefix.txt" <<'PY'
from pathlib import Path
import sys

runtime = Path(sys.argv[1]).resolve()
marker = Path(sys.argv[2])
previous = marker.read_text().strip() if marker.exists() else str(runtime)
if previous != str(runtime):
    for script in (runtime / "bin").iterdir():
        if script.is_symlink() or not script.is_file():
            continue
        content = script.read_bytes()
        if content.startswith(b"#!") or script.name.startswith("activate"):
            script.write_bytes(content.replace(previous.encode(), str(runtime).encode()))
marker.write_text(str(runtime) + "\n")
PY
uv pip install --python "$INFINIGEN_ENV/bin/python" -r "$HALO_ROOT/configs/datagen/infinigen_requirements.txt"
INFINIGEN_MINIMAL_INSTALL=True INFINIGEN_INSTALL_TERRAIN=False INFINIGEN_INSTALL_CUSTOMGT=False \
    uv pip install --python "$INFINIGEN_ENV/bin/python" --no-build-isolation --no-deps -e "$INFINIGEN_ROOT"
"${blender_python[0]}" -c 'import numpy, cv2, pxr, scipy, matplotlib, imageio; print("Blender bake imports passed")'
(
    cd "$INFINIGEN_ROOT"
    "$INFINIGEN_ENV/bin/python" -c 'import bpy, infinigen_examples.generate_indoors; print("Infinigen indoor imports passed", bpy.app.version_string)'
)
uv pip freeze --python "$INFINIGEN_ENV/bin/python" > "$HALO_LOCAL_ROOT/environment/infinigen-installed.txt"
