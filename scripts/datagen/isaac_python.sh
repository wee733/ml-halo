#!/usr/bin/env bash
# Python entrypoint for the upstream Isaac render driver.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
if [[ "${1:-}" != -p && "${1:-}" != --python ]]; then
    echo "Usage: scripts/datagen/isaac_python.sh -p <Python script or arguments>" >&2
    exit 2
fi
shift
export HALO_ISAACLAB_ROOT="${HALO_ISAACLAB_ROOT:-$HALO_LOCAL_ROOT/vendor/IsaacLab-2.3.2}"
if [[ ! -f "$HALO_ISAACLAB_ROOT/source/isaaclab/isaaclab/app/app_launcher.py" ]]; then
    echo "Isaac Lab compatibility overlay is missing. Run scripts/setup_workstation.sh first." >&2
    exit 1
fi
for package in isaaclab_tasks isaaclab_rl isaaclab_assets isaaclab; do
    export PYTHONPATH="$HALO_ISAACLAB_ROOT/source/$package:$PYTHONPATH"
done
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
exec "$HALO_PYTHON" "$@"
