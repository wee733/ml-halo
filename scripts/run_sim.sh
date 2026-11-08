#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HALO_TRACKER_BIN="${HALO_TRACKER_BIN:-$HALO_LOCAL_ROOT/build/sim/motiontrack_mujoco_zmq}"
if [[ ! -x "$HALO_TRACKER_BIN" ]]; then
    echo "Simulation tracker missing at $HALO_TRACKER_BIN. Run bash scripts/setup_sim.sh." >&2
    exit 1
fi
arguments=("$@")
has_output=0
for argument in "$@"; do
    case "$argument" in
        --output_dir|--output_dir=*|-h|--help) has_output=1 ;;
    esac
done
if [[ $has_output == 0 ]]; then
    arguments+=(--output_dir "$HALO_RUNS_ROOT/sim/run_$(date +%Y%m%d_%H%M%S_%N)")
fi
exec "$HALO_PYTHON" "$HALO_ROOT/system/halo_system/scripts/run_sim.py" "${arguments[@]}"
