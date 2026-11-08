#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

export HALO_ISAACLAB_ROOT="${HALO_ISAACLAB_ROOT:-$HALO_LOCAL_ROOT/vendor/IsaacLab-2.3.2}"
if [[ ! -f "$HALO_ISAACLAB_ROOT/source/isaaclab/isaaclab/app/app_launcher.py" ]]; then
    echo "Tracker Isaac Lab overlay is missing. Run scripts/setup_workstation.sh first." >&2
    exit 1
fi
for package in isaaclab_tasks isaaclab_rl isaaclab_assets isaaclab; do
    export PYTHONPATH="$HALO_ISAACLAB_ROOT/source/$package:$PYTHONPATH"
done
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

mkdir -p "$HALO_RUNS_ROOT/tracker"

case "${1:-}" in
    --list-tasks)
        shift
        exec "$HALO_PYTHON" "$HALO_ROOT/tracker/training/scripts/list_envs.py" "$@"
        ;;
    --convert-csv)
        shift
        exec "$HALO_PYTHON" "$HALO_ROOT/tracker/data/scripts/extract_sequences.py" --headless "$@"
        ;;
    --check-env)
        shift
        exec "$HALO_PYTHON" "$HALO_ROOT/tracker/training/scripts/check_env.py" --headless "$@"
        ;;
esac

default_motion=true
show_help=false
for argument in "$@"; do
    case "$argument" in
        --motion_data|--motion_data=*|--motion_data_folder|--motion_data_folder=*) default_motion=false ;;
        --help|-h) show_help=true ;;
    esac
done

arguments=(--headless --device cuda:0 --task Halo-MotionTracking-Robot_Only-PreRL --num_envs 256
    "hydra.run.dir=$HALO_RUNS_ROOT/tracker/hydra/$(date +%Y-%m-%d_%H-%M-%S)_$$")
if [[ "$default_motion" == true && "$show_help" == false ]]; then
    motion_folder="${HALO_TRACKER_MOTIONS:-$HALO_DATA_ROOT/g1_processed/robot_only/lafan}"
    if ! "$HALO_PYTHON" -c 'from pathlib import Path; import sys; sys.exit(not any(Path(sys.argv[1]).glob("*.npz")))' "$motion_folder"; then
        echo "No tracker NPZ motions in $motion_folder." >&2
        echo "Convert G1 CSV clips with --convert-csv, or pass --motion_data_folder / --motion_data." >&2
        exit 1
    fi
    arguments+=(--motion_data_folder "$motion_folder")
fi

exec "$HALO_PYTHON" "$HALO_ROOT/tracker/training/scripts/rsl_rl/train.py" "${arguments[@]}" "$@"
