#!/usr/bin/env bash
# Source from the workstation and data entrypoints.
HALO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HALO_ROOT/.local/settings.env" ]]; then
    source "$HALO_ROOT/.local/settings.env"
fi
export HALO_ROOT
export HALO_LOCAL_ROOT="$(realpath -ms -- "${HALO_LOCAL_ROOT:-$HALO_ROOT/.local}")"
export HALO_DATA_ROOT="$(realpath -ms -- "${HALO_DATA_ROOT:-$HALO_LOCAL_ROOT/data}")"
export HALO_RUNS_ROOT="$(realpath -ms -- "${HALO_RUNS_ROOT:-$HALO_LOCAL_ROOT/runs}")"
if [[ -z "${HALO_PYTHON:-}" ]]; then
    if [[ -x "$HOME/Isaac/IsaacLab/.venv/bin/python" ]]; then
        HALO_PYTHON="$HOME/Isaac/IsaacLab/.venv/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        HALO_PYTHON="$VIRTUAL_ENV/bin/python"
    else
        HALO_PYTHON="$(command -v python3)"
    fi
fi
export HALO_PYTHON="$(realpath -ms -- "$(command -v "$HALO_PYTHON")")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$HALO_LOCAL_ROOT/python:$HALO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
