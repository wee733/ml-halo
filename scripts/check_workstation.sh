#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$HALO_PYTHON" "$HALO_ROOT/scripts/check_workstation.py" "$@"
