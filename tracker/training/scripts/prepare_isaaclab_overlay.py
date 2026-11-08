"""Prepare a project-local Isaac Lab 2.3.2 overlay for the workstation's Isaac Sim 6.2."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ISAACLAB_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
PATCH = Path(__file__).resolve().parent / "compat" / "isaaclab-2.3.2-sim6.patch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Project-local Isaac Lab overlay directory.")
    args = parser.parse_args()
    root = args.root.resolve()

    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "v2.3.2",
             "https://github.com/isaac-sim/IsaacLab.git", str(root)],
            check=True,
        )
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != ISAACLAB_COMMIT:
        raise RuntimeError(f"Expected Isaac Lab v2.3.2 at {ISAACLAB_COMMIT}, found {commit} in {root}")

    command = ["git", "-C", str(root), "apply"]
    already_applied = subprocess.run(
        command + ["--reverse", "--check", str(PATCH)], capture_output=True, text=True
    ).returncode == 0
    if not already_applied:
        subprocess.run(command + ["--check", str(PATCH)], check=True)
        subprocess.run(command + [str(PATCH)], check=True)
    print(f"HALO Isaac Lab overlay ready at {root}")


if __name__ == "__main__":
    main()
