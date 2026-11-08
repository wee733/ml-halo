# Workstation Isaac Lab overlay

`prepare_isaaclab_overlay.py` checks out Isaac Lab v2.3.2 into the project's local vendor directory and applies
`isaaclab-2.3.2-sim6.patch`. `scripts/train_tracker.sh` selects this source through `PYTHONPATH` while using the
workstation's existing Python environment. The pinned source remains under Isaac Lab's BSD-3-Clause license.

The adapter supports HALO's rigid-body tasks on Isaac Sim 6.2. It updates the PhysX tensor import location,
defers unused legacy deformable type annotations, and guards the removed legacy deformable schema during
rigid-body material binding. The URDF path uses Isaac Sim's supported 3.x importer with the configured joint
drives, fixed-joint merging, and convex collider approximation.

Local mesh filenames are resolved before conversion. Missing local mesh files stop conversion. The generated
asset selects its PhysX or generic USD physics variant and reparents rigid links to the paths expected by HALO's
sensors. USD namespace edits preserve joint relationship targets, and link transforms are preserved in the robot
root frame. A free, jointless object receives rigid-body semantics. Conversion caches are process-local and
regenerated from the source URDF when a run starts.

The adapter requires explicit PD gains and supports the cylinder geometry used by HALO's system-identified G1.
Requests to replace actual cylinders with capsules, convert mimic joints, or override the root link raise errors.

## Validation commands

Run these from the repository root after workstation setup. Supply the G1 joint-space CSV and motion NPZ paths.

```bash
scripts/train_tracker.sh --list-tasks
scripts/train_tracker.sh --convert-csv --input /path/to/walk.csv --output .local/data/walk.npz --frame_range 1 60
scripts/train_tracker.sh --check-env --task Halo-MotionTracking-Distillation-ProprioDepth \
  --num_envs 1 --steps 10 --motion_data .local/data/walk.npz --report .local/validation/tracker_depth.json
scripts/train_tracker.sh --num_envs 4 --max_iterations 1 --motion_data .local/data/walk.npz \
  --run_name validation agent.torch_compile_mode=null
```

The workstation launcher defaults to 256 environments. The first training phase produces a privileged teacher.
Distillation and PostRL training additionally require motion datasets and trained teacher checkpoints configured
in a teacher registry. The bounded checks above cover environment execution and the training pipeline.
