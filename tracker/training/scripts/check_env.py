"""Reset and step a tracker task, checking observations and available depth images."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Halo-MotionTracking-Robot_Only-PreRL")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--motion_data", nargs="+")
parser.add_argument("--motion_data_folder", nargs="+")
parser.add_argument("--report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.report:
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.unlink(missing_ok=True)
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num_envs and --steps must be positive")


def main():
    import gymnasium as gym
    import torch

    import halo_tracker.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cli_args.apply_motion_data_args(cfg, args)
    env = gym.make(args.task, cfg=cfg)
    try:
        observations, _ = env.reset()
        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene["contact_forces"]
        peak_forces = torch.zeros(contact_sensor.num_bodies, device=env.unwrapped.device)
        root_heights = []
        for _ in range(args.steps):
            observations, rewards, _, _, _ = env.step(actions)
            if not torch.isfinite(rewards).all():
                raise RuntimeError("Environment rewards contain non-finite values")
            for name, value in observations.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Observation {name} contains non-finite values")
            forces = torch.linalg.vector_norm(contact_sensor.data.net_forces_w, dim=-1).amax(dim=0)
            if not torch.isfinite(forces).all() or not torch.isfinite(robot.data.root_pos_w).all():
                raise RuntimeError("Robot poses or contact forces contain non-finite values")
            peak_forces = torch.maximum(peak_forces, forces)
            root_heights.append(robot.data.root_pos_w[:, 2].clone())
        depth = {}
        for name, sensor in env.unwrapped.scene.sensors.items():
            output = getattr(sensor.data, "output", {})
            if "distance_to_image_plane" in output:
                value = output["distance_to_image_plane"]
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Depth sensor {name} contains non-finite values")
                depth[name] = {
                    "shape": list(value.shape),
                    "min_m": value.min().item(),
                    "max_m": value.max().item(),
                    "body_mesh_count": len(getattr(sensor, "_rpl_body_meshes", [])),
                }
        report = {
            "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task, "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed,
            "device": str(env.unwrapped.device),
            "observations": {name: list(value.shape) for name, value in observations.items()},
            "action_shape": list(actions.shape), "finite": True, "depth": depth,
            "physics": {
                "root_height_min_m": torch.stack(root_heights).min().item(),
                "root_height_max_m": torch.stack(root_heights).max().item(),
                "peak_contact_force_n": peak_forces.max().item(),
                "contact_bodies_over_1n": [
                    name for name, force in zip(contact_sensor.body_names, peak_forces.tolist()) if force > 1.0
                ],
                "robot_mass_kg": robot.root_physx_view.get_masses()[0].sum().item(),
                "native_articulation_count": robot.root_physx_view.count,
                "native_max_shapes_per_articulation": robot.root_physx_view.max_shapes,
                "native_view_homogeneous": robot.root_physx_view.is_homogeneous,
            },
        }
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


exit_code = 0
app = None
try:
    app = AppLauncher(args).app
    main()
except Exception as error:
    import traceback

    traceback.print_exc()
    exit_code = 1
    if args.report:
        args.report.write_text(json.dumps({
            "status": "failed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task, "error": f"{type(error).__name__}: {error}",
        }, indent=2) + "\n")
finally:
    if app is not None:
        app.close(exit_code=exit_code)
sys.exit(exit_code)
