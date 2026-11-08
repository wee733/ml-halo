#!/usr/bin/env python3
"""Run the upstream sit data stages with explicit local paths and existing runtimes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DATAGEN = ROOT / "datagen"
MODELS = (
    "nvidia/Kimodo-G1-RP-v1",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)


def link_cached_models(env):
    from huggingface_hub import snapshot_download

    for repo in MODELS:
        try:
            snapshot = Path(snapshot_download(repo, local_files_only=True))
        except Exception as exc:
            raise SystemExit(
                f"Model cache unavailable for {repo} ({type(exc).__name__}). "
                "Run scripts/datagen/setup_kimodo.sh --download-models after obtaining model access."
            ) from None
        if repo.startswith("nvidia/"):
            target = Path(env["CHECKPOINT_DIR"]) / repo.split("/")[1]
        else:
            target = Path(env["TEXT_ENCODERS_DIR"]) / repo
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            if target.resolve() != snapshot.resolve():
                raise SystemExit(f"Cached model link differs from the resolved snapshot at {target}")
        else:
            target.symlink_to(snapshot, target_is_directory=True)


def prepare_run(run, profile, manifest, scene_names):
    """Publish complete shared metadata when motion/render shards start concurrently."""
    with (run / ".config.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        profile_path = run / "profile.json"
        if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
            raise SystemExit(f"Configuration differs from {profile_path}. Choose a new --run name.")
        files = {
            "profile.json": json.dumps(profile, indent=2) + "\n",
            "scene_manifest.json": json.dumps(manifest) + "\n",
            "render_scenes.txt": "".join(name + "\n" for name in scene_names),
        }
        for name, content in files.items():
            temporary = run / (name + ".tmp")
            temporary.write_text(content)
            temporary.replace(run / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("scene", "motion", "render", "stitch"))
    parser.add_argument("--run", required=True, help="A distinct dataset batch name, reused to resume")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/datagen/sit_5090.json")
    parser.add_argument("--backend", choices=("mujoco", "isaac"), default="mujoco")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, help="Episodes per motion shard, up to the upstream ID stride of 1000")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without writing or running")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run):
        parser.error("--run must start with a letter or digit and contain letters, digits, dots, underscores or hyphens")
    if not 0 <= args.shard < args.num_shards:
        parser.error("require 0 <= --shard < --num-shards")
    profile = json.loads(args.profile.read_text())
    if args.episodes is not None:
        profile["episodes_per_shard"] = args.episodes
    count = profile["episodes_per_shard"]
    if not isinstance(count, int) or not 1 <= count <= 1000:
        parser.error("episodes_per_shard must be 1..1000 because episode IDs use shard*1000+i")
    indices = profile["scene_indices"]
    if not indices or len(set(indices)) != len(indices):
        parser.error("scene_indices must contain distinct upstream scene IDs")
    original = json.loads((DATAGEN / "scene_gen/spec/scenes_single.json").read_text())
    by_index = {scene["idx"]: scene for scene in original["scenes"]}
    if any(index not in by_index for index in indices):
        parser.error("scene_indices contains an unknown upstream scene ID")
    scenes = [dict(by_index[index], shard=0) for index in indices]
    manifest = {key: original[key] for key in ("campaign", "version", "seed_base")}
    manifest.update(n_scenes=len(scenes), n_shards=1, shard_members=[indices], scenes=scenes)

    local = Path(os.environ.get("HALO_LOCAL_ROOT", ROOT / ".local")).expanduser().resolve()
    data = Path(os.environ.get("HALO_DATA_ROOT", local / "data")).expanduser().resolve()
    runs = Path(os.environ.get("HALO_RUNS_ROOT", local / "runs")).expanduser().resolve()
    run = runs / "datagen" / args.run
    prefix = f"experiments/{args.run}"
    scene_prefix = f"scenes/{args.run}/single"
    spec_dir = data / prefix / "specs" / f"shard_{args.shard:03d}"
    dataset = data / prefix / f"dataset_{args.backend}"
    python = os.environ.get("HALO_PYTHON", sys.executable)
    runtime = str(Path(python).parent)
    env = os.environ.copy()
    env.update(
        PATH=f"{runtime}:{DATAGEN / 'launch'}:{env.get('PATH', '')}",
        PYTHONPATH=f"{DATAGEN}:{env.get('PYTHONPATH', '')}",
        VIRTUAL_ENV=str(Path(python).parent.parent),
    )
    overrides = {
        "HALO_PYTHON": python,
        "HALO_S3_ROOT": str(data),
        "OUTPUT_DIR": str(run / f"{args.stage}_{args.backend}" / f"shard_{args.shard:03d}"),
        "SCENE_S3_PATH": scene_prefix,
        "SPEC_S3_PATH": f"{prefix}/specs",
        "SCENE_MANIFEST": str(run / "render_scenes.txt"),
        "SCENES_PER_SHARD": str(len(scenes)),
        "EPS_PER_SHARD": str(count),
        "RENDER_BATCH": args.run,
        "DIRECT_WRITE_DEST": f"{prefix}/dataset_{args.backend}",
        "SHARED_PREFIX_ROOT": str(data / "rendered_output" / args.run / args.backend),
        "RENDER_DATA_DIR": str(run / f"render_inputs_{args.backend}" / f"shard_{args.shard:03d}"),
        "TARGET_FPS": str(profile["target_fps"]),
        "PUSH_TO_S3": "1",
        "ROLLBACK_EPS": "0",
        "SHARD_IDX": str(args.shard),
        "SHARD_ID": str(args.shard),
        "TOTAL_SHARDS": str(args.num_shards),
    }
    stage_command = ["bash", "launch/run_stage.sh"]
    if args.stage == "scene":
        # Scene membership is independent of the motion/render episode shards.
        blender = shutil.which("blender")
        blender_dir = Path(blender).resolve().parent if blender else local / "vendor/blender52"
        overrides.update(
            MANIFEST=str(run / "scene_manifest.json"),
            OUT_S3=str(data / scene_prefix),
            GEN_WORKERS=str(profile["scene_workers"]),
            GEN_MAX_READY=str(profile["scene_ready_queue"]),
            GEN_MEM_FLOOR_GB=str(profile["scene_memory_reserve_gb"]),
            GEN_DISK_FLOOR_GB=str(profile["scene_disk_reserve_gb"]),
            BLENDER_DIR=env.get("BLENDER_DIR", str(blender_dir)),
            INFINIGEN_ROOT=env.get("INFINIGEN_ROOT", str(local / "vendor/infinigen")),
            INFINIGEN_ENV=env.get("INFINIGEN_ENV", str(local / "runtimes/infinigen")),
            SHARD_IDX="0", SHARD_ID="0", TOTAL_SHARDS="1",
            OUTPUT_DIR=str(run / "scene"),
        )
        command = stage_command + ["scene_gen", "single", "--no-setup"]
    elif args.stage == "motion":
        overrides.update(
            CHECKPOINT_DIR=str(local / "cache/kimodo/checkpoints"),
            TEXT_ENCODERS_DIR=str(local / "cache/kimodo/text_encoders"),
            NEWTON_CACHE_PATH=str(local / "cache/newton"),
            LOCAL_CACHE="true", TEXT_ENCODER_MODE="local",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        )
        command = [python, "motion_gen/generate_sit_traj.py",
                   "--n", str(count), "--shard-id", str(args.shard), "--device", "cuda",
                   "--seed", str(profile["seed"]), "--batch", str(profile["motion_batch"]),
                   "--steps", str(profile["motion_steps"]), "--tier", "industry",
                   "--n-max-retry", str(profile["motion_retries"]),
                   "--check-workers", str(profile["motion_check_workers"]),
                   "--cam-kind", "fisheye", "--robot-variant", "g1_29dof_thor",
                   "--vdr-level", "light", "--asset-color", "default", "--chair-set", "pair",
                   "--single-standing-kf", "standpose", "--turn-nf", "50",
                   "--skip-existing", "--out", str(spec_dir)]
    elif args.stage == "render":
        overrides["ISAACLAB_SH"] = env.get("ISAACLAB_SH") or str(ROOT / "scripts/datagen/isaac_python.sh")
        if args.backend == "isaac":
            overrides["ROBOT_THOR_URDF"] = env.get("ROBOT_THOR_URDF") or str(
                local / "cache/robots/g1_thor/g1_29dof_v2_thor.urdf"
            )
        command = stage_command + ["render", "mujoco" if args.backend == "mujoco" else "prod", "--no-setup"]
    else:
        command = [python, "-m", "render.stitch", "--mode", "stitch",
                   "--source-prefix", str(dataset), "--workdir", str(run / f"stitch_{args.backend}"),
                   "--expected-shards", str(args.num_shards * min(count, len(scenes)))]
    env.update(overrides)
    if args.stage in ("scene", "render"):
        # run_stage applies these again after sourcing the upstream defaults.
        command.extend(f"{key}={value}" for key, value in overrides.items())
    print(f"Working directory {DATAGEN}")
    print(f"Profile {args.profile.resolve()}")
    print(f"Dataset {dataset}")
    print(f"Selected scenes {[scene['name'] for scene in scenes]}")
    print(f"Episodes per shard {count}, shard {args.shard}/{args.num_shards}")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    run.mkdir(parents=True, exist_ok=True)
    prepare_run(run, {"profile": profile, "num_shards": args.num_shards}, manifest,
                [scene["name"] for scene in scenes])
    if args.stage == "motion":
        link_cached_models(env)
    if args.stage == "render":
        missing = [str(spec_dir)] if not list(spec_dir.glob("episode_*/episode_spec.json")) else []
        mesh = "scan.obj" if args.backend == "mujoco" else "scan.usdc"
        for scene in scenes:
            scene_dir = data / scene_prefix / scene["name"]
            for name in (f"asset/{mesh}", "scene_analysis.json"):
                if not (scene_dir / name).is_file():
                    missing.append(str(scene_dir / name))
        if missing:
            raise SystemExit("Required inputs are missing\n" + "\n".join(missing))
        if args.backend == "isaac" and not os.environ.get("ROBOT_THOR_URDF"):
            robot_assets = ROOT / "assets/robots/g1"
            urdf = robot_assets / "urdf/g1_29dof_v2_thor.urdf"
            cached_urdf = Path(env["ROBOT_THOR_URDF"])
            cached_urdf.parent.mkdir(parents=True, exist_ok=True)
            content = urdf.read_text().replace('filename="meshes/', f'filename="{robot_assets}/meshes/')
            temporary = cached_urdf.with_suffix(f".{os.getpid()}.urdf.tmp")
            temporary.write_text(content)
            temporary.replace(cached_urdf)
    subprocess.run(command, cwd=DATAGEN, env=env, check=True)


if __name__ == "__main__":
    main()
