#!/usr/bin/env python3
"""Inspect the target contract and prepare HALO native Thor build artifacts."""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "configs/hardware/g1_thor.json"


def read_text(path):
    try:
        return Path(path).read_text().replace("\x00", " ").strip()
    except OSError:
        return None


def command_output(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "error": str(error)}


def identity():
    model = read_text("/proc/device-tree/model")
    compatible = read_text("/proc/device-tree/compatible")
    description = f"{model or ''} {compatible or ''}".lower()
    architecture = platform.machine()
    return {
        "architecture": architecture,
        "product_model": model,
        "device_tree_compatible": compatible,
        "tegra_release": read_text("/etc/nv_tegra_release"),
        "is_thor": architecture in ("aarch64", "arm64") and any(
            word in description for word in ("thor", "t5000", "tegra264")
        ),
    }


def resolve(path):
    return (ROOT / Path(os.path.expandvars(path)).expanduser()).resolve()


def run_commands(commands, dry_run):
    for command in commands:
        print(shlex.join(str(part) for part in command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True, cwd=ROOT)


def require_thor():
    observed = identity()
    if not observed["is_thor"]:
        raise ValueError(
            "Native Thor preparation requires a Thor host with visible device-tree identity. "
            "Use --dry-run to inspect commands on the workstation. "
            f"Observed architecture={observed['architecture']}, model={observed['product_model']!r}"
        )


def inventory(args, profile):
    snapshot = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_profile": profile["name"],
        "host": identity(),
        "kernel": platform.release(),
        "kernel_version": platform.version(),
        "python": sys.version,
        "memory": read_text("/proc/meminfo"),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "packages": {},
        "commands": {},
    }
    for package in ("torch", "transformers", "tensorrt", "depthai", "pyrealsense2"):
        try:
            snapshot["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            snapshot["packages"][package] = None
    probes = {
        "gpu": ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        "cuda": ["nvcc", "--version"],
        "cmake": ["cmake", "--version"],
        "protobuf": ["protoc", "--version"],
        "realsense": ["pkg-config", "--modversion", "realsense2"],
        "interfaces": ["ip", "-brief", "link"],
        "usb_devices": ["lsusb"],
    }
    for name, command in probes.items():
        snapshot["commands"][name] = command_output(command)
    output = json.dumps(snapshot, indent=2) + "\n"
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output)
    print(output, end="")


def build(args, profile):
    build_dir = resolve(profile["build_dir"])
    configure = [
        "cmake", "-S", ROOT / "system/cpp", "-B", build_dir, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release-RT", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DRT_DEPLOY_JETSON_PLATFORM=thor",
        f"-DRT_DEPLOY_CUDA_ARCH={profile['compute']['cuda_arch']}",
        "-DRT_DEPLOY_WITH_TRT=ON", "-DRT_DEPLOY_WITH_UNITREE=ON",
        "-DRT_DEPLOY_WITH_REALSENSE=ON", "-DRT_DEPLOY_WITH_MUJOCO=OFF",
        "-DRT_DEPLOY_WITH_ORT=OFF", "-DRT_DEPLOY_BUILD_TESTS=OFF",
    ]
    commands = [configure, [
        "cmake", "--build", build_dir, "--target",
        "motiontrack_real_zmq", "g1_state_monitor", "onnx_to_trt", "-j", str(args.jobs),
    ]]
    if not args.dry_run:
        require_thor()
    run_commands(commands, args.dry_run)


def export(args, profile):
    tool = resolve(profile["build_dir"]) / "tracker_build/onnx_to_trt"
    artifacts = profile["artifacts"]
    pairs = [("tracker_onnx", "tracker_plan"), ("depth_onnx", "depth_plan")]
    if not args.dry_run:
        require_thor()
    commands = []
    outputs = []
    for input_key, output_key in pairs:
        source, destination = resolve(artifacts[input_key]), resolve(artifacts[output_key])
        if not args.dry_run and not source.is_file():
            raise ValueError(f"Required ONNX file is missing: {source}")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        plan_output = destination if args.dry_run else temporary
        outputs.append((temporary, destination))
        command = [tool, "--onnx", source, "--plan", plan_output]
        if input_key == "depth_onnx":
            command.extend(["--weakly-typed", "--fp16"])
        commands.append(command)
    if not args.dry_run:
        if not tool.is_file():
            raise ValueError(f"Build the converter first: {tool}")
        for _, output_key in pairs:
            resolve(artifacts[output_key]).parent.mkdir(parents=True, exist_ok=True)
    try:
        run_commands(commands, args.dry_run)
        if not args.dry_run:
            for temporary, destination in outputs:
                temporary.replace(destination)
    finally:
        if not args.dry_run:
            for temporary, _ in outputs:
                temporary.unlink(missing_ok=True)
    if not args.dry_run:
        metadata = {
            "host": identity(),
            "cuda": command_output(["nvcc", "--version"]),
            "libraries": command_output(["dpkg-query", "-W", "libnvinfer*"]),
            "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "plans": [str(resolve(artifacts[key])) for _, key in pairs],
        }
        resolve(artifacts["tracker_plan"]).with_suffix(".build.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )


def check(args, profile):
    references = {
        "tracker_config": profile["robot"]["tracker_config"],
        "reference_robot": profile["robot"]["reference_model"],
        "depth_config": profile["sensors"]["reference_depth"]["config"],
    }
    for name in ("tracker_config", "reference_robot", "depth_config"):
        if not resolve(references[name]).is_file():
            raise ValueError(f"Profile source reference is missing: {references[name]}")
    required = dict(profile["artifacts"])
    required["tracker_binary"] = str(resolve(profile["build_dir"]) / "motiontrack_real_zmq")
    report = {
        "profile": profile["name"],
        "source_references_valid": True,
        "host": identity(),
        "artifacts": {name: {"path": str(resolve(path)), "exists": resolve(path).is_file()}
                      for name, path in required.items()},
        "pending_measurements": {
            "compute": {key: value for key, value in profile["compute"].items() if value is None},
            "robot": {key: value for key, value in profile["robot"].items() if value is None},
            "network": {key: value for key, value in profile["network"].items() if value is None},
            "rgb": {key: value for key, value in profile["sensors"]["reference_rgb"].items() if value is None},
            "depth": {key: value for key, value in profile["sensors"]["reference_depth"].items() if value is None},
        },
        "validation_scope": "source references and artifact presence; target timing and behavior require measurement",
    }
    report["artifact_preparation_complete"] = all(item["exists"] for item in report["artifacts"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["artifact_preparation_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory_parser = commands.add_parser("inventory", help="Read host and dependency information")
    inventory_parser.add_argument("--output", help="Optional JSON output path")
    build_parser = commands.add_parser("build", help="Build native Thor C++ binaries")
    build_parser.add_argument("--jobs", type=int, default=4)
    build_parser.add_argument("--dry-run", action="store_true")
    export_parser = commands.add_parser("export", help="Build Thor TensorRT plans from ONNX")
    export_parser.add_argument("--dry-run", action="store_true")
    commands.add_parser("check", help="Check source references and local deployment artifacts")
    args = parser.parse_args()
    if args.command == "build" and args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        profile = json.loads(args.profile.read_text())
        if profile.get("schema_version") != 1:
            raise ValueError("The hardware profile requires schema_version 1")
        handlers = {"inventory": inventory, "build": build, "export": export, "check": check}
        return handlers[args.command](args, profile) or 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
