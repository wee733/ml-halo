"""Inspect local dependencies and optionally exercise generator optimization on synthetic inputs."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def check_model(torch):
    sys.path.insert(0, str(ROOT / "generator"))
    from configs.config import load_config
    from models.motion_generator import MotionGenerator, model_config_from

    config = load_config(str(ROOT / "configs/training/generator_5090.yaml"))
    torch.manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model = MotionGenerator(model_config_from(config)).cuda().train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.training.learning_rate)
    rgb = torch.rand(config.data.batch_size, 3, *config.data.image_size, device="cuda")
    language = ["Sit on the chair."] * config.data.batch_size
    trajectory_shape = (config.data.batch_size, config.model.trajectory_length, config.model.output_dim)
    target = torch.randn(trajectory_shape, device="cuda")
    noise = torch.randn_like(target)
    timestep = torch.rand(config.data.batch_size, device="cuda")
    noisy = noise * (1 - timestep[:, None, None]) + target * timestep[:, None, None]
    velocity = model(rgb, language, noisy_trajectory=noisy, timestep=timestep)["velocity"]
    loss = torch.nn.functional.mse_loss(velocity, target - noise)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.training.max_grad_norm)
    if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("Generator loss or gradients are invalid.")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    prefix = target[:, :config.model.prev_chunk_frames]
    prediction = model.predict(rgb, language, prev_chunk=prefix)
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("Generator inference returned invalid trajectories.")
    if not torch.equal(prediction[:, :config.model.prev_chunk_frames], prefix):
        raise RuntimeError("Generator inference changed the supplied inpainting prefix.")
    torch.cuda.synchronize()
    return {
        "inputs": "synthetic RGB and trajectories for dependency and gradient validation",
        "decoder_initialization": "random",
        "optimizer_steps": 1,
        "prediction_shape": list(prediction.shape),
        "prefix_preserved": True,
        "loss": loss.item(),
        "gradient_norm": grad_norm.item(),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - start,
        "task_policy_checkpoint_created": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="Load cached Qwen and run one synthetic optimization step and prediction.")
    args = parser.parse_args()
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_check_requested": args.model,
        "python": sys.version, "executable": sys.executable, "modules": {}, "errors": [],
    }
    for name in (
        "torch", "transformers", "mujoco", "google.protobuf", "av", "cv2",
        "pyarrow", "zmq", "qwen_vl_utils", "flatdict", "torch.utils.tensorboard",
        "halo_comm.msgs.state_pb2", "halo_comm.msgs.motion_pb2",
        "halo_comm.msgs.discovery_pb2",
    ):
        try:
            module = importlib.import_module(name)
            report["modules"][name] = str(getattr(module, "__version__", "imported"))
        except ImportError as error:
            report["errors"].append(f"{name}: {error}")
    torch = sys.modules.get("torch")
    report["cuda_available"] = torch is not None and torch.cuda.is_available()
    if report["cuda_available"]:
        report["gpu"] = torch.cuda.get_device_name()
    else:
        report["errors"].append("CUDA is unavailable")
    if args.model and not report["errors"]:
        try:
            report["generator"] = check_model(torch)
        except Exception as error:
            report["errors"].append(f"Generator validation failed: {type(error).__name__}: {error}")
    output = Path(os.environ.get("HALO_LOCAL_ROOT", ROOT / ".local")) / "validation/workstation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
