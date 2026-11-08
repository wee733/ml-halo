---
tags: [robotics, humanoid, HALO, G1, simulation, RTX-5090]
---

# HALO: Perceptive Humanoid Motion from Simulation

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple-aiml-research.github.io/ml-halo/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.06908-b31b1b.svg)](https://arxiv.org/abs/2611.06908)

![HALO author demonstrations in real homes](docs/assets/teaser.webp)

*上图为作者项目展示。此 fork 的本机配置与验证范围见下文。*

## 5090 工作站配置

本 fork 使用 RTX 5090 完成数据处理、训练和仿真评估，G1 + Thor 作为后续机载目标。
已有 CUDA Python 环境通过 `HALO_PYTHON` 选择，配置位于 `configs/`，入口位于 `scripts/`，
数据、依赖 overlay、模型缓存和运行产物位于 Git 忽略的 `.local/`。

- [本机安装、验证与训练入口](WORKSTATION.md)
- [本地坐椅数据生成](scripts/datagen/README.md)
- [G1 + Thor 目标配置](scripts/thor/README.md)

上游尚未提供训练完成的 HALO generator / tracker 权重和训练数据。Qwen 基础模型已可用于生成器计算验证，
具体任务策略需完成数据生成与训练，或取得作者对应权重。

## Upstream project

<!-- TODO(release): authors. -->

This repository contains the code for data generation, model training, and
deployment from the HALO paper.

<!-- TODO(release): confirm scope vs. the paper. No tracker weights ship; the
     proprio and depth tracker students are trained with tracker/training. -->

## Repository layout

| Folder | What it is | Docs |
|---|---|---|
| [`datagen/`](datagen/) | Synthetic data pipeline: Infinigen scenes, Kimodo motion, Isaac / MuJoCo rendering; training data for the generator (rendered episodes) and the tracker (motion clips) | [README](datagen/README.md) |
| [`generator/`](generator/) | Motion generator: training and open-loop evaluation | [README](generator/README.md) |
| [`tracker/`](tracker/) | G1 motion-tracking policy: Isaac Lab training, Python and C++ (real-time) deployment | [README](tracker/README.md) |
| [`orchestrator/`](orchestrator/) | `halo_orchestrator`: the orchestrator (language request + egocentric image → plan of subtasks) on a local model co-served with the generator | [README](orchestrator/README.md) |
| [`system/`](system/) | Closed-loop system: `run_sim.py` / `run_real.py` launchers, generator / tracker / orchestrator nodes, live web viewer; `system/comm` (`halo_comm` messaging) and `system/serve` (`halo_serve`, on-device model scheduling) | [README](system/README.md) |
| [`assets/`](assets/) | Unitree G1 robot description (MJCF / URDF / USD), scene objects and scenes | [robots](assets/robots/README.md), [objects](assets/objects/README.md) |

## Data generation

To generate the training data, rendered episodes for the generator and motion clips for the tracker, see
[`datagen/README.md`](datagen/README.md).

## Training

- **Generator:** [`generator/README.md`](generator/README.md), which covers setup and
  single- / multi-node training.
- **Tracker:** [`tracker/training/README.md`](tracker/training/README.md), which
  covers Isaac Lab and the three training phases: privileged experts (`PreRL` tasks) → multi-teacher distillation
  → multi-critic fine-tuning (`PostRL` tasks).

## Deployment

The orchestrator runs a local model on the generator's `halo_serve` engine; see
[`orchestrator/README.md`](orchestrator/README.md#model).

- **Tracker on its own (MuJoCo sim-to-sim, real robot):** [`tracker/deploy/README.md`](tracker/deploy/README.md)
  (Python) and [`tracker/deploy_cpp/README.md`](tracker/deploy_cpp/README.md) (real-time C++)
- **Closed-loop in sim (MuJoCo):** [`system/README.md`](system/README.md#run-sim) (`run_sim.py`)
- **Closed-loop on the real robot (G1):** [`system/README.md`](system/README.md#run-real) (`run_real.py`)

## Citation

If you find this work useful, please cite:

```bibtex
@article{halo2026,
  title     = {Perceptive Humanoid Motion from Simulation},
  author    = {Hu, Siyuan and Abuduweili, Abulikemu and Meng, Yue and Liu, Shipeng and Leboutet, Quentin and
               Chen, Chen and M{\"u}ller, Matthias and Koltun, Vladlen},
  journal   = {arXiv preprint arXiv:2611.06908},
  url       = {https://arxiv.org/abs/2611.06908},
  year      = {2026}
}
```

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code.

<!-- TODO(release): license. Third-party assets (e.g. the Unitree G1 robot
     description) keep their original licenses. -->
