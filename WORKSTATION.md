# HALO 5090 工作站

本机负责合成数据、小批量训练、优化和仿真评估。项目以坐椅任务作为公开数据流程入口，机器人目标为完整 29 关节 G1 + Thor。

## 环境与目录

本次使用已有 Isaac Lab Python 3.12 环境，PyTorch 2.11 / CUDA 13、Transformers 5.10.4 和 Isaac Sim 6.2。默认优先选择 `$HOME/Isaac/IsaacLab/.venv/bin/python`，也可通过 `HALO_PYTHON` 指定已有解释器。

```text
configs/training/       单卡训练配置
configs/datagen/        坐椅数据配置与 Kimodo 版本
configs/hardware/       G1 + Thor 目标描述
scripts/                本地安装、验证、训练入口
.local/python/          MuJoCo、flatdict、Kimodo 兼容依赖
.local/vendor/          固定版本外部源码
.local/runtimes/        Infinigen 专用 Python 3.11
.local/build/sim/       本机 C++ 仿真跟踪器
.local/data/            数据集及动作数据
.local/runs/            训练与生成过程输出
.local/validation/      本机验证结果
.local/environment/     安装前后版本快照
```

上游模块位于 `generator/`、`tracker/`、`datagen/` 和 `system/`。仓库 `origin` 指向个人 fork，`upstream` 指向 Apple 官方仓库。模型、外部源码、日志和数据留在本机。

## 安装与检查

在仓库根目录执行。安装脚本复用已有 CUDA 环境，并用当前已装版本约束新增依赖。MuJoCo 3.2.5 等兼容包由项目入口通过 `PYTHONPATH` 加载。protobuf 复用当前环境，已核对 TensorBoard 与 HALO 消息模块可以同时导入。

```bash
bash scripts/setup_workstation.sh --download-model
bash scripts/check_workstation.sh
HF_HUB_OFFLINE=1 bash scripts/check_workstation.sh --model
bash scripts/train_tracker.sh --list-tasks
```

`--download-model` 下载生成器的 Qwen3.5-2B 基础模型。`--model` 用合成输入执行一次梯度更新和完整 15 步预测，核对数值、输出形状和前缀保持，并将结果写入 `.local/validation/workstation.json`。

安装脚本也会准备 Kimodo、Infinigen 和本机 C++ 仿真跟踪器。Infinigen 的 bpy 4.2 / NumPy 1.26 需要 Python 3.11，因此使用项目内专用运行时。Blender 复用本机 5.2.2，上游参考版本为 5.2.0。C++ 开发依赖下载并解包到 `.local/vendor/sim`。

本机测得 batch 2、150 帧、36 维输出的生成器检查通过，可训练参数约 4932 万，CUDA 峰值分配约 5.08 GiB。此检查使用随机初始化的 DiT 和合成轨迹，任务效果需要训练后的策略评估。

自定义路径写入 `.local/settings.env`，该文件由所有工作站入口加载。也可直接设置环境变量。

```bash
export HALO_PYTHON=/absolute/path/to/existing/python
export HALO_DATA_ROOT=/absolute/path/to/halo-data
export HALO_RUNS_ROOT=/absolute/path/to/halo-runs
```

## 生成器训练

先依照[本地数据流程](scripts/datagen/README.md)生成并 stitch 数据，再运行训练。入口会检查 `data/`、`videos/` 和 `meta/`，在新运行目录保存展开后的配置。

```bash
bash scripts/train_generator.sh \
  --data .local/data/experiments/sit-local-001/dataset_mujoco \
  --run sit-001 --dry-run
```

数据准备好后移除 `--dry-run`。默认冻结 Qwen，训练 8 层 DiT，batch size 为 2，可通过 `--batch-size`、`--epochs` 调整。该配置用于单卡小批量实验，复现论文规模需要增加数据量和训练预算。

## 跟踪器训练

HALO 使用 Isaac Lab 2.x 的四元数和传感器接口。本项目加载固定 2.3.2 源码 overlay，补充当前 Isaac Sim 6.2 所需接口适配。安装位置为 `.local/vendor/IsaacLab-2.3.2`，对应补丁由 `tracker/training/scripts/prepare_isaaclab_overlay.py` 应用。

本机适配已通过下述执行检查，与论文 Isaac Sim 5.1 环境的动力学等价性仍需单独比较。

将 G1 动作 CSV 按上游格式转换为 NPZ，然后运行 PreRL 训练。

```bash
bash scripts/train_tracker.sh --convert-csv --help
bash scripts/train_tracker.sh \
  --motion_data_folder /absolute/path/to/processed/npz \
  --num_envs 256
```

默认任务为 `Halo-MotionTracking-Robot_Only-PreRL`。训练输出位于 `.local/runs/tracker/`。相机 student 还需多教师蒸馏及 PostRL，阶段配置与命令见[上游跟踪器说明](tracker/training/README.md)。

本机已使用真实 LAFAN1 G1 动作的 60 帧片段完成 PhysX 转换，得到 99 帧、29 关节和 30 个刚体的 NPZ。PreRL 正常入口完成 4 个环境、1 轮 PPO 更新，共 96 个 transition。该检查从随机策略开始，记录中保留了早停和失败，任务效果需要进一步训练与独立评估。

兼容适配包含新版导入器的网格路径、刚体层级和单刚体物体处理。独立核对了 G1 的 211 个碰撞形状，以及质量、惯量和关节参数。ProprioDepth 场景完成 1 个环境、10 个仿真步，输出有限的 `1×1×48×84` 策略深度观测，29 个身体网格参与自身遮挡，并记录到真实 PhysX 足端接触力。具体结果保存在 `.local/validation/tracker*.json`。

## 本机闭环仿真入口

```bash
bash scripts/setup_sim.sh
bash scripts/run_sim.sh --help
```

C++ `motiontrack_mujoco_zmq` 使用 MuJoCo 3.2.5 与 ONNX Runtime，产物位于 `.local/build/sim/`。取得训练好的 generator checkpoint、匹配的配置和 tracker ONNX 后，复制上游单任务 YAML 到 `.local/`，填写这些路径并运行。

```bash
bash scripts/run_sim.sh --task_config /absolute/path/to/sit.yaml --duration 30 --web
```

## 仍需准备的输入

- 作者训练好的 HALO generator / tracker 权重和原始训练集尚未公开，闭环任务效果需要自行训练或取得对应权重。
- 坐椅动作生成依赖 `Meta-Llama-3-8B-Instruct`。本机已有凭据在 2026-11-08 的访问检查返回 HTTP 403，需要具有该模型访问权的账号授权后下载。
- 程序化房间生成会产生训练场景，运行时无需预先重建自己的工作场所。真实场地的相机、动力学与任务效果需要在机载阶段测量。

设备配置见 [G1 + Thor](scripts/thor/README.md)。本轮验证范围为 5090 工作站，Thor 环境、相机流和实机控制留待设备阶段完成。
