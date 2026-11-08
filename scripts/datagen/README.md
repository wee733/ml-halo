# 5090 本地坐椅数据流程

入口复用仓库根目录的 `scripts/env.sh`。设置 `HALO_PYTHON` 可选择已有 CUDA Python。数据默认进入 `.local/data`，工作目录进入 `.local/runs`，模型索引与计算缓存进入 `.local/cache`。Hugging Face 权重继续复用已有的 `HF_HOME` 缓存。

默认配置为一个上游程序生成房间、每个 motion shard 四段轨迹。`motion_batch=2` 控制每次 best-of-B 的候选数量，`motion_check_workers=1` 控制检查进程数，扩散步数保持 200。上游批量配置使用 16 个候选及 16 个检查进程。这里的较小配置尚未经过完整生成成功率评估。

## 命令

先安装公共工作站依赖、场景依赖和 Kimodo。场景安装复用 PATH 中的 Blender 5.2，准备 Infinigen 所需的 Python 3.11 运行时。Kimodo 安装使用项目本地 overlay，加上 `--download-models` 后先检查四个模型仓库的访问权，再下载权重。

```bash
bash scripts/setup_workstation.sh
bash scripts/datagen/setup_scene.sh
bash scripts/datagen/setup_kimodo.sh
bash scripts/datagen/setup_kimodo.sh --download-models
```

每个阶段均支持 `--dry-run`。它只打印配置与命令。以下顺序完成场景、轨迹、视觉数据和元数据合并。

```bash
bash scripts/datagen/run_local.sh scene  --run sit-local-001 --dry-run
bash scripts/datagen/run_local.sh motion --run sit-local-001 --dry-run
bash scripts/datagen/run_local.sh render --run sit-local-001 --backend mujoco --dry-run
bash scripts/datagen/run_local.sh stitch --run sit-local-001 --backend mujoco --dry-run
```

依赖与模型就绪后，删除相应命令的 `--dry-run` 即可执行。重复使用同一 `--run` 会恢复同一批次。新的参数配置使用新批次名称。配置可复制到 `.local/` 后通过 `--profile` 指定。

`--episodes` 表示每个 motion shard 的目标 episode 数。`--shard I --num-shards N` 对应 N 个独立的运动与渲染分片，每个分片都需单独执行。上游 episode ID 为 `I * 1000 + local_index`，因此单个 shard 支持 1–1000 段。场景生成入口始终生成配置中选出的场景集合，使用自己的单分片清单。stitch 在所有渲染分片完成后执行。

生成轨迹包含 `episode_spec.json` 和 `g1_qpos.npy`。渲染数据写入 `.local/data/experiments/<run>/dataset_mujoco` 或 `dataset_isaac`，两种渲染器分别保存。stitch 复用上游校验，核对 parquet、FPV/TPV 视频与元数据的 episode 集合。

## 模型与场景依赖

HALO 的坐椅生成器固定加载 `Kimodo-G1-RP-v1`，文本编码器依赖 LLM2Vec 和 `Meta-Llama-3-8B-Instruct`。2026-11-08 的文件访问检查中，Kimodo 与两个 McGill 仓库可读，Llama 返回 HTTP 403，需要具有该模型访问权的 Hugging Face 账号凭据。配置沿用上游这一模型组合。

Kimodo 固定为上游安装脚本指定的 `58e7818`。本机使用 Python 3.12、PyTorch 2.11/CUDA 13 和 Transformers 5.10.4。`setup_kimodo.sh` 调用 `check_kimodo.py`，验证 G1 骨架和转换器初始化，并使用小型随机 Llama 验证 eager/SDPA 前向、LLM2Vec pooling 和 PEFT adapter 的前向、合并、重新加载。Kimodo 声明的 Transformers 版本为 5.1.0，CPU 前向还与该版本作了对照。完整权重加载与实际文本生成运动仍需在模型访问就绪后验证。

注意力诊断发现，在上游 LlamaBiModel 接收二维 `attention_mask` 的路径中，改变未来 token 对首 token 的输出影响为零。上游指定的 Transformers 5.1.0 与本机 5.10.4 均表现如此。配置保留上游实现，诊断脚本同时记录这一观察。

场景链依赖 Blender 5.2 和上游固定版本的 Infinigen。本机复用 Blender 5.2.2，上游安装脚本指定 5.2.0。Infinigen 固定为上游版本 `892947be`，其 bpy 4.2 与 NumPy 1.26.4 使用 `.local/runtimes/infinigen` 下的 Python 3.11。入口从以下环境变量定位运行时，可写入 `.local/settings.env`。

```bash
export BLENDER_DIR=/absolute/path/to/blender52
export INFINIGEN_ROOT=/absolute/path/to/infinigen
export INFINIGEN_ENV=/absolute/path/to/infinigen-python-environment
# Optional override for another already-configured Isaac runtime.
export ISAACLAB_SH=/absolute/path/to/IsaacLab/isaaclab.sh
```

Blender 默认从 PATH 解析真实安装路径。Infinigen 的默认路径为 `.local/vendor/infinigen` 和 `.local/runtimes/infinigen`。上游场景安装器位于 `datagen/scene_gen/env/setup_scene_gen.sh`，会调用 apt 和 conda。本地 `setup_scene.sh` 使用 uv，并保存环境清单。执行阶段统一传入 `--no-setup`，安装由明确的 setup 命令触发。

本机内存约 30 GiB。上游多进程场景生成器默认等待至少 60 GB 空闲内存。本地单 worker 配置将该等待阈值设为 0，并限制待处理场景队列为 1。具体房间的峰值内存需要实际运行测量，相关资源参数可以在 profile 中调整。

默认 `--backend mujoco` 使用上游 MuJoCo 视觉渲染路径。选择 `--backend isaac` 时自动使用 `isaac_python.sh`，复用本机 Python 与 tracker 相同的 Isaac Lab 2.3.2 兼容 overlay。`ISAACLAB_SH` 可覆盖这一入口。两者都需要完成的房间资产。Isaac 渲染入口已验证 CLI 启动，完整房间批次仍需运行。HALO tracker 训练继续使用独立的训练入口。

Isaac 渲染启动时会在 `.local/cache/robots/g1_thor/` 准备带绝对 mesh 路径的 THOR URDF，复用仓库中的 mesh。`ROBOT_THOR_URDF` 可指定已有的已准备 URDF。
