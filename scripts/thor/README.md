# G1 + Thor 目标配置

RTX 5090 工作站承担数据处理、训练、优化、仿真评估和 ONNX 导出。Thor 承担机载感知、模型推理和 G1 控制。硬件与模型路径集中在 [g1_thor.json](../../configs/hardware/g1_thor.json)。

当前已检查源代码引用和构建命令。设备身份、JetPack、相机流、控制时延与任务效果需要在目标设备上测量，配置中的 `null` 保留这些待测值。

## 工作站可执行的检查

```bash
python3 scripts/thor/manage.py --help
python3 scripts/thor/manage.py build --dry-run
python3 scripts/thor/manage.py export --dry-run
python3 scripts/thor/manage.py check
```

`check` 验证 profile 引用的机器人、控制和深度配置，并列出模型与二进制的实际存在状态。模型或二进制缺失时返回 2。`artifact_preparation_complete` 仅表示文件齐备，时延和行为仍需单独实测。

## Thor 原生构建入口

在完成目标设备环境配置后运行。`inventory` 只读取当前机器，结果记录实际设备身份，缺失的命令保留错误信息。

```bash
python3 scripts/thor/manage.py inventory --output .local/deploy/g1-thor/inventory.json
python3 scripts/thor/manage.py build --jobs 4
```

构建复用 `system/cpp/CMakeLists.txt`，生成通信版 tracker、只读 `g1_state_monitor` 和 ONNX 转换器。使用 Thor CPU 目标和 CUDA 架构 110，与上游 `tracker/deploy_cpp/CMakeLists.txt` 的 Thor 设置对应。结果位于 `.local/build/thor/`。

依赖包括 C++20 编译器、CMake、Ninja、protobuf 编译器及开发库、ZeroMQ、TensorRT 开发库和插件、CUDA、Unitree SDK2、librealsense2。CMake 会通过上游 FetchContent 获取缺失的 Eigen 和 yaml-cpp。可以用 `UNITREE_SDK2_ROOT`、`TENSORRT_ROOT` 和 `CMAKE_PREFIX_PATH` 指定现有安装位置。

本入口仅构建应用层代码。系统依赖安装、驱动及实时内核配置依据目标 Thor 的现场清单处理。

## 模型交付

将工作站产物放入 profile 中的文件路径。`generator.yaml` 必须对应 `generator.pt` 的训练配置。跟踪器导出使用上游 `tracker/training/scripts/rsl_rl/play.py --export_only`，深度网络导出使用 `tracker/deploy_cpp/tools/export_las_onnx.py`。

深度 tracker 的 ONNX 输入为 `obs (1,434)`、`policy_temporal (1,30,166)` 和 `student_depth (1,1,48,84)`，输出控制 29 个关节。基线深度网络为 LiteAnyStereo LAS1，424×240 双目 IR 输入填充到 448×256，输出视差转换成米制深度。模型和预处理遵循 [上游深度说明](../../tracker/deploy_cpp/docs/depth.md)。

```bash
python3 scripts/thor/manage.py export
python3 scripts/thor/manage.py check
```

`export` 在目标 Thor 上生成 TensorRT 引擎，并记录该设备的 CUDA、TensorRT 包和设备身份。TensorRT 引擎与目标 GPU 和软件版本关联。使用 `.local/deploy/g1-thor/` 保存权重、导出文件、配置和本地清单。

## 相机与激光雷达

| 设备 | HALO 中的用途 | 接入状态 |
|---|---|---|
| OAK-1W | 头部 RGB，输入动作生成器 | 上游参考采集路径，需记录安装位置与标定 |
| D435i | 躯干近场深度，输入 tracker | 上游参考路径，双目 IR 经 LiteAnyStereo 处理 |
| D455 | RGB/深度候选 | 需实测支持的流模式，并匹配内外参和深度预处理 |
| 奥比中光 | RGB/深度候选 | 需确认型号、SDK 采集接口和标定 |
| MID-360 | 建图、定位或独立障碍感知 | 扩展模块，HALO 原始策略观测没有点云输入 |
| XT32 | 建图、定位或独立障碍感知 | 扩展模块，HALO 原始策略观测没有点云输入 |

替换相机时需同步模拟相机与真实相机的视场、安装位姿、图像预处理和时间戳。Thor 安装后的附加载荷和重心通过 profile 记录，再用于机器人动力学配置。

完整 HALO 运行入口为 `system/halo_system/scripts/run_real.py` 与 `motiontrack_real_zmq`。单任务使用 PyTorch/CUDA 生成器。当前上游多任务编排器的本地模型使用 MLX，Thor 路径需增加 CUDA 后端适配。
