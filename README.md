# LOOM 双臂数据生成

两台 Panda 在同一桌面分别抓放红色和蓝色方块，支持多相机记录及独立的双方向校准轨迹。仓库只包含仿真环境、数据生成、导出审计和测试。

## Conda 环境

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate loom-prototype
python -m pytest -q
ruff check .
```

环境使用 Python 3.12、Isaac Sim 6.0.1、Isaac Lab 3、PyTorch 2.11 / CUDA 12.8。普通单元测试只运行 CPU，不启动 Isaac Sim。GPU 实例需具有 CUDA/Vulkan 图形驱动。

## 在 GPU 实例上验证与采集

```bash
conda activate loom-prototype
python scripts/prepare_assets.py
export OMNI_KIT_ACCEPT_EULA=YES

# 依赖与资产检查，不启动仿真
python -m loom_prototype.verify

# 双臂物理任务、校准、全部高清相机及导出检查；临时数据随后自动清理
python -m loom_prototype.verify --sim

# 小规模采集：一个抓放、一个校准、第一台相机，保留高清分辨率
python -m loom_prototype.collect --smoke --output data/smoke

# 完整采集与审计
python -m loom_prototype.collect --output data/prototype
python -m loom_prototype.audit --data-root data/prototype
```

`OMNI_KIT_ACCEPT_EULA=YES` 表示接受 NVIDIA EULA。机器人资产按 `assets/franka-manifest.json` 下载并校验；两台机器人共用资产，但具有独立 articulation、控制器、夹爪和物体。

唯一配置为 [configs/prototype.json](configs/prototype.json)：默认 **1280×720、12 台相机、144 个双臂抓放 episode、20 个双臂校准 episode**。宽高可调，必须是偶数。物理步长 1/120 秒，每 4 个物理步控制并记录一次；实际采样时刻来自仿真 tick。相机逐帧编码，不把整段高清视频积存在内存中。

一次采集只启动一个进程。中断后以相同命令恢复，已完成 episode 会验证并跳过；中间文件留在输出旁的 `.<输出目录名>.work/`。全部通过审计后才原子发布到输出目录。已存在的正式输出不会覆盖；更改配置时应使用新输出目录。

## 数据格式

遵循 `LOOM-data/docs/contract/export-contract.md`，本项目的坐标映射见 [docs/export.md](docs/export.md)。每个物理 episode 包含所有相机，不按相机复制 state：

```text
data/prototype/
├── episodes.jsonl
└── episodes/<episode_id>/
    ├── state/
    │   ├── left_eef.parquet
    │   ├── right_eef.parquet
    │   ├── left_gripper.parquet
    │   └── right_gripper.parquet
    └── rgb/
        ├── v00.mp4
        ├── v00.parquet
        └── ...
```

EEF 是固定世界系下的真实 TCP 绝对位姿，`float64[12]`，旋转按完整矩阵列优先存储；夹爪是实际开度 `float32`。所有流具有共同原点的 `int64 timestamp_ns`，视频为无音轨 H.264 MP4，逐帧 Parquet 索引保存采集时刻。数据不含 action、训练划分或模型字段。

审计检查四路 state、SO(3)、夹爪范围、共享时钟、指令区间、视频逐帧解码与分辨率，以及双臂任务成功和校准运动证据。报告写入 `reports/`；数据、报告及大型资产不提交 Git。旧 HDF5 数据不受删除操作影响，但新工具只读写以上格式。

当前重写只做 CPU 单元测试和导出兼容性验证；双臂动力学与高清渲染需在 GPU 实例上运行 `verify --sim` 后确认。
