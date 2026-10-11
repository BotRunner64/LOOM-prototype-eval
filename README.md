# LOOM prototype：相机视角的 context 适配

按仓库 `LOOM-dev … .html` 中的 **Prototype 实验**实现。固定单臂 Panda、夹爪、桌面、红色方块与绿色目标区；随机化方块初始位置，只改变外部相机的平移和朝向。

本次交付范围是环境、完整训练/验证/测试数据、策略及训练/评测代码；**不启动训练**。数据准备结果与验证记录见 `reports/`，详细协议见 [docs/protocol.md](docs/protocol.md)。

## 当前验收结果

完整数据位于 `data/prototype/`：164 个独立物理 episode，在多个相机下记录为 1,472 个 HDF5 文件，共 563,208 帧、4.94 GB（十进制）。8 个基础训练视角、4 个共同留出视角，训练/验证/测试布局及校准参考独立。

| 数据 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: |
| 任务轨迹文件 | 768 | 128 | 384 |
| 校准轨迹文件 | 64 | 32 | 96 |

全部文件通过哈希、图像时序、30 Hz 时钟、动作坐标、物理成功、校准双方向及划分检查。8 项单元测试及 lint 通过；broad/narrow 在真实数据上完成 FP32/BF16 三条件前向验收，参数更新次数为 0。记录动作的物理回放完成抓放；随机初始化模型完成两个视角、六个条件的短程冻结评测接口检查。详细结果见 [readiness.json](reports/readiness.json)、[数据审计](reports/data-audit.json)、[broad 前向检查](reports/dry-run-broad.json)、[narrow 前向检查](reports/dry-run-narrow.json) 和 [评测接口检查](reports/eval-interface-check.json)。后者只验证流程，不是学习策略成功率。

策略是约 227 万参数的小 CNN + Transformer flow-matching head，沿用参考项目的 relative pose、rotation6D、30 Hz/16 步 chunk 和未来绝对夹爪约定。context dropout 同时监督有/无参考；六份配置覆盖 broad/narrow × 三个种子，训练预算一致，包含验证、断点恢复、冻结配对评测及置信区间统计。

## 环境

当前机器已经配置 `conda activate loom-prototype`（Python 3.12，Isaac Sim 6.0.1，Isaac Lab 3，PyTorch 2.11 CUDA 12.8），使用 RTX 5090。

在新机器上从仓库根目录运行：

```bash
conda env create -f environment.yml
conda activate loom-prototype
python scripts/prepare_assets.py
export OMNI_KIT_ACCEPT_EULA=YES
```

Isaac Sim 安装要求和 EULA 见 [NVIDIA 官方说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html)。机器人文件位于本仓库 `assets/franka/`，版本与 SHA256 见 `assets/franka-manifest.json`。场景其余元素由代码生成，无需外部 LOOM-env 仓库。当前环境通过克隆本机 `loom-env` 创建，随后移除了其 editable 包；完整软件清单见 `reports/environment-*.txt`。

## 数据准备与验证（不训练）

查看已有数据（不需要启动 Isaac Sim，也不会训练或修改数据）：

```bash
conda activate loom-prototype
python -m loom_prototype.viewer --data-root data/prototype --port 8765
```

浏览器打开 <http://127.0.0.1:8765>。可选择 train/val/test、任务或校准 episode，
播放、逐帧查看、拖动时间轴，并同步对比多相机 RGB；下方显示末端 XYZ、夹爪曲线和原始 state/action。
v08–v11 留出视角只在 test 中提供。viewer 只读取已有 HDF5，保留原始 30 Hz 时间轴和等待帧。

```bash
# 重新生成；已完成的 episode 会跳过，专家失败立即停止
python -m loom_prototype.collect --output data/prototype
python -m loom_prototype.audit --data-root data/prototype

# 单元测试及真实数据前向检查；不会调用 backward/optimizer.step
python -m pytest -q
# 用实际示范检查动作恢复与物理执行，不训练模型
python scripts/replay_oracle.py --report reports/oracle-raw.json
python -m loom_prototype.train --config configs/train_broad_17.json --dry-run
python -m loom_prototype.train --config configs/train_narrow_17.json --dry-run
```

采集需 CUDA/Vulkan 图形驱动；运行日志中的上游警告保留到 `logs/`。快速采集验证使用 `--smoke --output data/smoke`，不要用 smoke 数据训练。完整数据与大型机器人文件被 Git 忽略，代码、配置、资产校验清单及验收报告提交到 Git。

## 之后获得训练授权时

以下命令会更新参数，本次没有执行：

```bash
# 第一阶段：先训练 broad，再检验冻结模型的 context 收益
python -m loom_prototype.train --config configs/train_broad_17.json
# 断点恢复（配置及数据标识必须完全一致）
python -m loom_prototype.train --config configs/train_broad_17.json \
  --resume runs/broad/seed-17/checkpoint-010000.pt

python -m loom_prototype.evaluate \
  --checkpoint runs/broad/seed-17/checkpoint-020000.pt \
  --output runs/eval-broad-17.jsonl
python -m loom_prototype.statistics runs/eval-broad-17.jsonl \
  --output runs/summary-stage1.json
```

三个训练种子为 17、29、43，各自有 broad/narrow 配置。第一阶段在留出视角上出现可重复收益后，才开展第二阶段的 narrow 训练。第二阶段两组都评测共同留出的 `v08 v09 v10 v11`，每个训练种子对应相同测试布局、重复次数和参考：

```bash
python -m loom_prototype.train --config configs/train_narrow_17.json
python -m loom_prototype.evaluate \
  --checkpoint runs/narrow/seed-17/checkpoint-020000.pt \
  --views v08 v09 v10 v11 --output runs/eval-narrow-17.jsonl
# 对种子 29、43 重复后，将六份 eval JSONL 一起传入 statistics
python -m loom_prototype.statistics runs/eval-broad-17.jsonl runs/eval-narrow-17.jsonl \
  --output runs/summary-stage2-seed17.json
```

单个种子的 CI 只覆盖布局变化；多训练种子的配对 bootstrap 才覆盖初始化变化。尚无训练后闭环成功率，dry-run 的随机模型 loss 仅用于接口与数值验证。
