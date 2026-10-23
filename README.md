# HexZero

基于 [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)，参考 [Azalea](https://github.com/jseppanen/azalea) 的批量搜索设计。
用于 Hex 训练与桌面对弈。

## 功能

- **训练**：残差策略/价值网络、自博弈、经验回放、断点续训和 TensorBoard，可选混入教师数据。
- **训练加速**：并行自博弈与评估、批量推理、虚拟访问、叶节点去重、搜索树复用、Numba 热点编译及间隔保存断点。
- **本地推理**：C++ PUCT 搜索、ONNX Runtime CPU 推理、连续数组存储与搜索预算控制。
- **桌面对弈**：Tk 独立窗口，支持人人、人机、手动开局、前三候选分析、棋谱回放和比赛棋钟。

## 桌面运行

需要 Python 3.11+ 和 Tcl/Tk。Windows 标准 Python 安装版包含 Tk。

```bash
python -m pip install -r requirements-desktop.txt
python hex_desktop.py --models models
```

Windows 可运行 `START_DESKTOP.bat`；离线包运行 `START_HEX.bat`。
无模型时可人人对弈。启用人机与分析，将模型和清单一起放入 `models`，在设置中加载：

```text
models/iteration_轮次/trained.onnx
models/iteration_轮次/trained.onnx.json
```

默认每步 12800 次搜索、4 线程；搜索可选 800、1200、2000、3200、6400、12800、51200 次，线程可选 4～16。
公开版不附带模型；比赛版固定使用 3250。详见 [桌面使用说明](portable/README-Desktop.txt)。

## 训练

以下为 3250 断点的实际配置。该次训练前 800 轮为纯自博弈，第 801 轮起混入约 10% 教师样本。

### 参数与取舍

| 参数 | 实际选择 | 取舍 |
| --- | --- | --- |
| 棋盘 `board-size` | 11 | 直接学习比赛棋盘 |
| 每轮对局 `num-eps` | 128 | 增加新样本，代价是更新间隔更长 |
| 每步搜索 `num-mcts-sims` | 800 | 平衡搜索标签质量与采集速度 |
| 后端 / 批量 `mcts-backend` / `mcts-batch-size` | `azalea` / 32 | 合并网络调用；大批次会延迟统计回传 |
| 探索 `cpuct` | 0.5 | 平衡已搜索价值与策略先验 |
| 树复用 `reuse-search-tree` | 开启 | 保留已搜索分支，减少重复计算 |
| 温度 `temp-threshold` | 16 | 前 15 手采样，之后选最高访问动作 |
| 根噪声 | α=0.03，混入 25% | 自博弈扩大探索，评估关闭 |
| 并行 `num-workers` / `games-per-worker` | 12 / 每任务 1 局 | 动态分配长短棋局；每任务加载一次模型 |
| 进程内线程 `selfplay-threads` / `arena-threads` | 各 1 | 减少多进程争抢计算线程 |
| 判胜 `win-check-mode` | `numba` | 编译六邻接遍历 |
| FIFO `replay-buffer-size` / 单轮队列 `maxlen-of-queue` | 各 100000 | 控制数据规模与陈旧程度 |
| 历史 `history-iters` | 20 | 使用 FIFO 时由容量控制保留范围 |
| 数据增强 | 原局面及旋转 180° | 增加合法对称样本 |
| 网络主干 | 6 个残差块，64 通道 | 平衡表达能力与推理计算量 |
| 输入 / 策略头 | 3 通道 / 4 通道 | 输入为己方、对方、常数平面 |
| 价值头 | 8 通道、128 隐藏单元，保留 BN | 增加价值表达容量和计算量 |
| 优化器 | Adam，β=(0.9, 0.999)，ε=1e-8 | 自适应步长，保存一、二阶矩 |
| 学习率 / 权重衰减 | 固定 0.001 / 0 | 无自动衰减和 L2 正则 |
| 训练批量 / `epochs` | 128 / 3 | 提高样本复用；每个 epoch 有放回抽样 |
| 损失 | 策略交叉熵 + 价值均方误差，各权重 1 | 同时学习搜索分布与终局结果 |
| 教师数据（可选） | 本次从第 801 轮起约 10% | 独立于 FIFO，其余样本继续自博弈 |
| 评估 `arena-compare` / `arena-workers` | 40 局 / 12 进程 | 双方各先手 20 局，并行缩短等待 |
| 接受门槛 `update-threshold` | 0.55 | 至少 22 胜才接受，否则恢复旧权重和优化器 |
| 人工先验 | 开局库、战术优先关闭，启发权重 0 | 由网络与搜索决定走法 |
| 随机种子 `seed` | 3134582485 | 派生对局种子，续训恢复随机状态 |
| 保存间隔 / 保留数量 | 每 10 轮 / 最近 3 对 | 减少写盘与占用，结束时也保存 |
| 日志 | TensorBoard、自博弈 JSONL 开启 | 完整策略 JSONL、额外价值调试关闭 |

网络固定参数见 [NNet.py](hex/pytorch/NNet.py)。满经验池每轮约 2346 次梯度更新。
自博弈前 15 手保存温度策略分布，后续保存一热目标；教师样本保留原始访问分布及终局胜负。

### 启动与续训

```bash
python -m pip install -r requirements.txt
python main_hex.py --board-size 11 --mcts-backend azalea \
  --num-iters 800 --num-eps 128 --num-mcts-sims 800 \
  --mcts-batch-size 32 --cpuct 0.5 --reuse-search-tree \
  --temp-threshold 16 --root-dirichlet-noise \
  --dirichlet-alpha 0.03 --dirichlet-fraction 0.25 \
  --parallel-self-play --num-workers 12 --games-per-worker 1 \
  --selfplay-threads 1 --arena-workers 12 --arena-threads 1 \
  --arena-compare 40 --update-threshold 0.55 --win-check-mode numba \
  --replay-buffer-size 100000 --maxlen-of-queue 100000 --history-iters 20 \
  --num-res-blocks 6 --value-channels 8 \
  --epochs 3 --batch-size 128 --learning-rate 0.001 \
  --no-use-opening-book --no-use-tactical-check --heuristic-prior-weight 0 \
  --seed 3134582485 --checkpoint-interval 10 --keep-checkpoints 3 \
  --checkpoint-dir temp/hex11 --tensorboard --save-selfplay-jsonl
```

续训继承断点设置；`--num-iters` 是本次执行轮数。从 800 轮继续到 5000：

```bash
python main_hex.py --board-size 11 --checkpoint-dir temp/hex11 \
  --load-model --num-iters 4200 --tensorboard --save-selfplay-jsonl
tensorboard --logdir temp/hex11/tensorboard
```

续训使用模型与经验池配对快照，`latest.json` 指向最新一对；ONNX 仅用于推理。
其他参数见 `python main_hex.py --help`。

### 可选教师数据

纯自博弈可独立完成训练。需要教师指导时，可用 `scripts/collect_azalea_teacher.py` 采集：

```bash
python scripts/collect_azalea_teacher.py --output data/azalea_teacher \
  --reference-source ../azalea \
  --reference-checkpoint ../azalea/models/hex11-20180712-3362.policy.pth \
  --exclude-openings data/eval_openings.json --games 512 --workers 4
```

`eval_openings.json` 是评测用双手开局列表，每项含 `actions`；采集时排除这些开局。
在训练或续训命令中追加 `--teacher-data data/azalea_teacher/teacher.npz --teacher-fraction 0.1`，每批混入约 10% 教师样本。
从含教师设置的断点恢复时，追加 `--teacher-fraction 0` 可关闭混入。

## 本地引擎

训练使用 Python/Numba 批量搜索；本地对弈使用 C++ 搜索和 ONNX 推理。两者均复用搜索树、批量评估叶节点。

```bash
python scripts/build_native_search.py
python scripts/export_play_onnx.py --checkpoint checkpoint_3250.pth.tar --output iteration_3250
```

Windows 交叉编译加 `--windows --compiler 编译器路径`；原生库不可用时回退 Python/Numba。
详见 [原生搜索说明](hex/native/README.txt)。

历史对照测试的耗时降低：

| 优化 | 耗时降低 |
| --- | --- |
| 自博弈由 8 增至 12 个进程 | 约 15.4% |
| 搜索热点编译、状态缓存等优化 | 约 43.5% |
| 在已有优化基础上重写 C++ 搜索 | 约 20.8% |

各项来自独立对照，不能叠加为总提速，具体表现取决于局面和运行环境。

## 规则与棋谱

红方先手连接上下，蓝方后手连接左右；A1 在左下，不使用 Pie 交换规则。
支持 GB2312 比赛 TXT 和 UTF-8 JSON。MARK 为 −2～+2；分析 Q 是当前执棋方估值。

## 项目结构

```text
main_hex.py          训练入口
Coach.py             自博弈、训练与评估
hex_desktop.py       桌面窗口
hex_play_core.py     对局、棋钟与棋谱
hex/pytorch/        策略/价值网络
hex/native/         C++ 搜索
scripts/            导出、评估与打包
portable/           Windows 启动器
```

## 许可

项目新增代码与修改采用 [Apache-2.0](LICENSE)。
alpha-zero-general 原有代码保留 [MIT](licenses/alpha-zero-general-MIT.txt)；Azalea 派生代码保留其 [许可](hex/azalea_cpu/LICENSE) 和 [修改说明](hex/azalea_cpu/NOTICE.txt)。详见 [NOTICE](NOTICE)。
