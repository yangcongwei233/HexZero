# HexZero：基于 AlphaZero 的海克斯棋博弈引擎

为2026中国计算机博弈大赛（导师的任务）准备的海克斯棋训练程序，参考了开源hex训练框架azalea和alphazero算法框架，codex老师负责结合优化

HexZero 是一个面向计算机博弈比赛的海克斯棋（Hex）AlphaZero 项目。项目基于 `alpha-zero-general` 改造，保留其清晰的 `Game / MCTS / Coach / NeuralNet` 训练框架，同时参考 `azalea` 的 Hex 专用工程化设计，加入更适合 Hex 自博弈训练的并行 self-play、批量 MCTS 评估、启发式先验、快速胜负判断、棋谱导出和 replay buffer 等能力。

当前重点不是通用棋类框架，而是训练和评估 Hex 专用智能体。

## 项目特点

- 支持 5x5、7x7、11x11 等不同棋盘尺寸。
- 基于 AlphaZero 的自博弈强化学习流程：self-play、神经网络训练、Arena 对战评估。
- 保留原始 classic MCTS 后端，可稳定回退。
- 新增 azalea-style MCTS 后端，支持批量叶节点评估、搜索树复用、virtual loss 和根节点 Dirichlet 噪声。
- 接入 Hex 规则先验，包括一步必胜、一步必防、简化双桥响应、边界形态和启发式 policy prior。
- 支持 numba 快速胜负判断；无 numba 时自动回退 BFS。
- 支持多进程并行 self-play，worker 独立加载 checkpoint，只生成训练样本，主进程负责训练。
- 支持离线模型评估：多个 checkpoint 两两循环赛，并输出相对 Elo 排名。
- 支持 JSONL 自博弈棋谱导出，便于后续分析、可视化和数据清洗。
- 支持 replay buffer，保存最近若干轮或固定容量的训练样本。

## 目录结构

核心训练文件：

```text
.
├── main_hex.py                  # Hex 训练入口
├── Coach.py                     # AlphaZero 训练循环
├── MCTS.py                      # classic MCTS 后端
├── Arena.py                     # 新旧模型对战评估
├── Game.py                      # 通用棋类接口
├── NeuralNet.py                 # 通用神经网络接口
└── utils.py                     # 工具类
```

Hex 专用实现：

```text
hex/
├── HexGame.py                   # Hex 对接 alpha-zero-general 的 Game 实现
├── HexLogic.py                  # Hex 棋盘逻辑、BFS/numba 胜负判断基础实现
├── fast_hex.py                  # 快速合法动作和胜负判断封装
├── azalea_mcts.py               # azalea-style 批量 MCTS 后端
├── parallel_self_play.py        # 多进程并行 self-play
├── replay_buffer.py             # 训练样本 replay buffer
├── hex_tactics.py               # Hex 战术规则先验
├── hex_heuristics.py            # Hex 启发式 policy prior
├── opening_book.py              # 开局库读取逻辑
├── opening_book.json            # 5x5 / 7x7 / 11x11 开局推荐
├── hex_action_selector.py       # 开局库、战术检查、MCTS 的动作选择封装
├── pytorch/
│   ├── HexNNet.py               # Hex 神经网络结构
│   └── NNet.py                  # PyTorch NNetWrapper，含 predict_batch
├── test_hex.py                  # Hex 基础规则测试
├── test_hex_priors.py           # 规则先验和 heuristic prior 测试
└── test_azalea_backend.py       # azalea 后端、batch MCTS、replay/jsonl 测试
```

辅助工具：

```text
visualize_hex.py                 # Hex 可视化脚本
visualize_hex_batch.py           # 批量可视化脚本
hex_play_server.py               # 人类/Azalea 对训练模型的本地测试台
play_local.sh                    # 本地对弈启动脚本
scripts/sync_play_model.py       # 从服务器导出最新已保存模型的推理权重
hex_eval.py                      # 离线 checkpoint 循环赛/Elo 评估
hex_web/                         # Web 对弈/展示前端
```

当前仓库已整理为 HexZero 专用结构，删除了 `alpha-zero-general` 原项目中与 Hex 无关的示例棋类、预训练模型和通用 Docker 示例。主要开发和训练入口是 `main_hex.py` 与 `hex/`。

## 环境准备

建议使用独立 Python 环境。当前开发环境示例：

```bash
git clone https://github.com/yangcongwei233/HexZero.git
cd HexZero

python -m pip install -r requirements.txt
```

主要依赖包括：

- Python 3
- numpy
- torch
- tqdm
- numba（可选，用于加速 Hex 胜负判断）
- coloredlogs（可选，用于更友好的日志输出）

如果没有安装 numba，项目会自动回退到 BFS 胜负判断。

## 快速测试

运行 Hex 相关单元测试：

```bash
cd HexZero

python -m unittest \
  hex.test_hex \
  hex.test_hex_priors \
  hex.test_azalea_backend \
  hex.test_regressions
```

检查 Python 文件是否可编译：

```bash
python -m compileall -q hex Coach.py main_hex.py
```

## 训练示例

### classic 后端

classic 是默认 MCTS 后端，行为接近原 `alpha-zero-general`。

```bash
cd HexZero

python main_hex.py \
  --board-size 5 \
  --num-iters 1 \
  --num-eps 4 \
  --num-mcts-sims 20 \
  --arena-compare 2 \
  --temp-threshold 5 \
  --heuristic-prior-weight 0.05
```

### azalea-style 后端

azalea 后端启用批量 MCTS evaluation、搜索树复用和 Hex 专用工程化流程。

```bash
cd HexZero

CUDA_VISIBLE_DEVICES=0 python main_hex.py \
  --board-size 5 \
  --num-iters 1 \
  --num-eps 4 \
  --num-mcts-sims 20 \
  --arena-compare 2 \
  --temp-threshold 5 \
  --heuristic-prior-weight 0.05 \
  --parallel-self-play \
  --num-workers 2 \
  --mcts-backend azalea \
  --mcts-batch-size 4 \
  --reuse-search-tree \
  --win-check-mode numba \
  --selfplay-device cpu
```

如果多进程 CUDA self-play 稳定，可以尝试：

```bash
--selfplay-device cuda
```

如果 CUDA 多进程不稳定，建议保持：

```bash
--selfplay-device cpu
```

主进程训练仍可使用 CUDA。

### 11x11 新架构 GPU 直接训练

当前默认网络使用 Hex 方向统一 canonical、`policy_channels=4`、`value_channels=8`，适合从头重新训练。建议写入新 checkpoint 目录，不要加载旧模型：

```bash
cd HexZero

CUDA_VISIBLE_DEVICES=0 python main_hex.py \
  --board-size 11 \
  --value-channels 8 \
  --checkpoint-dir ./temp/hex_11x11_azalea_canonical_v8/ \
  --num-iters 200 \
  --num-eps 200 \
  --num-mcts-sims 400 \
  --arena-compare 40 \
  --temp-threshold 12 \
  --heuristic-prior-weight 0.05 \
  --parallel-self-play \
  --num-workers 8 \
  --selfplay-device cuda \
  --mcts-backend azalea \
  --mcts-batch-size 32 \
  --reuse-search-tree \
  --win-check-mode numba
```

如果队友机器只有一张显卡，通常使用 `CUDA_VISIBLE_DEVICES=0`；如果要指定第二张显卡，可改成 `CUDA_VISIBLE_DEVICES=1`。这条命令不要加 `--load-model`。

价值网络通道数由 `--value-channels` 控制，默认推荐 `8`。如果显存或算力不够，可以改成：

```bash
--value-channels 2
```

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--board-size` | Hex 棋盘尺寸，例如 5、7、11 |
| `--value-channels` | value head 通道数，默认 8；算力不足可改成 2 |
| `--num-iters` | 训练迭代轮数 |
| `--num-eps` | 每轮 self-play 局数 |
| `--num-mcts-sims` | 每步 MCTS 模拟次数 |
| `--arena-compare` | 新旧模型 Arena 对战局数 |
| `--heuristic-prior-weight` | Hex 启发式先验混合权重，训练可从 0.05 开始 |
| `--mcts-backend classic\|azalea` | MCTS 后端，默认 classic |
| `--mcts-batch-size` | azalea 后端每批叶节点评估数量 |
| `--reuse-search-tree` | 执行动作后复用搜索树 |
| `--parallel-self-play` | 启用多进程 self-play |
| `--num-workers` | self-play worker 数量 |
| `--selfplay-device cpu\|cuda` | self-play worker 使用 CPU 或 CUDA |
| `--win-check-mode bfs\|fast\|numba` | 胜负判断模式，默认 numba |
| `--use-opening-book` | 启用开局库，默认关闭 |
| `--use-tactical-check` | 启用硬战术检查，默认关闭 |
| `--save-selfplay-jsonl` | 保存自博弈 JSONL 棋谱 |
| `--selfplay-output-dir` | JSONL 输出目录 |
| `--save-policy-in-jsonl` | JSONL 中保存完整 policy，文件会明显变大 |
| `--replay-buffer-size` | replay buffer 最大样本容量 |

训练时建议默认只开启较弱的启发式先验，例如：

```bash
--heuristic-prior-weight 0.05
```

`--use-opening-book` 和 `--use-tactical-check` 更适合评估、比赛或人机对局，不建议作为默认训练硬规则。

## 离线模型评估

训练过程仍使用 `Coach.py` 中的新旧模型 Arena 对战规则决定是否接受新模型。Arena 和离线循环赛都交换先后手，评估时关闭根节点探索噪声。后续模型强度评估可以单独使用 `hex_eval.py`，不影响训练流程。

示例：比较多个 11x11 checkpoint，并加入随机策略作为 0 Elo 参考点。

```bash
cd HexZero

python hex_eval.py \
  temp/hex_11x11/checkpoint_2.pth.tar \
  temp/hex_11x11/checkpoint_4.pth.tar \
  temp/hex_11x11/checkpoint_7.pth.tar \
  --board-size 11 \
  --include-random \
  --games-per-pair 40 \
  --num-mcts-sims 100 \
  --mcts-backend azalea \
  --mcts-batch-size 16 \
  --heuristic-prior-weight 0.05 \
  --output-json runs/eval_hex11/results.json
```

评估逻辑参考 azalea：

- 对所有模型做两两循环赛；
- 统计每组对局的 `[先列模型胜, 平局, 后列模型胜]`，对局交替交换先后手，局数必须为正偶数；
- 使用 Bradley-Terry/Elo 模型计算相对强度；
- 默认第一个参评对象为 0 Elo；如果加入 `--include-random`，随机策略作为 0 Elo 参考点。

## Azalea 教师样本（可选）

可以使用原版 Azalea 参考权重生成独立教师池，再以小比例混入已有自博弈训练，无须从零开始。示例采集 512 局：

```bash
python scripts/collect_azalea_teacher.py \
  --output runs/azalea_teacher_512 \
  --games 512 --workers 8 --simulations 800 --device cpu \
  --exclude-openings runs/local_play/diverse_openings_60.json
```

每局由同一个原版 Azalea 引擎自对弈，保留它的网络、MCTS、batch 与 cpuct。默认前 15 手按温度 1 采样，之后贪心落子；搜索启用 alpha=0.03、比例 0.25 的原版根节点噪声。策略标签始终使用**温度处理前的完整访问次数归一化分布**，价值标签是当前执棋方视角的最终胜负，不用落子 one-hot 或搜索 Q 替代。原版 1-based 原始合法走法映射为本项目 canonical 策略；后手棋盘取 `-board.T`，策略相应转置。

采样排除评测集的两手开局及其 180° 等价形式，保留双方胜负，并检查所有落子、独立 BFS 胜负与整局重复。空棋盘及单手局面仍是双方任务共有的状态。输出包括压缩 `teacher.npz`、带完整性哈希的 `manifest.json`、棋谱 `games.jsonl` 和进度文件；未完成的数据不会用于训练。

从完整模型/经验池断点继续时加入：

```bash
python main_hex.py --board-size 11 --load-model \
  --checkpoint-dir YOUR_CHECKPOINT_DIRECTORY \
  --teacher-data runs/azalea_teacher_512/teacher.npz --teacher-fraction 0.1 \
  --num-iters 100
```

教师数据独立于 FIFO 自博弈经验池。每批约 10% 教师、90% 自博弈，教师样本按实际先后手等量抽样，并随机应用合法的 180° 增强。训练批量、每轮优化器步数和原有并行自博弈/评估逻辑保持原设定。教师文件路径、SHA256 和混合比例写入断点配置并在续训时恢复；若文件被改动会拒绝加载。显式传 `--teacher-fraction 0` 可关闭教师指导。

TensorBoard 新增 `teacher/dataset_samples`、`teacher/target_fraction`、`teacher/batch_fraction`，以及 `iteration/teacher_fraction`、`iteration/teacher_samples`、`iteration/teacher_policy_loss`、`iteration/teacher_value_loss`、`iteration/selfplay_policy_loss`、`iteration/selfplay_value_loss`。两类 loss 与原有 iteration loss 同样汇总最后一个 epoch；实际混合比例和样本数统计本次训练调用的所有 epoch。加入不同来源的样本可能改变总 loss 的数值，棋力提升仍用独立评测判断。

`scripts/continue_hex_run.py` 可通过 `--resume-source-dir`、`--teacher-manifest`、`--teacher-fraction` 在当前训练段自然结束后切换到教师指导版本。它会等待采样完整发布并校验数据，保留切换前的完整模型/经验池快照，然后恢复优化器、RNG 和经验池继续训练；不会把未完成文件或评测棋谱直接塞入训练。

## 本地对弈测试

对手固定为指定的 **HexZero 训练模型**，己方可选人类或相邻 `azalea` 项目提供的参考模型。浏览器界面支持先后手、搜索次数、随机种子、单步/自动对战、暂停、悔棋、胜利路径和棋谱导出。比赛显示采用红方先手连接上下、蓝方后手连接左右，左下角 A1，底边 A～K、左边从下到上 1～11，不使用交换规则。仅显示颜色与外部坐标转换，模型内部 +1 先手上下、-1 后手左右和训练数据约定不变。

```bash
./play_local.sh --device cpu
# 浏览器打开 http://127.0.0.1:8765/
```

启动脚本优先使用 `HEX_PYTHON` 指定的解释器，否则使用本机已有的 `lab232` 环境或 `python3`。也可直接运行：

```bash
python hex_play_server.py \
  --checkpoint models/local_play/trained.pth.tar \
  --reference-source ../azalea \
  --reference-checkpoint ../azalea/models/hex11-20180712-3362.policy.pth \
  --device cpu --mcts-sims 800
```

棋盘大小和网络结构自动从训练断点恢复；旧权重缺少棋盘尺寸时须指定 `--board-size`。默认只监听本机回环地址。没有训练模型时会报错，不会用随机模型替代。参考模型缺失或尺寸不一致时，人机对弈仍可使用，Azalea 选项会禁用。Azalea 模式额外需要 `numba` 和原项目源码。

训练模型使用本项目批量 MCTS（batch 32、cpuct 0.5、树复用）；参考模型直接加载原 Azalea 网络与原 MCTS，保留参考断点的 batch 和 cpuct 等配置。双方均关闭根节点探索噪声。界面的搜索次数传给双方引擎，但两套代码对模拟和批次的计数不同，相同参数不代表实际计算量完全相等。它用于本地对弈测试，单局结果不作为正式棋力结论。

人机模式会自动走模型的一手；模型对战可点“下一手”或“自动对战”。暂停会在正在计算的一手结束后生效。人机悔棋撤回上一次人的落子及之后模型的回应；模型对战每次撤回一手。刷新页面可继续最近棋局；服务最多保留两局，重启会清空内存棋局。

点击“新建手动开局”即可替红、蓝双方交替落子，手数不限；“新开一局”则创建普通对局。点击“完成开局，开始对弈”后恢复人机对弈，或自动开始模型对战。开局阶段每次悔棋撤回一手；退回手动开局边界时会重新进入设置阶段。手动棋子会同步给两个模型，棋谱通过 `source: manual_opening` 区分人为设置与模型选择。

比赛打谱支持 GB2312 TXT 的 `{[HEX][先手队 R][后手队 B][先手胜或后手胜][时间地点][赛事];R(E,7)MARK[1];B(E,6)}` 格式，以及带详细评语的 UTF-8 JSON。可逐手跳转、分析当前局面、手工标注 −2～2、由当前分析生成落子方视角评语，或另建分支续弈。TXT 仅写标准整数 MARK；Q、候选主变化和详细文字保存在 JSON。未结束的棋局不自动伪造胜负，需先保存 JSON；声明赛果但未连通的节选/认输棋谱会保留声明并提示不能验证。实现见 `hex/record_format.py`。

“候选分析 · TOP 3”可选当前执棋模型、训练模型或 Azalea；人类回合默认由训练模型给建议。点击“分析当前局面”，显示最多三个合法候选，按 MCTS 访问次数排序，同次数按 Q 值、网络先验排序。Q 是当前执棋方视角的搜索平均价值，范围 −1～+1，不是校准胜率；零访问候选显示“未搜索”，网络先验也不是胜率。每条主变化继续选访问最多的分支，最多六手，到未搜索的叶节点或终局即停止。点击候选或续着坐标可预览，虚线棋子上的 +1、+2 表示未来手数；预览不会实际落子。手动开局/人类回合还可点击“落下”采用候选。

分析沿用本局搜索次数、双方原来的网络与 MCTS 实现，但使用独立搜索树和随机数状态，不给实战引擎追加搜索预算。结果在棋局不变时缓存，落子、悔棋或结束手动开局后失效。自动对战时可直接点击“暂停并分析”，正在计算的一手结束后会自动暂停并展示分析；点击“继续对弈”恢复。低搜索次数可能只产生很短的主变化。双方对模拟次数的计数差异也同样适用于分析。

完整棋局自动保存到 `runs/local_play/games/`，也可以随时导出。棋谱包含双方模型来源、搜索设置、种子、实际先后手、每手耗时和胜负；这些测试棋谱不会自动加入训练回放。

从训练服务器的最新**已保存断点**更新本机模型：

```bash
python scripts/sync_play_model.py \
  --server user@training-server --port 22 \
  --run-dir /path/to/training/run \
  --remote-python /path/to/python
# 停止并重新启动本地对弈服务后生效
```

可将本机默认连接设置写入不提交到 Git 的 `local_config/server.json`，字段为 `server`、`port`、`run_dir`、`remote_python`，之后可不带参数运行；命令行参数优先。脚本只读服务器断点，传输约 2.4 MiB 推理权重，不下载回放，不在服务器写入额外断点。来源轮次、校验值记录在 `trained.pth.tar.json`；服务器后续训练不会改变已加载的本地模型。

Windows 10/11 x64 离线版构建产物为 `dist/HexLocal-Windows-x64.zip`：解压后双击 `START_HEX.bat`。2026-10-21 发布包默认使用 3250 轮模型、12800 次搜索，包含 Azalea 参考模型。使用 ONNX Runtime 原生 CPU 推理与兼容版 C++ 搜索内核，保持 FP32，不使用量化；无需安装 Python、PyTorch、CUDA 或 C++ 编译器。网页模型卡片中可选 4～16 个 CPU 线程，应用后保留棋局，并保存设置供下次启动使用。发布包默认 4 线程。

Git 仓库保存源码、配置和测试；模型、训练日志、经验池、服务器备份与 Windows 安装包不随源码提交。首次克隆后需自行准备模型；预编译原生库也不入库，可按 `hex/native/README.txt` 构建。

每次启动先在独立进程检查运行库、架构、写权限和三个模型的真实推理；`CHECK_ENV.bat` 可单独检查，记录在 `logs/environment-check.json`。`SELF_TEST.bat` 进一步检查手动开局、双方分析及交接。包内保留原始权重和 Azalea 源码，并包含经过数值验证的 ONNX 模型。导出脚本为 `scripts/export_play_onnx.py`，打包脚本为 `scripts/build_windows_portable.py`，使用说明在 `portable/README-Windows.txt`。Windows 实机启动与本机 Linux 上的功能验证分开记录。

## 输出文件

默认 checkpoint 位于：

```text
temp/hex_尺寸x尺寸/
```

例如：

```text
temp/hex_11x11/
```

默认每 **10 轮**及正常训练结束时保存一次模型和样本对，只保留最近 **3 对新建快照**。不自动清理迁移前的旧文件。Arena 接受新模型后立即用于后续自博弈，磁盘保存仍按上述间隔进行；间隔内意外中断会回退到上一次快照。

常见训练输出包括：

- `*.pth.tar`：模型 checkpoint
- `*.examples`：与同编号模型配对的训练样本历史（包含棋盘与增强版本信息）
- `latest.json`：指向最新完整模型/样本对的清单
- `latest.pth.tar` / `best.pth.tar`：最近保存的已接受模型的硬链接，不额外复制权重
- `tensorboard/`：启用监控后的事件日志
- `selfplay_jsonl/`：自博弈 JSONL 棋谱
- 日志文件、缓存文件、可视化临时文件

这些文件通常不应上传到 GitHub。`.gitignore` 已默认忽略 checkpoint、训练日志、自博弈数据、大模型文件和缓存文件。

## JSONL 棋谱

启用：

```bash
--save-selfplay-jsonl --selfplay-output-dir ./selfplay_jsonl
```

每行是一局棋，默认不保存完整 policy，以免文件过大。每局至少包含：

- `board_size`
- `worker_id`
- `seed`
- `checkpoint`
- `mcts_backend`
- `mcts_batch_size`
- `winner`
- `moves`
- `final_board`
- `avg_search_time`
- `heuristic_prior_weight`

`policies` 与 `moves` 一一对应，每步 policy 使用当前行棋方的 canonical 坐标，不能直接当作原始棋盘坐标；增强后的训练样本不写入该字段。串行与并行 self-play 都支持导出。

如需保存完整 policy，可额外添加：

```bash
--save-policy-in-jsonl
```

## 修复后的搜索与训练行为

- 数据增强只使用原棋盘和旋转 180°，禁止破坏 Hex 邻接关系的单轴翻转。
- 批量 MCTS 使用边上的虚拟访问分散同批搜索，重复叶子只推理一次，但每条模拟路径都回传。新根的网络评估占一次模拟预算，因此新树配置 400 次模拟对应 399 次根边回传。
- 根噪声在根展开后采样一次，整次搜索共用；树中原始网络先验不被覆盖。评估不加探索噪声。
- 并行 worker 从 checkpoint 恢复网络结构，包括 value 通道和是否使用 value BN；每个网络实例拥有独立配置。
- 棋桥硬应手检查完整的两个己方端点和连接空位；一般边界评分仅用于软启发式，不再直接替代搜索。
- classic 和 azalea 都支持一次模拟时合法的先验回退。Arena 每局重置策略，azalea 在双方落子后均可推进搜索树。
- 并行 self-play 保留 spawn worker、独立模型与批量推理。worker 交接文件只包含推理状态，写入临时目录并自动清理，不累积成训练断点。
- 每局种子由主 seed、迭代号与任务/局号派生；worker 异步完成，但回放数据按任务顺序归并。

## 并行自博弈与对战评估

自博弈和评估可以分别配置进程数。22 核服务器上的短测支持两阶段各用 12 个 worker；两个阶段顺序执行，不会同时启动 24 个计算 worker：

```bash
--parallel-self-play --num-workers 12 --games-per-worker 1 \
--selfplay-device cuda --selfplay-threads 1 \
--arena-workers 12 --arena-device cuda --arena-threads 1
```

评估默认仍为串行（`--arena-workers 1`）。并行模式使用 spawn，每个 worker 加载两套固定模型，再逐局领取任务。先后手严格平衡，随机种子由训练 seed、迭代号、对局编号和模型身份确定，与 worker 数和完成顺序无关。每局搜索树独立，评估关闭根探索噪声；开局库、战术检查和搜索参数沿用训练配置。

每次评估只创建两份临时推理权重，worker 结束后自动删除，不产生额外训练断点。新增参数保存在运行配置中，续训自动继承，也可由命令行覆盖。TensorBoard 增加 `arena/workers` 和 `arena/games_per_second`，方便核对并发数和评估吞吐。

## TensorBoard 可视化与保存频率

安装 `requirements.txt` 后，在训练命令中加入：

```bash
--tensorboard \
--tensorboard-dir ./runs/hex11_fixed \
--tensorboard-interval 10 \
--checkpoint-interval 10 \
--keep-checkpoints 3
```

启动查看器：

```bash
tensorboard --logdir ./runs/hex11_fixed --host 127.0.0.1 --port 6006
```

在浏览器访问 `http://127.0.0.1:6006`。日志包括策略/价值/总损失、学习率、自博弈耗时与每秒局数、每轮样本数、回放容量与平均价值目标、新旧模型胜率、模型接受结果以及训练/评估耗时。小批次损失默认每 10 个优化步骤记录一次，其余指标每轮记录；日志频率与断点频率互相独立。

如果训练在远程服务器上运行，可按实际 SSH 用户、主机和端口在本机转发：

```bash
ssh -p 22 -N -L 6006:127.0.0.1:6006 user@training-server
```

## 续训与旧数据迁移

新 checkpoint 保存网络结构、优化器、优化步数、迭代号、随机状态及训练运行配置。拒绝候选模型时同时恢复旧权重和优化器；TensorBoard 步数保持递增。续训默认读取 `latest.json` 指向的完整快照对，并从下一轮开始；搜索、自博弈和保存配置会继承，命令行显式选项可以覆盖（布尔项也支持 `--no-parallel-self-play` 等关闭选项）。

```bash
python main_hex.py --board-size 11 \
  --checkpoint-dir ./temp/hex_11x11_fixed \
  --load-model --num-iters 20 \
  --tensorboard --tensorboard-dir ./runs/hex11_fixed
```

旧版无版本信息的 `.examples` 可能包含错误翻转样本，因此不再自动加载，也不能可靠地逐条修复。只迁移旧权重时，请显式指定旧文件、起始轮数和 `--fresh-replay`，重新生成训练数据；旧权重缺少的优化器状态无法恢复。网络结构变更时再使用 `--allow-partial-checkpoint`，普通续训不需要此选项。

```bash
python main_hex.py --board-size 11 \
  --checkpoint-dir ./temp/hex_11x11_fixed \
  --load-model --checkpoint-file /absolute/path/to/old_model.pth.tar \
  --start-iter 1 --fresh-replay --num-iters 20
```

修改 checkpoint 目录可保留旧实验。快照轮换仅清理本次保存清单登记的文件，不遍历删除其他训练断点。

## 项目来源和说明

本项目基于 `alpha-zero-general` 改造，核心训练循环仍沿用其易读、易扩展的设计；Hex 专用训练结构参考了 `azalea` 项目中的批量搜索、自博弈工程组织和 replay buffer 思路。

当前仓库用于个人研究、比赛训练和工程实验。建议只提交源码、配置、测试、开局库和文档，不提交模型权重、训练样本、日志和缓存。

Windows 离线版支持根目录 `models` 自动扫描：复制任意模型子目录中的 `trained.onnx` 与 `trained.onnx.json`，网页中选择并新开局，选择及线程数会保存。默认附带 750、800、880 轮和 Azalea。新增断点可在开发环境用 `python scripts/export_play_onnx.py --checkpoint checkpoint_新轮次.pth.tar --output iteration_新轮次` 转换，再复制两个输出文件；Windows 端无需 PyTorch。
