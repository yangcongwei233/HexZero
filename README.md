# HexZero

为 2026 中国计算机博弈大赛准备的 Hex 训练与对弈程序，基于 alpha-zero-general，参考 Azalea 的批量搜索设计。

## 项目简介

- **训练**：策略/价值网络、自博弈、经验回放、并行评估和 TensorBoard，可从断点续训。
- **桌面对弈**：Tk 独立窗口，支持人人、人机、手动开局、前三候选分析、棋谱回放及比赛棋钟。
- **本地引擎**：C++ PUCT 搜索 + ONNX Runtime CPU 推理，支持搜索树复用、批量推理和预算控制。

## 程序特色

### 训练加速

- **批量搜索**：一次收集多个叶节点，使用虚拟访问分散搜索，再批量调用网络；重复叶节点只推理一次，每条搜索路径分别回传。
- **多进程自博弈与评估**：独立对局并行执行，可分别设置进程数和每进程推理线程数；评估进程复用已加载的模型。
- **搜索树复用与热点编译**：落子后保留对应子树；Numba 编译选边和判胜等热点循环，减少 Python 开销。
- **减少磁盘读写**：评估仅传递推理权重，训练回滚状态保存在内存；模型和经验池按间隔配对保存，并清理过期快照。

### 与 Azalea 的关系

训练流程基于 alpha-zero-general，批量 MCTS 参考 [Azalea](https://github.com/jseppanen/azalea)。残差策略/价值网络、根节点噪声和温度采样属于共同采用的 AlphaZero 方法。

| 方面 | Azalea | 本项目 |
| --- | --- | --- |
| 搜索 | 批量选叶、虚拟损失、叶节点去重、树复用，使用 Numba | 参考这些设计，适配本项目坐标和价值视角；重复叶节点共享推理结果，保留各路径回传 |
| 并行训练 | 已有多进程自博弈及评估 | 在 alpha-zero-general 流程中实现并行自博弈、平衡先后手评估和可复现的任务种子 |
| 网络输入 | 将棋子类别嵌入为 4 通道 | 己方、对方、常数共 3 通道；网络深度和各分支宽度可配置 |
| 教师数据 | 以自博弈学习为主 | 可混入 Azalea 的搜索策略分布和终局结果，混入比例可配置，其余继续自博弈 |
| 本地部署 | Python/Numba 搜索与 PyTorch 网络 | C++ 搜索、ONNX 推理、独立桌面窗口、比赛棋钟和标准棋谱 |

`hex/azalea_cpu/` 是用于参考模型评估和教师数据采集的 Azalea 派生实现，保留原许可；桌面版使用本项目模型和搜索引擎。

## 项目结构

```text
main_hex.py              # 训练入口
Coach.py                 # 自博弈、训练和评估循环
hex_desktop.py           # 桌面窗口
hex_play_core.py         # 对局、棋钟和棋谱
hex/
  desktop_service.py     # 桌面服务与模型管理
  native/search.cpp      # 本地 C++ 搜索
  native_mcts.py         # C++ 搜索接口
  onnx_inference.py      # ONNX 推理
  azalea_mcts.py         # 批量 MCTS
  pytorch/               # 网络与训练
  parallel_self_play.py  # 并行自博弈
  parallel_arena.py      # 并行评估
scripts/                 # 导出模型、评估及打包工具
portable/                # Windows 启动器和使用说明
```

## 桌面运行

需要 Python 3.11+ 和 Tcl/Tk。Windows 标准 Python 安装版包含 Tk。

```bash
python -m pip install -r requirements-desktop.txt
python hex_desktop.py --models models
```

Windows 也可运行 `START_DESKTOP.bat`。无模型时可直接人人对弈；人机对弈需将以下文件一起放入 `models`，再在“设置”中加载：

```text
models/iteration_轮次/trained.onnx
models/iteration_轮次/trained.onnx.json
```

桌面版不加载 Azalea 参考模型。默认每步 12800 次搜索、4 线程；搜索可选 800、1200、2000、3200、6400、12800、51200 次，推理线程可选 4～16。

| Windows 版本 | 模型 | 用途 |
| --- | --- | --- |
| 公开版 | 不附带 | 人人对弈；加入模型后启用人机和分析 |
| 比赛版 | 锁定指定 3250 权重 | 人人、人机对弈，禁止换模 |

离线包解压到新目录后运行 `START_HEX.bat`。`CHECK_ENV.bat` 检查环境，`START_HEX_PYTHON.bat` 使用兼容搜索。详见 [桌面使用说明](portable/README-Desktop.txt)。

## 训练

下面按已选定的 **3250 完整断点**及续训记录说明实际配置，而非脚本默认值。该模型出自同一条从零开始的训练：1～800 轮为纯自博弈，从第 801 轮起混入约 10% 教师样本；后续总目标延长到 5000 轮。目标轮数不代表已完成轮数。

### 实际参数与取舍

| 参数 | 实际选择 | 用途与取舍 |
| --- | --- | --- |
| 棋盘 `board-size` | 11 | 与比赛规则一致；直接在完整棋盘上学习，初期难度高于小棋盘 |
| 每轮自博弈 `num-eps` | 128 局 | 每轮补充足量新经验；增大可降低单轮样本波动，也会拉长模型更新间隔 |
| 每步搜索 `num-mcts-sims` | 800 | 在搜索标签质量和采集速度之间折中；与桌面对弈的搜索上限独立 |
| 搜索后端 `mcts-backend` | `azalea` | 使用本项目批量搜索实现，便于批量网络评估 |
| 搜索批量 `mcts-batch-size` | 32 | 合并网络调用；批次越大，同批路径越依赖尚未回传的统计，虚拟访问用于分散探索 |
| 探索系数 `cpuct` | 0.5 | 平衡已搜索价值与策略先验；增大会更偏向探索，不能单凭数值判断棋力 |
| 搜索树复用 `reuse-search-tree` | 开启 | 保留实际落子对应子树，节省重复计算；后续搜索会继承已有访问统计 |
| 温度阈值 `temp-threshold` | 16 | 前 15 手温度为 1，按访问分布采样；第 16 手起为 0，偏向最优访问动作，减少后段随机落子 |
| 根噪声 `root-dirichlet-noise` | 开启 | 自博弈扩大探索范围；评估关闭，避免用随机扰动衡量新旧模型 |
| 噪声形状 `dirichlet-alpha` | 0.03 | 小参数产生较集中的随机探索方向；过度集中也可能反复探索弱着 |
| 噪声占比 `dirichlet-fraction` | 0.25 | 根先验按 75% 网络策略与 25% 随机分布混合，兼顾已有知识和探索 |
| 自博弈并行 `parallel-self-play` / `num-workers` | 开启 / 12 | 并行生成独立棋局；继续增多会增加进程、模型副本及调度开销 |
| 任务粒度 `games-per-worker` | 每任务 1 局 | 长短棋局可动态分配，减少尾部等待；每任务仍需独立加载推理快照，不等于模型跨任务常驻 |
| 工作线程 `selfplay-threads` / `arena-threads` | 各 1 | 限制每进程的 CPU 数值计算线程，减少多进程争抢；不是整机只使用一个线程 |
| 判胜 `win-check-mode` | `numba` | 编译六邻接遍历，保留 BFS 回退；首次运行需要预热 |
| 经验池 `replay-buffer-size` | 100000 条 | FIFO 保留近期经验；容量越大越稳定，但旧策略数据滞留更久 |
| 单轮收集上限 `maxlen-of-queue` | 100000 条 | 限制单轮采集队列，避免异常局数造成无限增长 |
| 历史轮数 `history-iters` | 元数据为 20 | 本次已启用 FIFO，实际保留范围由 100000 条容量控制，不能再按 20 轮相乘 |
| 数据增强 | 原局面及旋转 180° | 每步生成两份合法对称样本；不使用会改变六邻接关系的单轴镜像 |
| 残差主干 `num-res-blocks` / `num_channels` | 6 块 / 64 通道 | 保留局部连接表达能力，同时控制推理开销；通道数为网络配置值 |
| 输入编码 | 己方、对方、常数共 3 通道 | 显式表达规范棋盘，后手先转置并换色，减少网络自行学习视角差异的负担 |
| 策略分支 `policy_channels` | 4 通道 | 压缩空间特征后输出全棋盘动作概率；更宽会增加参数与计算 |
| 价值分支 `value-channels` / `value_hidden` | 8 通道 / 128 | 给整局胜负估计更多表达容量；代价是比更窄价值头增加计算，不能保证价值误差更低 |
| 价值归一化 `disable_value_bn` | `false`，保留 BN | 使用批归一化稳定特征尺度；推理依赖训练累计的统计量 |
| 优化器 | Adam，β=(0.9, 0.999)，ε=1e-8 | 逐参数自适应步长，便于应对变化的自博弈数据；额外保存一、二阶矩状态 |
| 学习率 `learning-rate` | 0.001，固定 | 配置简单且续训连续；当前没有自动衰减，后期仍可能波动 |
| 权重衰减 | 0 | 沿用实际 Adam 配置，不额外施加 L2；也意味着没有这项显式正则约束 |
| 训练批量 `batch-size` | 128 | 平滑单批梯度并提高批量计算效率；与搜索批量 32 是不同参数 |
| 每轮训练 `epochs` | 3 | 每个 epoch 按经验池规模安排批次，但实际是有放回抽样，不保证逐条遍历；增加会提高复用率并可能过拟合旧数据 |
| 损失 | 策略交叉熵 + 价值均方误差，各权重 1 | 同时学习落子分布与终局结果；两项数值尺度不同，不能直接用大小判断谁学得更好 |
| 教师混入 `teacher-fraction` | 第 801 轮起约 0.1 | 每批约 10% 教师、90% 自博弈，补充参考策略又保留自主改进；教师集独立保存，不挤占 FIFO 容量 |
| 教师目标 | 原始搜索访问分布 + 终局胜负 | 学习搜索后的软策略和实际结果；价值标签不是教师网络的中间估值 |
| 新旧模型评估 `arena-compare` / `arena-workers` | 40 局 / 12 进程 | 双方各先手 20 局；并行缩短等待，但样本量仍有限 |
| 接受门槛 `update-threshold` | 0.55 | 无平局时至少 22/40 胜才接受候选；未通过则恢复旧权重和优化器，可减少退步但不保证对外棋力单调提高 |
| 人工先验 | 开局库关闭、战术优先关闭、`heuristic-prior-weight=0` | 正式训练不强制人工走法，避免启发式覆盖搜索目标；一步胜负仍由搜索中的规则判定 |
| 随机种子 `seed` | 3134582485 | 派生对局种子便于复现；正常续训恢复完整随机状态，而非每段重新从该种子开始 |
| 保存 `checkpoint-interval` / `keep-checkpoints` | 每 10 轮 / 最近 3 对 | 减少写盘与空间占用；正常结束也保存，意外中断可能丢失最近未保存的轮次 |
| 日志与诊断 | TensorBoard 开启；自博弈 JSONL 开启；完整策略 JSONL 关闭；额外价值调试关闭 | 保留曲线和落子记录，减少逐手大数组及额外诊断开销；训练目标仍保存在经验快照中 |

温度为零的自博弈阶段，当前实现保存的是选定动作的一热策略目标；前 15 手保存采样前的温度策略分布。教师样本则保留归一化的原始访问分布。满经验池时，每轮约执行 `3 × ceil(100000 / 128) = 2346` 次梯度更新；评估拒绝后日志步数继续累计，所以不能把日志步数直接理解为当前权重保留的更新次数。

### 启动与续训

下面复用实际算法参数启动纯自博弈阶段。目录为通用示例；设备选项按部署环境配置，这里不披露实际设备配置。

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

网络固定配置中的 `num_channels=64`、`policy_channels=4`、`value_hidden=128` 及 Adam 参数见 `hex/pytorch/NNet.py`；其余均在命令中显式指定或由表中说明。自博弈、网络更新和评估依次执行，12 个自博弈进程和 12 个评估进程并非同时运行。

第 800 轮完成且教师集校验通过后，继续 4200 轮可到目标 5000。教师文件需提前采集并验证，示例路径不代表仓库附带该数据：

```bash
python main_hex.py --board-size 11 --checkpoint-dir temp/hex11 \
  --load-model --num-iters 4200 \
  --teacher-data data/teacher.npz --teacher-fraction 0.1 --tensorboard --save-selfplay-jsonl
tensorboard --logdir temp/hex11/tensorboard
```

**`--num-iters` 是本次执行轮数。** 例如从完整 3250 断点继续到 5000，应填 `1750`；3250 断点记录的续训段为从 3001 开始执行 2000 轮，不能误读为总上限 2000。续训需要模型与经验池配对快照，`latest.json` 指向最新完整快照；普通 ONNX 导出只供推理。

正常续训继承断点中的网络、搜索、优化器和教师设置。不要使用 `--fresh-replay` 或重新指定种子，除非有意丢弃经验或更改随机过程；日志开关在新命令中显式启用。其他选项见 `python main_hex.py --help`。

## 搜索与推理

- **树复用**：双方落子后保留对应子树，减少重复搜索。
- **批量评估**：虚拟访问分散叶节点，同批重复叶子只推理一次，所有路径仍回传。
- **本地加速**：C++ 完成选边、状态转换、判胜和回传；ONNX 开启图优化与 CPU 多线程。
- **减少运行开销**：模型和推理会话常驻，连续数组保存搜索统计，避免逐节点 Python 对象操作；启动时预热搜索内核。
- **预算控制**：按搜索次数、时间和节点上限停止；落子后释放无关子树。

C++ 后端用于本地对弈；训练使用 Python/Numba 批量搜索及多进程。多线程推理不等于多线程树搜索。

编译本地搜索库：`python scripts/build_native_search.py`。Windows 交叉编译加 `--windows --compiler 编译器路径`；库不可用时自动回退 Python/Numba。详见 [原生搜索说明](hex/native/README.txt)。

导出模型：

```bash
python scripts/export_play_onnx.py --checkpoint checkpoint_3250.pth.tar --output iteration_3250
```

### 已有加速记录

以下为历史对照测试中的**耗时降低比例**。本地推理统计一次落子的完整搜索过程，包含网络推理，不含模型加载和预热。

| 优化 | 对照条件 | 耗时降低 |
| --- | --- | --- |
| 自博弈由 8 增至 12 个进程 | 同一固定网络、48 局、每步 800 次搜索，生成样本一致 | 约 15.4% |
| 搜索热点编译与状态缓存等优化 | 1180 模型、ONNX、4 线程、每步 1200 次搜索 | 约 43.5% |
| 在上述工程优化之后重写 C++ 搜索 | 2830 模型、ONNX、8 线程、每步 2000 次搜索 | 约 20.8% |

两组本地搜索测试分别覆盖 3 个和 4 个固定真实局面，预热后重复计时，先取各局面中位数，再求平均。每行只比较该次测试中的前后版本；不同测试不能直接叠加为总提速，也不代表所有机器和棋局的表现。原始日志不随源码发布。

## 规则与棋谱

红方先手连接上下，蓝方后手连接左右；A1 在左下，不使用 Pie 交换规则。
支持 GB2312 比赛 TXT 和 UTF-8 JSON。MARK 为 −2～+2；分析 Q 是当前执棋方估值，不是校准胜率。
随机种子用于并列走法选择，不能保证改变开局。源码运行棋谱保存在 `runs/desktop/games`，离线版保存在 `userdata/games`。

## 源码与许可

仓库仅提交训练、对弈源码及文档，不提交模型、训练数据、日志、测试文件和发布包。
旧网页入口 `hex_play_server.py` 与 `hex_web/` 保留兼容使用。

项目采用 [MIT](LICENSE)；Azalea 派生代码保留其 [Apache-2.0 许可](hex/azalea_cpu/LICENSE) 和 [修改说明](hex/azalea_cpu/NOTICE.txt)。
