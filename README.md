# HexZero

为 2026 中国计算机博弈大赛准备的 Hex 训练与对弈程序，基于 alpha-zero-general，参考 Azalea 的批量搜索设计。

## 项目简介

- **训练**：策略/价值网络、自博弈、经验回放、并行评估和 TensorBoard，可从断点续训。
- **桌面对弈**：Tk 独立窗口，支持人人、人机、手动开局、前三候选分析、棋谱回放及比赛棋钟。
- **本地引擎**：C++ PUCT 搜索 + ONNX Runtime CPU 推理，保留 FP32，不使用量化。

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
  parallel_self_play.py # 并行自博弈
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

```bash
python -m pip install -r requirements.txt
python main_hex.py --board-size 11 --mcts-backend azalea \
  --num-iters 100 --num-eps 64 --num-mcts-sims 400 \
  --mcts-batch-size 32 --reuse-search-tree \
  --root-dirichlet-noise --dirichlet-alpha 0.03 \
  --checkpoint-dir temp/hex11 --tensorboard
```

以上是启动示例。自博弈可加 `--parallel-self-play --num-workers 12`，评估可加 `--arena-workers 12`；两阶段顺序运行，进程数按机器配置调整。GPU 自博弈需显式指定 `--selfplay-device cuda`。

续训和查看曲线：

```bash
python main_hex.py --board-size 11 --checkpoint-dir temp/hex11 --load-model --num-iters 200
tensorboard --logdir temp/hex11/tensorboard
```

`--num-iters` 是总轮数上限。默认每 10 轮及正常结束时保存，只保留最近 3 对新建模型/经验池快照；`latest.json` 指向最新完整快照。其他参数见 `python main_hex.py --help`。

教师数据为可选项：`--teacher-data teacher.npz --teacher-fraction 0.1` 表示约 10% 教师样本、90% 自博弈。数据采集和评估工具在 `scripts/`，不随桌面程序运行。

## 搜索与推理

- **树复用**：双方落子后保留对应子树，减少重复搜索。
- **批量评估**：虚拟访问分散叶节点，同批重复叶子只推理一次，所有路径仍回传。
- **本地加速**：C++ 完成选边、状态转换、判胜和回传；ONNX 开启图优化与 CPU 多线程。
- **预算控制**：按搜索次数、时间和节点上限停止；落子后释放无关子树。

C++ 后端用于本地对弈；训练使用 Python/Numba 批量搜索及多进程。多线程推理不等于多线程树搜索。

编译本地搜索库：`python scripts/build_native_search.py`。Windows 交叉编译加 `--windows --compiler 编译器路径`；库不可用时自动回退 Python/Numba。详见 [原生搜索说明](hex/native/README.txt)。

导出模型：

```bash
python scripts/export_play_onnx.py --checkpoint checkpoint_3250.pth.tar --output iteration_3250
```

## 规则与棋谱

红方先手连接上下，蓝方后手连接左右；A1 在左下，不使用 Pie 交换规则。
支持 GB2312 比赛 TXT 和 UTF-8 JSON。MARK 为 −2～+2；分析 Q 是当前执棋方估值，不是校准胜率。
随机种子用于并列走法选择，不能保证改变开局。源码运行棋谱保存在 `runs/desktop/games`，离线版保存在 `userdata/games`。

## 源码与许可

仓库仅提交训练、对弈源码及文档，不提交模型、训练数据、日志、测试文件和发布包。
旧网页入口 `hex_play_server.py` 与 `hex_web/` 保留兼容使用。

项目采用 [MIT](LICENSE)；Azalea 派生代码保留其 [Apache-2.0 许可](hex/azalea_cpu/LICENSE) 和 [修改说明](hex/azalea_cpu/NOTICE.txt)。
