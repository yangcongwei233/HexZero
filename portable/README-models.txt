可添加模型的 models 文件夹

新模型请按下面的结构复制，子文件夹名称可自定；也支持直接放在 models 中。
models\iteration_880\trained.onnx
models\iteration_880\trained.onnx.json

两个文件必须同时存在、名称相互对应；扫描会递归查找子文件夹。
可将导出的 HexZero-880-teacher 模型包解压后，整个子文件夹放到这里。
不同模型建议放在不同子文件夹，避免覆盖旧版本。

网页“你的训练模型”中可选择模型，空闲时每 5 秒自动扫描，也可点击“刷新模型列表”。
点击“切换并新开局”会先验证新模型实际推理，保存旧局棋谱，然后开始新局。
切换后不会沿用旧模型搜索树。选择和 CPU 线程分别记忆，重新启动 START_HEX.bat 会恢复。
如果上次选中的模型已移除或损坏，启动时会提示并改用目录中有效、轮次最高的模型。
新增模型复制不完整、校验失败、尺寸不兼容时，会在网页显示原因，其他有效模型仍可使用。

当前离线版支持 HexZero 11×11、canonical_version=1、FP32 ONNX。
PyTorch 的 .pth/.pth.tar/.pt 不能直接由这个轻量离线版加载，需要先在训练环境导出。
在项目的训练/开发环境（已安装 PyTorch、ONNX、ONNX Runtime）运行：
python scripts/export_play_onnx.py --checkpoint 路径/checkpoint_新轮次.pth.tar --output 路径/iteration_新轮次
将输出的 trained.onnx 与 trained.onnx.json 一起复制到 Windows 的 models 子文件夹即可。
转换会验证 ONNX 与原模型的策略、价值输出；Windows 端不需要安装 PyTorch。

azalea 子文件夹用于参考模型，不会列在训练模型选择菜单中，请保留。
