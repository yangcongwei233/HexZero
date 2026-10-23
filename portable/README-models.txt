models 文件夹说明

公开版默认没有任何权重，可直接人人对弈。
需要人机和分析功能时，请同时复制两个文件：
models/iteration_轮次/trained.onnx
models/iteration_轮次/trained.onnx.json

复制完成后在设置中刷新、选择并加载模型，随后可新开人机对弈。
目录空闲时约每 5 秒扫描；不完整、格式不兼容或哈希错误的模型不能加载。
支持 HexZero 11×11、canonical_version=1、FP32 ONNX，不支持直接加载 PyTorch 权重。
开发环境转换命令：
python scripts/export_play_onnx.py --checkpoint checkpoint_新轮次.pth.tar --output iteration_新轮次
Windows 用户不需要 PyTorch，只需复制导出的 ONNX 和 JSON。

比赛版仅附带和允许指定的 3250 模型，校验实际内容；不能通过重命名、改轮次或设置换模。
比赛版没有模型选择入口；若内置文件缺失或不符，启动会报错。
两版都不加载 Azalea 参考权重。
