HexZero 本地 C++ 搜索（ABI 2）

用途
用于桌面和旧网页的 HexZero 模型搜索，不替换训练后端。
选边、状态转换、BFS 判胜、叶子去重和回传在 C++ 执行；网络由 ONNX 推理。
保留 FP32 网络、FP64 树统计及原 PUCT 运算顺序，不启用量化或 fast-math。

构建
Linux x86-64：python scripts/build_native_search.py
Windows x86-64：python scripts/build_native_search.py --windows --compiler <MinGW-w64 g++ 路径>
仅要求 SSE2，不要求 AVX2/AVX512。Windows 静态链接 GCC/C++ 运行库。

加载
native_mcts.py 与动态库必须配套使用 ABI 2。
HEX_SEARCH_BACKEND=auto：自动加载，失败时回退 Python/Numba（默认）。
HEX_SEARCH_BACKEND=cpp：要求 C++ 可用，否则报错。
HEX_SEARCH_BACKEND=python：强制使用兼容搜索。

边界
同一棵树须串行访问；ABI 2 错误信息存于树内，避免 Windows DLL 线程析构问题。
节点按需增长，落子后仅保留对应子树；65536 为正常节点预算，131072 为硬上限。
节点预算不检测系统剩余内存。支持 1～11 路，更大棋盘自动回退 Python。
分析视图是只读快照；换根会使旧视图失效。
