HexZero local-play C++ search (ABI 2)

ABI 2 stores errors in each search tree and avoids thread-local C++ destructors.
Update native_mcts.py and the library together. Older libraries fall back to
Python search in auto mode instead of calling an incompatible error ABI.

Scope: HexZero trained-model search only. The original Azalea reference and
server self-play/training keep their existing search implementations.
The local TrainedPlayer selects this library automatically. If the library
cannot load, it logs the reason and uses the original Python/Numba search.
Set HEX_SEARCH_BACKEND=python to force the old implementation, or cpp to
require the C++ implementation and fail visibly if unavailable.

The model and ONNX inference remain FP32; tree values/priors remain FP64.
No quantization, reduced search budget, fast-math, FMA contraction, heuristic
change, speculative search, or changed batching is involved.
Selection, virtual reservations, canonical board transitions, BFS terminal
checks, inference deduplication and path backup execute inside C++.
NumPy retains policy masking/normalization to preserve its reduction order.
The public tree view is for reading analysis; changing snapshot dictionaries
does not change the native search. A move/reset invalidates old node views.

Memory: only the chosen child subtree survives a move. Nodes grow on demand;
no 240 MB fixed allocation. A defensive 131072-node limit reports an error
instead of unbounded growth. Supported sizes: 1..11. Default automatic local
selection falls back for larger boards. Training does not opt into this ABI.
Each tree belongs to one search at a time, as in the existing local server.

Build Linux x86-64:
  python scripts/build_native_search.py
Build Windows x86-64 with MinGW-w64:
  python scripts/build_native_search.py --windows --compiler <g++ path>
Both use baseline x86-64 (SSE2), without requiring AVX2 or AVX512. Windows
uses MinGW's win32 thread model and links GCC/C++ runtimes statically;
no pthread runtime or compiler install is needed on
the player's PC. Build metadata includes source/binary SHA-256 and flags.
Windows startup performs a real 65-simulation smoke check and records the
selected backend in logs/environment-check.json. If a Windows DLL is missing
or cannot load, automatic mode retains the Python fallback.

Validation:
  python -m unittest hex.test_native_mcts
Differential tests compare every node/edge and the complete neural inference
input sequence, including noise, tree reuse, both seats, batch duplicates,
terminal positions, small budgets and recovery from failed inference.
Real ONNX timings and full-tree comparisons are recorded under
runs/local_play/cpp_search_rewrite_20261020/.
