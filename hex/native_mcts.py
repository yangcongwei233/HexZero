"""本地对弈 C++ 搜索桥接：FP32 推理、FP64 树统计，训练不使用此后端。"""
import ctypes as ct
from pathlib import Path
import platform
import weakref

import numpy as np

from .azalea_mcts import AzaleaMCTS, arg_value

_LIB = None


def library():
    """加载 ABI 2 并声明指针签名，防止 ctypes 默认按 32 位整数截断地址。"""
    global _LIB
    if _LIB is not None:
        return _LIB
    if platform.machine().lower() not in ('amd64', 'x86_64'):
        raise OSError('C++ search requires x86-64; use the Python search fallback')
    name = {'Windows': 'hex_search.dll', 'Linux': 'libhex_search.so'}.get(platform.system())
    if name is None:
        raise OSError('C++ search is not built for this operating system')
    lib = ct.CDLL(str(Path(__file__).resolve().parent / 'native' / name))
    p, i, d = ct.c_void_p, ct.c_int, ct.c_double
    signatures = {
        'hx_abi': (i, []), 'hx_error': (ct.c_char_p, [p]),
        'hx_create': (p, [i]), 'hx_destroy': (None, [p]),
        'hx_reset': (i, [p, p]), 'hx_size': (i, [p]), 'hx_root_batch': (i, [p]),
        'hx_select': (i, [p, i, p, d]), 'hx_prepare': (i, [p, p, i]),
        'hx_commit': (i, [p, p, p, i]), 'hx_advance': (i, [p, i]),
        'hx_snapshot': (i, [p, i] + [p] * 8),
    }
    for name, (result, args) in signatures.items():
        func = getattr(lib, name)
        func.restype, func.argtypes = result, args
    if lib.hx_abi() != 2:
        raise OSError('C++ search library ABI mismatch')
    _LIB = lib
    return lib


def ptr(array):
    return ct.c_void_p(array.ctypes.data)


class NativeNode:
    """只读节点快照：换根使 generation 失效，搜索后按 epoch 刷新。"""
    def __init__(self, tree, index):
        self.tree, self.index, self.generation = weakref.proxy(tree), index, tree.generation
        self.epoch, self.data = -1, None

    def snapshot(self):
        tree = self.tree
        if tree.generation != self.generation:
            raise RuntimeError('Search tree changed; request a current node view')
        if self.epoch == tree.epoch:
            return self.data
        a, n = tree.actions, tree.game.n
        board = np.empty((n, n), np.int8)
        prior, totals = np.empty(a, np.float64), np.empty(a, np.float64)
        counts, children, order = [np.empty(a, np.int32) for _ in range(3)]
        info, edge_prior = np.empty(7, np.int32), np.empty(1, np.float64)
        tree.call('hx_snapshot', self.index, *map(ptr, (board, prior, totals, counts, children, order, info, edge_prior)))
        visited = [int(x) for x in order[:info[5]]]
        self.data = dict(board=board, P=prior if info[4] and info[3] == 0 else None,
                         valids=(board.ravel() == 0).astype(np.int8) if info[4] and info[3] == 0 else None,
                         terminal=None if info[3] == 2 else int(info[3]), N=int(info[2]),
                         Nsa={a: int(counts[a]) for a in visited},
                         Wsa={a: float(totals[a]) for a in visited},
                         Qsa={a: float(totals[a]) / int(counts[a]) for a in visited},
                         virtual_visits={}, child_indices=children,
                         parent_index=int(info[0]), action_from_parent=None if info[1] < 0 else int(info[1]),
                         prior=float(edge_prior[0]), expanded=bool(info[4]))
        assert info[6] == 0, 'Native virtual reservations leaked outside selection'
        for array in (board, prior, children, self.data['valids']):
            if array is not None:
                array.flags.writeable = False
        self.epoch = tree.epoch
        return self.data

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        data = self.snapshot()
        if name == 'children':
            return {a: self.tree.node(int(i)) for a, i in enumerate(data['child_indices']) if i >= 0}
        if name == 'parent':
            return self.tree.node(data['parent_index']) if data['parent_index'] >= 0 else None
        if name in data:
            return data[name]
        raise AttributeError(name)

    def child(self, action):
        action = int(action)
        if not 0 <= action < self.tree.actions:
            return None
        index = self.snapshot()['child_indices'][action]
        return self.tree.node(int(index)) if index >= 0 else None

    def is_expanded(self):
        return self.snapshot()['expanded']


class NativeTree:
    """持有 C++ 树并自动释放；落子保留对应子树，未搜索过的落点清空树。"""
    def __init__(self, game):
        self.game, self.actions = game, game.getActionSize()
        self.lib = library()
        self.handle = self.lib.hx_create(game.n)
        if not self.handle:
            raise RuntimeError(self.lib.hx_error(None).decode())
        self._finalizer = weakref.finalize(self, self.lib.hx_destroy, self.handle)
        self.generation, self.epoch, self.views = 0, 0, {}
        self._root = None

    def call(self, name, *args):
        result = getattr(self.lib, name)(self.handle, *args)
        if result < 0:
            raise RuntimeError(self.lib.hx_error(self.handle).decode('utf-8', 'replace'))
        return result

    def node(self, index):
        if index not in self.views:
            self.views[index] = NativeNode(self, index)
        return self.views[index]

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        if value is not None:
            raise ValueError('Use set_root to replace a native tree')
        self.call('hx_reset', None)
        self._invalidate(False)

    def _invalidate(self, has_root):
        self.generation += 1
        self.epoch += 1
        self.views.clear()
        self._root = self.node(0) if has_root else None

    def reset(self, canonical_board=None):
        board = self.game.getInitBoard() if canonical_board is None else canonical_board
        board = np.ascontiguousarray(board, dtype=np.int8)
        if board.shape != (self.game.n, self.game.n) or np.any((board < -1) | (board > 1)):
            raise ValueError('Invalid native search board')
        self.call('hx_reset', ptr(board))
        self._invalidate(True)

    def set_root(self, canonical_board):
        if self.root is None or not np.array_equal(self.root.board, canonical_board):
            self.reset(canonical_board)

    def update_with_move(self, action):
        success = bool(self.call('hx_advance', int(action)))
        self._invalidate(success)
        return success


class _NativeBatch:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class NativeMCTS(AzaleaMCTS):
    """C++ 执行树遍历与回传；搜索预算、根噪声及最终选择沿用父类。"""
    backend = 'cpp'

    def __init__(self, game, nnet, args):
        super().__init__(game, nnet, args)
        self.tree = NativeTree(game)
        self._boards = np.empty((max(1, int(arg_value(args, 'mctsBatchSize', 1))), game.n, game.n), np.int8)

    def _select_batch(self, batch_size):
        """在 C++ 中选择多条 PUCT 路径，暂记 virtual loss，减少批内重复选择。"""
        prior = np.ascontiguousarray(self._root_prior, dtype=np.float64)
        self.tree.call('hx_select', batch_size, ptr(prior), float(arg_value(self.args, 'cpuct', 1.)))
        self.tree.epoch += 1
        return _NativeBatch(batch_size)

    def _safe_batch_size(self, requested):
        """按 65536 节点预算缩小批次；C++ 硬上限为 131072，不检测系统剩余内存。"""
        # 每条路径至多新建一个节点，提前限制批量以保留可用搜索结果。
        return min(requested, max(0, 65536 - self.tree.call('hx_size')))

    def _evaluate_and_backup(self, leaves):
        """叶节点去重后批量推理，终局跳过网络；所有模拟路径仍逐条回传。"""
        if not isinstance(leaves, _NativeBatch):
            if len(leaves) != 1 or leaves[0][0] is not self.tree.root or leaves[0][1]:
                raise ValueError('Native search expects a selected batch or the unevaluated root')
            self.tree.call('hx_root_batch')
        count = len(leaves)
        if len(self._boards) < count:
            self._boards = np.empty((count, self.game.n, self.game.n), np.int8)
        try:
            n = self.tree.call('hx_prepare', ptr(self._boards), len(self._boards))
            boards = self._boards[:n]
            if n:
                policies, values = self.nnet.predict_batch(boards)
                policies = np.asarray(policies)
                values = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
                if policies.shape != (n, self.game.getActionSize()) or values.shape != (n,):
                    raise ValueError('Network output shape does not match selected leaves')
                # 保留 NumPy 逐行求和顺序，避免浮点分组改变搜索结果。
                policies = np.asarray([self._masked_policy(b, p) for b, p in zip(boards, policies)], dtype=np.float64)
            else:
                policies, values = np.empty((0, self.game.getActionSize())), np.empty(0)
            self.tree.call('hx_commit', ptr(policies), ptr(values), n)
        finally:
            self.tree.epoch += 1


def local_mcts(game, net, args):
    """旧包自动回退；设置 HEX_SEARCH_BACKEND=python 可强制使用兼容搜索。"""
    import logging
    import os
    backend = os.environ.get('HEX_SEARCH_BACKEND', 'auto').lower()
    if backend not in ('auto', 'cpp', 'python'):
        raise ValueError('HEX_SEARCH_BACKEND must be auto, cpp or python')
    if backend == 'python' or (backend == 'auto' and not 1 <= game.n <= 11):
        return AzaleaMCTS(game, net, args)
    try:
        return NativeMCTS(game, net, args)
    except (OSError, AttributeError):
        if backend == 'cpp':
            raise
        logging.getLogger(__name__).warning('C++ search unavailable; using Python search', exc_info=True)
        return AzaleaMCTS(game, net, args)
