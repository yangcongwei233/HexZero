# 派生自 Azalea（Apache-2.0），于 2026-10-15 适配 NumPy/ONNX 推理。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, astuple
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from numpy import int32, float32
from numpy.random import RandomState

from .typing import SearchableEnv, GameState

DEBUG = False

# 容量估算：每局 300 手 × 每手 1000 次评估 × 每次 30 个后续动作。
MAX_NODES = 10000000
INITIAL_NODES = 4096


class SearchTreeFull(Exception):
    pass


class SearchTree:
    """保存已探索的局面、动作边及 MCTS 统计。
    num_children 为 0 表示终局，为 −1 表示尚未评估。
    total_value 采用当前节点执棋方视角；prior_prob 采用父节点执棋方视角。"""

    def __init__(self):
        self.num_nodes = 0
        self.root_id: int32 = 0

        # 树的连接关系。
        self.parent = np.empty(INITIAL_NODES, dtype=int32)
        self.first_child = np.empty(INITIAL_NODES, dtype=int32)
        self.num_children = np.empty(INITIAL_NODES, dtype=int32)

        # MCTS 统计。
        self.num_visits = np.empty(INITIAL_NODES, dtype=float32)
        self.total_value = np.empty(INITIAL_NODES, dtype=float32)
        self.prior_prob = np.empty(INITIAL_NODES, dtype=float32)

        self.reset()

    def reserve(self, required):
        """按需扩容，保持节点编号和原容量上限不变。"""
        if required > MAX_NODES:
            raise SearchTreeFull('too many nodes')
        if required <= len(self.parent):
            return
        capacity = min(MAX_NODES, max(required, len(self.parent) * 2))
        arrays = {}
        for name in ('parent', 'first_child', 'num_children', 'num_visits', 'total_value', 'prior_prob'):
            old = getattr(self, name)
            grown = np.empty(capacity, dtype=old.dtype)
            grown[:self.num_nodes] = old[:self.num_nodes]
            arrays[name] = grown
        for name, array in arrays.items():
            setattr(self, name, array)

    def reset(self):
        """清空搜索树。"""
        self.num_nodes = 1
        self.root_id = 0

        # 创建尚未评估的根节点。
        self.parent[0] = -1
        self.first_child[0] = -1
        self.num_children[0] = -1
        self.num_visits[0] = 0
        self.total_value[0] = 0
        self.prior_prob[0] = 1.0

    def search(self, game: SearchableEnv, network: Network, *,
               num_simulations: int = 100,
               temperature: float = 1.0,
               exploration_coef: float = 1.0,
               exploration_noise_scale: float = 1.0,
               exploration_noise_alpha: float = 1.0,
               batch_size: int = 10,
               rng: Optional[RandomState] = None, time_limit=0) \
            -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """执行 AlphaZero 风格的蒙特卡洛树搜索。
        temperature 控制温度，exploration_noise_scale 控制探索噪声；
        rng 默认使用种子 0。返回动作概率、局面价值和调试指标。"""
        # 避免循环导入。
        from . import mcts

        if rng is None:
            rng = RandomState(0)

        network.eval()
        metrics = mcts.sample_paths(
            self, game, network,
            num_simulations, batch_size,
            exploration_coef,
            exploration_noise_scale,
            exploration_noise_alpha,
            rng, time_limit=time_limit)
        # 选择动作。
        stats = self.root.move_stats
        if np.any(stats.num_visits > 0):
            move_probs = as_distribution(stats.num_visits, temperature)
        else:
            # 存储上限可能在首个子节点模拟完成前触发。
            weights = stats.prior_prob.astype(np.float64)
            if temperature == 0:
                weights = (weights == weights.max()).astype(np.float64)
            move_probs = weights / weights.sum()
        value = self.root.total_value / max(1, self.root.num_visits)
        metrics['search_root_width'] = np.sum(stats.num_visits > 0)
        metrics['search_root_visits'] = np.mean(stats.num_visits)
        metrics['search_root_children'] = len(stats.num_visits)
        metrics['search_tree_nodes'] = self.num_nodes
        return move_probs, value, metrics

    def move(self, move_id: int) -> None:
        """执行 move_id 指定的动作，以对应子节点为新根，丢弃祖先和兄弟分支。"""
        if not self.root.is_evaluated():
            # 进入尚未探索的局面，无可复用子树。
            self.reset()
            return

        assert self.root_id < self.num_nodes, 'root node is unevaluated'
        node = self.root.child(move_id)
        if not node.is_evaluated():
            # 进入尚未探索的局面，无可复用子树。
            self.reset()
        else:
            self.root_id = node.id
            if self.num_nodes > 500000:
                self.compact()

    def compact(self):
        """只保留可达节点，保持统计和子节点顺序不变。"""
        levels = []
        level = np.array([self.root_id], dtype=np.int32)
        while len(level):
            levels.append(level)
            counts = self.num_children[level].clip(min=0)
            total = int(counts.sum())
            if not total:
                break
            starts = self.first_child[level]
            offsets = np.cumsum(counts, dtype=np.int64) - counts
            level = (np.repeat(starts - offsets, counts) + np.arange(total)).astype(np.int32)
        order = np.concatenate(levels)
        mapping = np.full(self.num_nodes, -1, dtype=np.int32)
        mapping[order] = np.arange(len(order), dtype=np.int32)
        capacity = max(INITIAL_NODES, len(order))
        child_counts = self.num_children[order].copy()
        for name in ('parent', 'first_child', 'num_children', 'num_visits', 'total_value', 'prior_prob'):
            old = getattr(self, name)
            kept = np.empty(capacity, dtype=old.dtype)
            kept[:len(order)] = old[order]
            if name in ('parent', 'first_child'):
                # 终局节点的 first_child 可能指向数组末尾之后。
                # 只有已展开的非终局节点拥有有效子节点范围。
                valid = child_counts > 0 if name == 'first_child' else kept[:len(order)] >= 0
                if name == 'first_child':
                    kept[:len(order)][~valid] = -1
                kept[:len(order)][valid] = mapping[kept[:len(order)][valid]]
            setattr(self, name, kept)
        self.parent[0] = -1
        self.root_id, self.num_nodes = 0, len(order)

    @property
    def root(self) -> 'SearchNode':
        """访问根节点。"""
        return SearchNode(self, self.root_id)

    @contextmanager
    def search_forward(self, game: SearchableEnv) \
            -> Generator['ForwardSearchIterator', None, None]:
        """保存当前游戏状态，返回从根节点开始的搜索迭代器。"""
        assert self.root_id >= 0 and self.root_id < self.num_nodes, \
            'invalid root node'
        assert self.root.is_evaluated(), 'unevaluated root'
        assert not self.root.is_terminal(), 'terminal root'
        game.snapshot()
        try:
            yield ForwardSearchIterator(self.root, game)
        finally:
            game.restore()


class SearchNode:
    """搜索节点及其 MCTS 统计的访问接口。"""

    def __init__(self, tree: SearchTree, node_id: int) -> None:
        assert node_id >= 0, 'illegal node'
        assert node_id < tree.num_nodes, 'illegal node'
        self.tree = tree
        self._id = node_id

    def __eq__(self, other: 'SearchNode') -> bool:
        return self.id == other.id

    @property
    def id(self) -> int32:
        """读取节点编号。"""
        return self._id

    @property
    def parent(self) -> 'SearchNode':
        """获取父节点。"""
        assert not self.is_root(), 'root has no parent'
        return SearchNode(self.tree, self.tree.parent[self.id])

    def child(self, move_id: int32) -> 'SearchNode':
        """获取 move_id 指定的子节点。"""
        assert self.is_evaluated(), 'unevaluated node'
        assert move_id >= 0, 'illegal child'
        assert move_id < self.num_children, 'illegal child'
        node_id = self.tree.first_child[self.id] + move_id
        return SearchNode(self.tree, node_id)

    @property
    def move_stats(self) -> 'SearchStats':
        """返回各动作的向量化搜索统计，统一为当前执棋方视角。"""
        assert self.is_evaluated(), 'unevaluated nodes have no stats'
        first_child = self.tree.first_child[self.id]
        node_ids = slice(first_child, first_child + self.num_children)
        # 先验已采用父节点视角；
        # 价值采用子节点自身视角，需要取反。
        return SearchStats(self.tree.num_visits[node_ids],
                           -self.tree.total_value[node_ids],
                           self.tree.prior_prob[node_ids])

    @property
    def num_children(self) -> int32:
        return self.tree.num_children[self.id]

    @property
    def num_visits(self) -> float32:
        return self.tree.num_visits[self.id]

    @num_visits.setter
    def num_visits(self, visits: float32) -> None:
        self.tree.num_visits[self.id] = visits
    
    @property
    def total_value(self) -> float32:
        return self.tree.total_value[self.id]

    @total_value.setter
    def total_value(self, value: float32) -> None:
        self.tree.total_value[self.id] = value
    
    @property
    def prior_prob(self):
        return self.tree.prior_prob[self.id]

    def is_evaluated(self) -> bool:
        """判断是否已评估；未评估叶子的子节点和搜索统计尚未初始化。"""
        return self.num_children >= 0

    def is_terminal(self) -> bool:
        """判断是否为终局；终局节点已评估且没有子节点。"""
        return self.num_children == 0

    def is_root(self) -> bool:
        """判断是否为根节点。"""
        return self.id == self.tree.root_id

    def is_leaf(self) -> bool:
        """判断是否为叶节点。"""
        return (not self.is_evaluated()) or self.is_terminal()

    def create_child_nodes(self, num_children: int,
                           prior_prob: np.ndarray) -> None:
        """创建子节点并初始化搜索统计。"""
        self.tree.reserve(self.tree.num_nodes + num_children)

        first_node_id = self.tree.num_nodes
        self.tree.num_nodes += num_children
        assert self.tree.num_nodes <= MAX_NODES, 'tree nodes buffer overflow'

        self.tree.first_child[self.id] = first_node_id
        self.tree.num_children[self.id] = num_children

        node_ids = slice(first_node_id, first_node_id + num_children)
        self.tree.parent[node_ids] = self.id
        self.tree.first_child[node_ids] = -1
        self.tree.num_children[node_ids] = -1  # 标记为尚未评估。
        self.tree.num_visits[node_ids] = 0
        self.tree.total_value[node_ids] = 0
        self.tree.prior_prob[node_ids] = prior_prob


@dataclass
class SearchStats:
    num_visits: np.ndarray  # 一维 float32 数组。
    total_value: np.ndarray  # 一维 float32 数组。
    prior_prob: np.ndarray  # 一维 float32 数组。


class ForwardSearchIterator:
    """搜索时同步推进游戏状态和树节点。"""

    def __init__(self, node: SearchNode, game: SearchableEnv) -> None:
        self._game = game
        self._node = node
        if DEBUG:
            print(f'search: node {self._node.id} '
                  f'nch {self._node.num_children} '
                  f'game {hash(str(astuple(self._game.state)))} '
                  f'nmv {len(self._game.state.legal_moves)}')

    def step(self, move_id: int) -> None:
        """按当前合法动作列表中的序号 move_id，同步推进棋盘和搜索树。"""
        state = self._game.state
        assert move_id < len(state.legal_moves), \
            f"move id {move_id} out of range"
        assert not state.result, 'game ended in select'
        move = state.legal_moves[move_id]
        self._game.step(move)
        self._node = self._node.child(move_id)
        if DEBUG:
            print(f'step {move}: node {self._node.id} '
                  f'nch {self._node.num_children} '
                  f'game {hash(str(astuple(self._game.state)))} '
                  f'nmv {len(self._game.state.legal_moves)}')
            # 此处暂不打印棋盘状态。

    @property
    def state(self) -> GameState:
        """获取当前搜索局面的游戏状态。"""
        return self._game.state

    @property
    def node(self) -> SearchNode:
        """获取当前搜索位置的树节点。"""
        return self._node


def as_distribution(counts: np.ndarray, temperature: float = 1.0) \
        -> np.ndarray:
    """按温度调整访问次数，并归一化为动作概率分布。"""
    assert all(counts >= 0)
    log_pi = np.log(counts.clip(min=1))
    log_pi[counts == 0] = -np.inf
    if temperature:
        log_pi = log_pi / temperature
    else:
        log_pi[log_pi < log_pi.max()] = -np.inf
    # 避免数值不稳定。
    log_pi = log_pi.astype(np.float64)
    log_z = np.logaddexp.reduce(log_pi)
    pi = np.exp(log_pi - log_z)
    return pi


def print_tree(tree: SearchTree) -> None:
    """以树状缩进打印节点、动作和价值统计。"""
    for line in format_tree_lines(tree):
        print(line)


def format_tree_lines(tree: SearchTree) -> Generator[str, None, None]:
    yield format_node(tree.root)
    for line in format_subtree_lines(tree, tree.root, 1, []):
        yield line


def format_subtree_lines(tree: SearchTree, node: SearchNode, indent: int,
                         last: List[bool]) -> Generator[str, None, None]:
    indent += 1
    for i in range(node.num_children):
        child = node.child(i)
        last.append(i == node.num_children - 1)
        prefix = format_tree_prefix(last)
        suffix = format_node(child, move_id=i)
        yield prefix + suffix
        if not child.is_leaf():
            for line in format_subtree_lines(tree, child, indent, last):
                yield line
        last.pop()


def format_tree_prefix(last: List[bool]) -> str:
    if not last:
        return ''

    prefix = ''.join('    ' if x else '│   ' for x in last[:-1])
    prefix += '└── ' if last[-1] else '├── '
    return prefix


def format_node(node: SearchNode, move_id: Optional[int] = None) -> str:
    num_visits = node.num_visits
    value = node.total_value / max(1, num_visits)
    prior = node.prior_prob
    txt = f'{node.id} '
    if move_id is not None:
        txt += f'[{move_id}] '
    txt += f'value {value:.2f} vis {num_visits:.0f} (pri {prior:.2f})'
    return txt
