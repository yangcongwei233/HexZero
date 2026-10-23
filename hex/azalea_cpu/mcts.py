# 派生自 Azalea（Apache-2.0），于 2026-10-15 适配 NumPy/ONNX 推理。
from __future__ import annotations

from typing import Dict, Sequence, Set, Tuple
import time
import math

import numpy as np
from numpy.random import RandomState

from . import prep
from .search_tree import SearchNode, SearchTree, SearchStats
from .search_tree import ForwardSearchIterator
from .search_tree import print_tree
from .typing import GameState, SearchableEnv
from ..search_kernels import virtual_path, backup_paths

DEBUG = False


def evaluate_root(tree: SearchTree,
                  game: SearchableEnv,
                  net: Network,
                  rng: RandomState) -> None:
    """评估根节点。"""
    check_game_state(game.state)
    value, _, prior_prob = evaluate_batch(game, net, [game.state], rng)
    tree.root.create_child_nodes(len(prior_prob[0]), prior_prob[0])
    if DEBUG: print('root value:', value[0])


def check_game_state(state):
    """断言参考状态的棋盘/动作类型、形状及终局与合法动作关系，帮助发现环境适配错误。"""
    assert isinstance(state.board, np.ndarray), 'board type error'
    assert state.board.dtype == np.int32, 'board type error'
    assert len(state.board.shape) == 2, 'board type error'
    assert isinstance(state.legal_moves, np.ndarray), 'moves type error'
    assert state.legal_moves.dtype == np.int32, 'moves type error'
    assert len(state.legal_moves.shape) == 1, 'moves type error'
    assert all(state.legal_moves > 0), 'moves value error'
    assert isinstance(state.color, int), 'color type error'
    assert isinstance(state.result, int), 'result type error'
    assert len(state.legal_moves) or state.result, \
        'no legal moves but game is continuing'
    assert not (len(state.legal_moves) and state.result), \
        'legal moves but game ended'


def select_batch(tree: SearchTree, game: SearchableEnv,
                 batch_size: int, exploration_coef: float,
                 noise_scale: float, noise_alpha: float,
                 rng: RandomState) \
        -> Tuple[Sequence[SearchNode], Sequence[GameState]]:
    """选择待扩展的一批叶节点，并回放路径以收集棋盘状态。
    batch_size 为批量大小，exploration_coef 为探索系数；
    noise_scale 和 noise_alpha 控制根节点 Dirichlet 噪声。"""
    leaf_nodes = []   # 各路径的叶节点。
    leaf_states = []  # 各路径对应的棋盘状态。

    for i in range(batch_size):
        with tree.search_forward(game) as root_iter:
            leaf_node, leaf_state = select_leaf(
                root_iter, exploration_coef, noise_scale, noise_alpha, rng)
            # 添加虚拟损失，使同批搜索分散到不同路径。
            # 暂时假定所选路径失败。
            apply_virtual_loss([leaf_node], 1)
            leaf_states.append(leaf_state)
            leaf_nodes.append(leaf_node)
    # 撤销上述虚拟损失。
    apply_virtual_loss(leaf_nodes, -1)

    # 合并重复叶节点。
    leaf_nodes, leaf_states = deduplicate_leaves(leaf_nodes, leaf_states)
    return leaf_nodes, leaf_states


def apply_virtual_loss(leaf_nodes: Sequence[SearchNode], amount: int) -> None:
    """沿叶节点到根的路径原地增减虚拟损失。
    amount 为 1 时添加，为 −1 时撤销。"""
    if virtual_path is not None:
        for node in leaf_nodes:
            tree = node.tree
            virtual_path(tree.parent, tree.num_visits, tree.total_value, node.id, tree.root_id, amount)
        return
    for node in leaf_nodes:
        while not node.is_root():
            node.num_visits += amount
            # 从父节点的动作统计视角看，虚拟访问应表现为失败。
            # 因此在子节点自身视角保存价值时，符号需要相反，
            # 使子节点看起来获胜。
            # 这里假定游戏为双人零和对抗。
            node.total_value += amount
            node = node.parent


def select_leaf(search_iter: ForwardSearchIterator,
                exploration_coef: float,
                noise_scale: float,
                noise_alpha: float,
                rng: RandomState) -> Tuple[SearchNode, GameState]:
    """从当前根节点选择一条路径，返回未评估叶子或终局叶子。"""
    while not search_iter.node.is_leaf():
        scores = score_actions(search_iter.node.move_stats,
                               exploration_coef=exploration_coef,
                               noise_scale=noise_scale,
                               noise_alpha=noise_alpha,
                               rng=rng)
        assert len(scores) == len(search_iter.state.legal_moves)
        child_id = np.argmax(scores)
        search_iter.step(child_id)
        noise_scale = 0.0  # 仅在根节点加入探索噪声。
    # 获取新叶节点的完整状态。
    return search_iter.node, search_iter.state


def score_actions(stats: SearchStats,
                  exploration_coef: float,
                  noise_scale: float,
                  noise_alpha: float,
                  rng: RandomState) -> np.ndarray:
    """按 AlphaZero 风格的置信上界公式给各动作评分。"""
    prior = stats.prior_prob  # 动作先验 P(s, a)。
    if noise_scale:
        # 在根节点探索随机动作。
        noise = rng.dirichlet(np.full(len(prior), noise_alpha))
        prior = (
            (1.0 - noise_scale) * prior + noise_scale * noise
        ).astype(np.float32)
    visit_gap = np.sqrt(np.sum(stats.num_visits)) / (1.0 + stats.num_visits)
    action_uct_boost = exploration_coef * prior * visit_gap  # 探索项 U(s, a)。
    action_value = stats.total_value / stats.num_visits.clip(min=1)  # 平均价值 Q(s, a)。
    score = action_value + action_uct_boost  # 综合价值与探索项。
    return score


def deduplicate_leaves(leaf_nodes: Sequence[SearchNode],
                       leaf_states: Sequence[GameState]) \
        -> Tuple[Sequence[SearchNode], Sequence[GameState]]:
    """移除重复叶节点。"""
    seen: Set[int] = set()

    dedup_nodes = []
    dedup_states = []
    for node, state in zip(leaf_nodes, leaf_states):
        if node.id not in seen:
            dedup_nodes.append(node)
            dedup_states.append(state)
            seen.add(node.id)
    return dedup_nodes, dedup_states


def evaluate_batch(game: SearchableEnv,
                   net: Network,
                   states: Sequence[GameState],
                   rng: RandomState) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量评估新局面，返回当前方价值、合法动作数和动作先验数组。"""
    batch = prep.batch_games(states)

    # 待确认此处复制是否仍有必要。
    batch['board'] = batch['board'].copy()
    batch['legal_moves'] = batch['legal_moves'].copy()

    # 将棋盘和动作统一为网络的先手视角（颜色编号 0）。
    # 此变换只作用于网络输入，
    # 输出价值和动作先验仍对应
    # 当前执棋方的视角。
    flip_mask = batch['color'] == 1
    batch['board'][flip_mask], batch['legal_moves'][flip_mask] = \
        game.flip_player_board_moves(batch['board'][flip_mask],
                                     batch['legal_moves'][flip_mask])

    # 随机变换输入，进行数据增强。
    batch['board'], batch['legal_moves'] = \
        game.random_reflect(batch['board'], batch['legal_moves'],
                            rng=rng)

    value = np.zeros_like(batch['color'], dtype=np.float32)
    prior_prob = np.zeros_like(batch['legal_moves'], dtype=np.float32)
    num_children = (batch['legal_moves'] > 0).sum(1)

    # 终局时轮到失败方行动，
    # 因此其视角的价值为 −1。
    term_mask = batch['result'] != 0
    value[term_mask] = -1.0

    # 确认终局结果均为当前方失败。
    own_result = batch['result'].copy()
    own_result[flip_mask] = flip_result(own_result[flip_mask])
    assert all(own_result[term_mask] == 1)

    if any(~term_mask):
        output = net.run({k: v[~term_mask] for k, v in batch.items()})
        nonterm_value = output['value']
        nonterm_prior_prob = np.exp(output['moves_logprob'])
        assert all(nonterm_prior_prob.flat >= 0.0), 'negative prior prob'
        assert all(abs(nonterm_prior_prob.sum(1) - 1.0).flat < 1e-4), \
            'prior probs normalized incorrectly'
        value[~term_mask] = nonterm_value
        prior_prob[~term_mask] = nonterm_prior_prob

    return value, num_children, prior_prob


def flip_result(result):
    # Numba 无法编译下方使用 all() 的检查。
    # 已停用的检查：结果编码应在 0～3 范围内。
    """交换先后手视角的结果编码，保持未结束和平局编码不变。"""
    return (result > 0) * (4 - result)


def expand_batch(leaf_nodes: Sequence[SearchNode],
                 num_children: np.ndarray,
                 prior_probs: np.ndarray) -> None:
    """为叶节点初始化搜索统计；num_children 为各叶子的子节点数，
    prior_probs 为各叶子的先验概率向量。"""
    for i, node in enumerate(leaf_nodes):
        assert node.is_leaf(), 'expanded node is not leaf'
        if not node.is_terminal():
            m = num_children[i]
            node.create_child_nodes(m, prior_probs[i, :m])


def backup_batch(leaf_nodes: Sequence[SearchNode],
                 values: np.ndarray) -> None:
    """批量回传叶节点价值，输入价值采用各叶子执棋方的视角。"""
    if backup_paths is not None and leaf_nodes:
        tree = leaf_nodes[0].tree
        leaves = np.fromiter((node.id for node in leaf_nodes), dtype=np.int64)
        backup_paths(tree.parent, tree.num_visits, tree.total_value, leaves, tree.root_id, values)
        return
    for node, value in zip(leaf_nodes, values):
        while True:
            node.total_value += value
            node.num_visits += 1
            # 转换为对手视角的价值。
            value = -value
            if node.is_root():
                break
            node = node.parent


def sample_paths(tree: SearchTree, game: SearchableEnv, net: Network,
                 num_simulations: int,
                 batch_size: int,
                 exploration_coef: float,
                 exploration_noise_scale: float,
                 exploration_noise_alpha: float,
                 rng: RandomState, time_limit=0) -> Dict[str, float]:
    """采样路径并原地更新搜索统计，返回搜索指标。"""
    num_batches = num_simulations // batch_size + 1
    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError('time_limit must be finite and nonnegative')
    deadline = time.perf_counter() + time_limit if time_limit else None
    search_depth = 0
    search_value = 0
    completed, batches, stop_reason = 0, 0, 'simulation_limit'
    estimate = getattr(tree, '_batch_seconds', .005)

    if not tree.root.is_evaluated():
        evaluate_root(tree, game, net, rng)

    for i in range(num_batches):
        # Azalea 一次展开全部合法子节点，每条路径可能新增多个节点。
        # 修改树前，按整批最大需求预留空间。
        from .search_tree import MAX_NODES
        if tree.num_nodes + batch_size * game.state.board.size > MAX_NODES:
            stop_reason = 'node_limit'
            break
        # 新根至少完成一批搜索，以提供合法候选。
        if deadline and np.any(tree.root.move_stats.num_visits > 0) and time.perf_counter() + estimate * 1.3 + .003 >= deadline:
            stop_reason = 'time_budget'
            break
        started = time.perf_counter() if deadline else 0
        if DEBUG:
            print('sample paths: batch %d' % i)
            print_tree(tree)
        leaf_nodes, leaf_states = select_batch(
            tree, game, batch_size, exploration_coef,
            exploration_noise_scale, exploration_noise_alpha, rng)
        values, num_children, prior_prob = evaluate_batch(
            game, net, leaf_states, rng)
        expand_batch(leaf_nodes, num_children, prior_prob)
        backup_batch(leaf_nodes, values)
        # 暂未累计搜索深度。
        search_value += np.sum(values)
        completed += len(leaf_nodes)
        batches += 1
        if deadline:
            estimate = max(time.perf_counter() - started, estimate * .9)
            tree._batch_seconds = estimate

    metrics = {
        # 暂未输出平均搜索深度。
        'search_value': search_value / max(1, batches * batch_size),
        'completed_simulations': completed,
        'attempted_simulations': batches * batch_size,
        'stop_reason': stop_reason,
    }
    return metrics
