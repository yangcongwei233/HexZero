"""编译数值热点，保持搜索顺序、预算和 IEEE 浮点运算语义。"""
import math

try:
    from numba import njit
except ImportError:
    njit = None


def _select_edge(valids, prior, visits, totals, pending, parent_visits, cpuct):
    """扫描合法动作，结合真实和虚拟访问计算 PUCT 评分，返回最高分动作；运算顺序与 Python 搜索一致。"""
    best_action, best_score = -1, -math.inf
    root = math.sqrt(parent_visits + 1e-8)
    for action in range(len(valids)):
        if not valids[action]:
            continue
        n = visits[action] + pending[action]
        q = (totals[action] - pending[action]) / n if n else 0.0
        score = q + cpuct * prior[action] * root / (1 + n)
        if score > best_score:
            best_score, best_action = score, action
    return best_action


def _virtual_path(parent, visits, totals, leaf, root, amount):
    """沿父链向根增减虚拟访问和价值，用于同批路径探索；不修改根节点。"""
    while leaf != root:
        visits[leaf] += amount
        totals[leaf] += amount
        leaf = parent[leaf]


def _backup_paths(parent, visits, totals, leaves, root, values):
    """按输入顺序逐条回传叶子价值，每上行一层翻转符号，同时累计节点访问次数。"""
    for i in range(len(leaves)):
        node, value = leaves[i], values[i]
        while True:
            totals[node] += value
            visits[node] += 1
            value = -value
            if node == root:
                break
            node = parent[node]


select_edge = njit(cache=True, fastmath=False)(_select_edge) if njit else None
virtual_path = njit(cache=True, fastmath=False)(_virtual_path) if njit else None
backup_paths = njit(cache=True, fastmath=False)(_backup_paths) if njit else None
