"""编译数值热点，保持搜索顺序、预算和 IEEE 浮点运算语义。"""
import math

try:
    from numba import njit
except ImportError:
    njit = None


def _select_edge(valids, prior, visits, totals, pending, parent_visits, cpuct):
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
    while leaf != root:
        visits[leaf] += amount
        totals[leaf] += amount
        leaf = parent[leaf]


def _backup_paths(parent, visits, totals, leaves, root, values):
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
