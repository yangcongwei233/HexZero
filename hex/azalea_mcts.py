import logging
import math
import time

import numpy as np

from .search_kernels import select_edge


EPS = 1e-8
log = logging.getLogger(__name__)


def arg_value(args, name, default):
    try:
        return getattr(args, name)
    except (AttributeError, KeyError):
        return default


class EdgeStats(dict):
    """Sparse public statistics with a synchronized dense view for native PUCT."""
    def __init__(self, size, dtype):
        super().__init__()
        self.dense = np.zeros(size, dtype=dtype)

    def __setitem__(self, key, value):
        self.dense[key] = value
        super().__setitem__(key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.dense[key] = 0

    def clear(self):
        super().clear()
        self.dense.fill(0)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=0):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        if len(default) > 1:
            raise TypeError('pop expected at most 2 arguments')
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default:
            return default[0]
        raise KeyError(key)

    def popitem(self):
        key, value = super().popitem()
        self.dense[key] = 0
        return key, value

    def __ior__(self, other):
        self.update(other)
        return self

    def __reduce__(self):
        return type(self), (len(self.dense), self.dense.dtype), None, None, iter(self.items())


class SearchNode:
    def __init__(self, board, parent=None, action_from_parent=None, prior=1.0):
        self.board = np.asarray(board, dtype=np.int8).copy()
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = float(prior)
        self.children = {}
        self.P = None
        self.valids = None
        self.terminal = None
        self.N = 0
        self.virtual_visits = EdgeStats(self.board.size, np.int64)
        self.Nsa = EdgeStats(self.board.size, np.int64)
        self.Wsa = EdgeStats(self.board.size, np.float64)
        self.Qsa = {}

    def is_expanded(self):
        return self.P is not None or self.terminal is not None

    def child(self, action):
        return self.children.get(int(action))

    def backup(self, action, value):
        action = int(action)
        old_n = self.Nsa.get(action, 0)
        new_n = old_n + 1
        self.Nsa[action] = new_n
        self.Wsa[action] = self.Wsa.get(action, 0.0) + float(value)
        self.Qsa[action] = self.Wsa[action] / new_n
        self.N += 1


class SearchTree:
    def __init__(self, game):
        self.game = game
        self.root = None

    def reset(self, canonical_board=None):
        self.root = SearchNode(canonical_board if canonical_board is not None else self.game.getInitBoard())

    def set_root(self, canonical_board):
        if self.root is None or not np.array_equal(self.root.board, canonical_board):
            self.reset(canonical_board)

    def update_with_move(self, action):
        if self.root is None:
            return False
        child = self.root.child(action)
        if child is None:
            self.root = None
            return False
        child.parent = None
        child.action_from_parent = None
        self.root = child
        return True


class AzaleaMCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.tree = SearchTree(game)
        self.rng = np.random.default_rng(arg_value(args, "seed", None))
        self.last_value = 0.0
        self.last_topk = []
        self.last_search_time = 0.0
        self.last_simulations = 0
        self.last_stop_reason = 'simulation_limit'
        self._batch_seconds = None
        self._node_count = 0
        self._root_prior = None

    def getActionProb(self, canonicalBoard, temp=1, add_root_noise=None):
        start = time.perf_counter()
        time_limit = float(arg_value(self.args, 'searchTimeLimit', 0.0))
        if not math.isfinite(time_limit) or time_limit < 0:
            raise ValueError('searchTimeLimit must be finite and nonnegative')
        deadline = start + time_limit if time_limit else None
        canonicalBoard = np.asarray(canonicalBoard, dtype=np.int8)
        if arg_value(self.args, "reuseSearchTree", False):
            self.tree.set_root(canonicalBoard)
        else:
            self.tree.reset(canonicalBoard)

        if add_root_noise is None:
            add_root_noise = bool(arg_value(self.args, "rootDirichletNoise", False))

        sims = int(arg_value(self.args, "numMCTSSims", 25))
        batch_size = max(1, int(arg_value(self.args, "mctsBatchSize", 1)))
        if sims < 1 or temp < 0:
            raise ValueError("numMCTSSims must be positive and temperature nonnegative")
        completed = 0
        root = self.tree.root
        self._node_count = 0
        if arg_value(self.args, 'maxSearchNodes', 0) and not hasattr(self.tree, 'handle'):
            stack = [root]
            while stack:
                node = stack.pop()
                self._node_count += 1
                stack.extend(node.children.values())
        if not root.is_expanded():
            root_started = time.perf_counter()
            self._evaluate_and_backup([(root, [])])
            completed = 1  # Root evaluation consumes one simulation, not a batch.
            if deadline is not None and self._batch_seconds is None:
                self._batch_seconds = (time.perf_counter() - root_started) * batch_size
        if root.terminal:
            raise ValueError("Cannot search a terminal board")
        self._root_prior = root.P.copy()
        if add_root_noise:
            legal = np.flatnonzero(root.valids)
            alpha = float(arg_value(self.args, "dirichletAlpha", 0.3))
            frac = float(arg_value(self.args, "dirichletFraction", 0.25))
            if alpha <= 0 or not 0 <= frac <= 1:
                raise ValueError("Dirichlet alpha must be positive and fraction in [0, 1]")
            noise = self.rng.dirichlet(np.full(len(legal), alpha))
            self._root_prior[legal] = (1 - frac) * root.P[legal] + frac * noise
        stop_reason = 'simulation_limit'
        while completed < sims:
            if deadline is not None:
                # A native inference call cannot be interrupted safely midway.
                # Reserve a conservative estimate of the next complete batch.
                estimate = max(0.001, self._batch_seconds or 0.005)
                if time.perf_counter() + estimate * 1.3 + 0.003 >= deadline:
                    stop_reason = 'time_budget'
                    break
            current_batch = min(batch_size, sims - completed)
            current_batch = self._safe_batch_size(current_batch)
            if current_batch <= 0:
                stop_reason = 'node_limit'
                break
            batch_started = time.perf_counter() if deadline is not None else 0
            leaves = self._select_batch(current_batch)
            self._evaluate_and_backup(leaves)
            completed += len(leaves)
            if deadline is not None:
                duration = time.perf_counter() - batch_started
                self._batch_seconds = max(duration, (self._batch_seconds or duration) * 0.9)

        counts = np.array(
            [self.tree.root.Nsa.get(a, 0) for a in range(self.game.getActionSize())],
            dtype=np.float64,
        )
        self.last_search_time = time.perf_counter() - start
        self.last_simulations = completed
        self.last_stop_reason = stop_reason
        self.last_value = self._root_value()
        self.last_topk = self._topk(counts)

        valids = self.game.getValidMoves(canonicalBoard, 1)
        counts *= valids
        if counts.sum() <= 0:
            counts = self._root_prior.copy()

        if temp == 0:
            best = np.flatnonzero(counts == counts.max())
            action = int(self.rng.choice(best))
            probs = np.zeros_like(counts)
            probs[action] = 1.0
            return probs.tolist()

        counts = (counts / counts.max()) ** (1.0 / float(temp))
        probs = counts / counts.sum()
        return probs.tolist()

    def update_with_move(self, action):
        if not arg_value(self.args, "reuseSearchTree", False):
            self.tree.root = None
            return False
        return self.tree.update_with_move(action)

    def _safe_batch_size(self, requested):
        limit = arg_value(self.args, 'maxSearchNodes', 0)
        return min(requested, max(0, limit - self._node_count)) if limit else requested

    def _select_batch(self, batch_size):
        leaves = []
        reservations = []
        try:
            for _ in range(batch_size):
                node = self.tree.root
                path = []
                while node.is_expanded() and node.terminal == 0:
                    action = self._select_action(node)
                    path.append((node, action))
                    node.virtual_visits[action] = node.virtual_visits.get(action, 0) + 1
                    reservations.append((node, action))
                    child = node.child(action)
                    if child is None:
                        next_s, next_player = self.game.getNextState(node.board, 1, action)
                        next_s = self.game.getCanonicalForm(next_s, next_player)
                        child = SearchNode(next_s, parent=node, action_from_parent=action, prior=node.P[action])
                        node.children[action] = child
                        self._node_count += 1
                    node = child
                # Keep every selected path; deduplicate inference, not backups.
                leaves.append((node, path))
        finally:
            for node, action in reservations:
                node.virtual_visits[action] -= 1
                if node.virtual_visits[action] == 0:
                    del node.virtual_visits[action]
        return leaves

    def _select_action(self, node):
        prior = self._root_prior if node is self.tree.root else node.P
        parent_visits = max(1, node.N + sum(node.virtual_visits.values()))
        cpuct = float(arg_value(self.args, "cpuct", 1.0))
        if select_edge is not None:
            return int(select_edge(node.valids, prior, node.Nsa.dense, node.Wsa.dense,
                                   node.virtual_visits.dense, parent_visits, cpuct))
        best_action, best_score = -1, -float("inf")
        for action in np.flatnonzero(node.valids):
            action = int(action)
            pending = node.virtual_visits.get(action, 0)
            n = node.Nsa.get(action, 0) + pending
            q = (node.Wsa.get(action, 0.0) - pending) / n if n else 0.0
            score = q + cpuct * prior[action] * math.sqrt(parent_visits + EPS) / (1 + n)
            if score > best_score:
                best_score, best_action = score, action
        return best_action

    def _evaluate_and_backup(self, leaves):
        if not leaves:
            return

        eval_items = []
        values = {}
        seen = set()
        for node, _ in leaves:
            if id(node) in seen:
                continue
            seen.add(id(node))
            ended = node.terminal if node.terminal is not None else self.game.getGameEnded(node.board, 1)
            node.terminal = ended
            if ended != 0:
                values[id(node)] = float(ended)
            else:
                eval_items.append(node)

        if eval_items:
            boards = [node.board for node in eval_items]
            policies, net_values = self.nnet.predict_batch(boards)
            for node, policy, value in zip(eval_items, policies, net_values):
                node.P = self._masked_policy(node.board, policy)
                node.valids = self.game.getValidMoves(node.board, 1)
                values[id(node)] = float(np.asarray(value).reshape(-1)[0])

        for node, path in leaves:
            if node.terminal == 0 and node.P is None:
                continue
            leaf_value = values[id(node)]
            value = -leaf_value
            for parent, action in reversed(path):
                parent.backup(action, value)
                value = -value

    def _masked_policy(self, board, policy):
        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        heuristic_prior_weight = max(0.0, min(1.0, arg_value(self.args, "heuristicPriorWeight", 0.0)))
        if heuristic_prior_weight > 0:
            from hex.hex_heuristics import heuristic_policy
            heuristic = heuristic_policy(board, 1, self.game)
            policy = (1.0 - heuristic_prior_weight) * policy + heuristic_prior_weight * heuristic
        valids = self.game.getValidMoves(board, 1)
        policy = policy * valids
        total = float(policy.sum())
        if total > 0:
            return policy / total
        log.error("All valid moves were masked in azalea backend, using uniform valid policy.")
        policy = valids.astype(np.float64)
        return policy / max(float(policy.sum()), 1.0)

    def _root_value(self):
        root = self.tree.root
        if root is None or not root.Nsa:
            return 0.0
        total = 0.0
        visits = 0
        for action, n in root.Nsa.items():
            total += root.Qsa[action] * n
            visits += n
        return total / max(1, visits)

    def _topk(self, counts, k=5):
        actions = np.argsort(counts)[::-1][:k]
        return [(int(a), int(counts[a])) for a in actions if counts[a] > 0]


class AzaleaStylePolicy:
    def __init__(self, game, nnet, args, mcts_factory=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = (mcts_factory or AzaleaMCTS)(game, nnet, args)
        self.last_move = None

    def startGame(self):
        self.mcts.tree.root = None
        self.last_move = None

    def __call__(self, canonical_board):
        action, _, _, _, _ = self.choose_action(canonical_board, temperature=0, add_root_noise=False)
        self.execute_action(action)
        return action

    def select_action(self, board, player):
        action, _, _, _, _ = self.choose_action(
            self.game.getCanonicalForm(board, player), temperature=0,
            raw_board=board, player=player, add_root_noise=False,
        )
        self.execute_action(action)
        return action

    def choose_action(self, canonical_board, temperature=1, raw_board=None, player=1, add_root_noise=None):
        action = None
        if raw_board is not None and arg_value(self.args, "useOpeningBook", False):
            from hex.opening_book import opening_book_action
            action = opening_book_action(raw_board, player, self.game)
        if action is None and raw_board is not None and arg_value(self.args, "useTacticalCheck", False):
            from hex.hex_tactics import tactical_action
            action = tactical_action(raw_board, player, self.game, last_move=self.last_move)

        if action is not None:
            action = self.game.actionToCanonical(action, player)
            pi = np.zeros(self.game.getActionSize(), dtype=np.float64)
            pi[int(action)] = 1.0
            return int(action), pi.tolist(), 0.0, [], 0.0

        pi = self.mcts.getActionProb(canonical_board, temp=temperature, add_root_noise=add_root_noise)
        action = int(self.mcts.rng.choice(len(pi), p=pi)) if temperature != 0 else int(np.argmax(pi))
        return action, pi, self.mcts.last_value, self.mcts.last_topk, self.mcts.last_search_time

    def execute_action(self, action):
        return self.mcts.update_with_move(action)

    def notify(self, board, action):
        # Arena sends the mover's canonical action; a reused root has that perspective.
        mover = 1 if np.count_nonzero(board == 1) == np.count_nonzero(board == -1) else -1
        self.last_move = self.game.actionFromCanonical(action, mover)
        self.execute_action(action)
