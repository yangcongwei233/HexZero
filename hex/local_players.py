"""Local match adapters. The reference uses Azalea's original network and MCTS."""
import copy
from dataclasses import dataclass
import hashlib
import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np

from hex.HexGame import HexGame
from hex.azalea_mcts import AzaleaStylePolicy
from hex.native_mcts import local_mcts
from utils import dotdict


class TrainedPlayer:
    def __init__(self, game, net, simulations, seed, move_time_ms=0):
        self.game = game
        self.move_time_ms = move_time_ms
        self.policy = AzaleaStylePolicy(game, net, dotdict(
            numMCTSSims=simulations, mctsBatchSize=32, cpuct=0.5,
            reuseSearchTree=True, rootDirichletNoise=False,
            heuristicPriorWeight=0, seed=seed, searchTimeLimit=0, maxSearchNodes=30000,
        ), mcts_factory=local_mcts)

    def choose(self, board, player, time_limit=None):
        started = time.perf_counter()
        self.policy.mcts.args.searchTimeLimit = (
            self.move_time_ms / 1000 if time_limit is None else time_limit)
        action, _, value, _, _ = self.policy.choose_action(
            self.game.getCanonicalForm(board, player), temperature=0,
            raw_board=board, player=player, add_root_noise=False,
        )
        return self.game.actionFromCanonical(action, player), {
            'seconds': time.perf_counter() - started, 'value': float(value),
            'searchSimulations': self.policy.mcts.last_simulations,
            'searchStopReason': self.policy.mcts.last_stop_reason,
            'moveTimeMs': self.move_time_ms,
        }

    def observe(self, board, player, action):
        self.policy.execute_action(self.game.actionToCanonical(action, player))
        self.policy.last_move = action


@dataclass
class ReferenceState:
    color: int
    legal_moves: np.ndarray
    result: int
    board: np.ndarray


class ReferenceEnvironment:
    """Azalea's 1-based moves and 0/1/2 boards over our tested Hex rules.

    Its second-player normalization is an anti-diagonal reflection, whereas
    HexZero uses a transpose. Keep Azalea's exact convention here.
    """
    def __init__(self, size):
        self.game = HexGame(size)
        self.board = self.game.getInitBoard()
        self.player = 1
        self.winner = None
        self._snapshot = None
        self._state_key = None
        self._cached_state = None

    @property
    def state(self):
        # Search asks for the same position several times before making a move.
        # Include the bytes so even direct board edits cannot return stale state.
        key = self.board.tobytes(), self.player, self.winner
        if key == self._state_key:
            return self._cached_state
        legal = (np.flatnonzero(self.board.ravel() == 0) + 1).astype(np.int32)
        if self.winner:
            legal = np.empty(0, dtype=np.int32)
        state = ReferenceState(
            0 if self.player == 1 else 1, legal,
            (3 if self.winner == 1 else 1) if self.winner else 0,
            np.where(self.board == -1, 2, self.board).astype(np.int32),
        )
        self._state_key, self._cached_state = key, state
        return state

    def step(self, move):
        if self.winner:
            raise ValueError('Cannot move after the game ends')
        self.board, self.player = self.game.getNextState(self.board, self.player, int(move) - 1)
        result = self.game.getGameEnded(self.board, self.player)
        self.winner = self.player * result if result else None

    def snapshot(self):
        self._snapshot = self.board.copy(), self.player, self.winner

    def restore(self):
        board, self.player, self.winner = self._snapshot
        self.board = board.copy()

    @staticmethod
    def flip_player_board_moves(board, moves):
        board, moves = np.asarray(board), np.asarray(moves)
        if board.ndim == 2:
            board, moves = board[None], moves[None]
        swapped = (board > 0) * (3 - board)
        flipped = np.flip(np.rot90(swapped, axes=(-2, -1)), axis=-1).copy()
        n = board.shape[-1]
        tiles = moves - 1
        transformed = (n - 1 - tiles % n) * n + (n - 1 - tiles // n) + 1
        return flipped, np.where(moves > 0, transformed, 0).astype(np.int32)

    @staticmethod
    def random_reflect(board, moves=None, rng=None):
        # This is also a no-op in the original reference implementation.
        return (board, moves) if moves is not None else board


class AzaleaReference:
    def __init__(self, source_dir, checkpoint, board_size, device):
        import torch
        source = Path(source_dir).resolve() / 'azalea'
        if not (source / 'policy.py').is_file():
            raise FileNotFoundError(f'找不到 azalea 源码：{source}')
        # Import only required original modules. The old package __init__ imports
        # unrelated training/CLI dependencies and an obsolete numba.jitclass API.
        namespace = '_hexzero_reference_' + hashlib.sha256(str(source).encode()).hexdigest()[:12]
        if namespace not in sys.modules:
            package = types.ModuleType(namespace)
            package.__path__ = [str(source)]
            sys.modules[namespace] = package
        Policy = importlib.import_module(namespace + '.policy').Policy
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['policy'].copy()
        if state['board_size'] != board_size:
            raise ValueError('Azalea 参考模型与训练模型的棋盘大小不同')
        if state['network_type'] not in ('HexNetwork', 'azalea.network.HexNetwork'):
            raise ValueError('只支持 Azalea HexNetwork 参考模型')
        state['network_type'] = namespace + '.network.HexNetwork'
        self.prototype = Policy()
        self.prototype.load_state_dict(state)
        self.prototype.net.to(device).eval()
        self.checkpoint = Path(checkpoint).resolve()
        self.board_size = board_size

    def player(self, simulations, seed):
        policy = copy.copy(self.prototype)
        policy.settings = {'move_sampling': False, 'move_exploration': False}
        policy.rng = np.random.RandomState(seed)
        policy.simulations = simulations
        policy.reset()
        return ReferencePlayer(policy, ReferenceEnvironment(self.board_size))


class ReferencePlayer:
    def __init__(self, policy, environment):
        self.policy, self.environment = policy, environment

    def choose(self, board, player, time_limit=None):
        if player != self.environment.player or not np.array_equal(board, self.environment.board):
            raise RuntimeError('Azalea 棋盘状态不同步')
        started = time.perf_counter()
        if time_limit:
            if not getattr(self.policy, 'supports_time_limit', False):
                raise ValueError('限时搜索需要 ONNX 本地版本的 Azalea')
        self.policy.time_limit = time_limit or 0
        move, info = self.policy.choose_action(self.environment)
        return int(move) - 1, {'seconds': time.perf_counter() - started,
                               'value': float(info['value']),
                               'searchSimulations': int(info.get('metrics', {}).get('completed_simulations', 0)),
                               'searchStopReason': info.get('metrics', {}).get('stop_reason', 'simulation_limit')}

    def observe(self, board, player, action):
        if player != self.environment.player or not np.array_equal(board, self.environment.board):
            raise RuntimeError('Azalea 棋盘状态不同步')
        self.policy.execute_action(int(action) + 1, self.environment.state.legal_moves)
        self.environment.step(int(action) + 1)
