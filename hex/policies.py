import numpy as np

from MCTS import MCTS
from utils import dotdict


class ClassicPolicy:
    """评估适配器：每局独立搜索树，原始棋盘先验和规范化动作。"""

    def __init__(self, game, nnet, args, seed=None):
        """包装经典 MCTS，保存网络和搜索参数，供统一评估接口调用。"""
        self.game, self.nnet = game, nnet
        self.args = dotdict(dict(args))
        if seed is not None:
            self.args.seed = seed
        self.rng = np.random.default_rng(self.args.get('seed'))
        self.mcts = None
        self.last_move = None

    def startGame(self):
        """为新对局重建搜索器并清空上一手，避免跨局复用统计。"""
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.mcts.rng = self.rng
        self.last_move = None

    def __call__(self, canonical_board):
        """接收规范棋盘并选择贪心动作，兼容评估器可调用策略接口。"""
        if self.mcts is None:
            self.startGame()
        return int(np.argmax(self.mcts.getActionProb(canonical_board, temp=0)))

    def select_action(self, board, player):
        """在原棋盘上处理可选先验或战术，再转换为规范坐标供评估器执行。"""
        action = None
        if self.args.get('useOpeningBook', False):
            from .opening_book import opening_book_action
            action = opening_book_action(board, player, self.game)
        if action is None and self.args.get('useTacticalCheck', False):
            from .hex_tactics import tactical_action
            action = tactical_action(board, player, self.game, last_move=self.last_move)
        if action is not None:
            return self.game.actionToCanonical(action, player)
        return self(self.game.getCanonicalForm(board, player))

    def notify(self, board, action):
        """记录实际上一手；经典搜索不显式换根，由状态键复用已有缓存。"""
        mover = 1 if np.count_nonzero(board == 1) == np.count_nonzero(board == -1) else -1
        self.last_move = self.game.actionFromCanonical(action, mover)
