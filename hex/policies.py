import numpy as np

from MCTS import MCTS
from utils import dotdict


class ClassicPolicy:
    """Arena adapter: per-game trees, raw-board priors and canonical actions."""

    def __init__(self, game, nnet, args, seed=None):
        self.game, self.nnet = game, nnet
        self.args = dotdict(dict(args))
        if seed is not None:
            self.args.seed = seed
        self.rng = np.random.default_rng(self.args.get('seed'))
        self.mcts = None
        self.last_move = None

    def startGame(self):
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.mcts.rng = self.rng
        self.last_move = None

    def __call__(self, canonical_board):
        if self.mcts is None:
            self.startGame()
        return int(np.argmax(self.mcts.getActionProb(canonical_board, temp=0)))

    def select_action(self, board, player):
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
        mover = 1 if np.count_nonzero(board == 1) == np.count_nonzero(board == -1) else -1
        self.last_move = self.game.actionFromCanonical(action, mover)
