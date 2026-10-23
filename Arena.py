import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """组织两个策略进行对弈。"""

    def __init__(self, player1, player2, game, display=None):
        """player1、player2 接收规范化棋盘并返回动作；game 提供规则。
        详细输出模式需传入棋盘显示函数 display。"""
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """执行一局，返回胜者：1 表示 player1，−1 表示 player2。
        平局时返回规则定义的非零平局值。"""
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            active = players[curPlayer + 1]
            if hasattr(active, 'select_action'):
                canonical_action = active.select_action(board, curPlayer)
            else:
                canonical_action = active(self.game.getCanonicalForm(board, curPlayer))
            action = self.game.actionFromCanonical(canonical_action, curPlayer)

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if not 0 <= canonical_action < len(valids) or valids[canonical_action] == 0:
                raise ValueError(f'Action {canonical_action} is not valid')

            # 通知对手引擎更新搜索树。
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, canonical_action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """双方各先手一半，返回各自胜局数及平局数。"""

        if num < 2 or num % 2:
            raise ValueError("Arena game count must be a positive even number")
        oneWon = twoWon = draws = 0
        original = self.player1, self.player2
        try:
            for index in tqdm(range(num), desc="Arena.playGames"):
                swapped = bool(index % 2)
                self.player1, self.player2 = original[::-1] if swapped else original
                result = self.playGame(verbose=verbose)
                if result in (-1, 1):
                    result *= -1 if swapped else 1
                    oneWon += result == 1
                    twoWon += result == -1
                else:
                    draws += 1
        finally:
            self.player1, self.player2 = original
        return oneWon, twoWon, draws
