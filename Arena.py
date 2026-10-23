import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
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

            # Notifying the opponent for the move
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
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

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
