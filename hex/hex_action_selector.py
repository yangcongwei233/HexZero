import numpy as np

from MCTS import MCTS
from utils import dotdict

from .hex_tactics import tactical_action
from .opening_book import opening_book_action


def select_action_with_priors(
    board,
    player,
    game,
    nnet,
    mcts_args,
    use_opening_book=False,
    use_tactical_check=False,
    last_move=None,
    temp=0,
):
    if use_opening_book:
        action = opening_book_action(board, player, game)
        if action is not None:
            return int(action), "opening_book", None

    if use_tactical_check:
        action = tactical_action(board, player, game, last_move=last_move)
        if action is not None:
            return int(action), "tactical", None

    args = dotdict(mcts_args.copy()) if isinstance(mcts_args, dict) else mcts_args
    mcts = MCTS(game, nnet, args)
    pi = mcts.getActionProb(game.getCanonicalForm(board, player), temp=temp)
    canonical_action = int(mcts.rng.choice(len(pi), p=pi)) if temp else int(np.argmax(pi))
    action = game.actionFromCanonical(canonical_action, player)
    return action, "mcts", pi
