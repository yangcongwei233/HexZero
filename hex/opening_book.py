import json
from pathlib import Path

import numpy as np


BOOK_PATH = Path(__file__).with_name("opening_book.json")


def board_string(board):
    values = np.asarray(board, dtype=np.int8).ravel()
    return "".join(str(int(v)) for v in values)


def book_key(board, player, game):
    return "{}|{}|{}".format(game.n, int(player), board_string(board))


def load_opening_book(path=BOOK_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def opening_book_action(board, player, game, book=None):
    if book is None:
        book = load_opening_book()

    actions = book.get(book_key(board, player, game))
    if not actions:
        return None

    valids = game.getValidMoves(board, player)
    for action in actions:
        action = int(action)
        if 0 <= action < len(valids) and valids[action] > 0:
            return action
    return None
