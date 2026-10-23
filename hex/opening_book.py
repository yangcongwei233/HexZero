import json
from pathlib import Path

import numpy as np


BOOK_PATH = Path(__file__).with_name("opening_book.json")


def board_string(board):
    """将棋盘编码为稳定字符串，作为开局库键的一部分。"""
    values = np.asarray(board, dtype=np.int8).ravel()
    return "".join(str(int(v)) for v in values)


def book_key(board, player, game):
    """组合棋盘尺寸、当前方和棋盘编码，区分同一局面的不同落子方。"""
    return "{}|{}|{}".format(game.n, int(player), board_string(board))


def load_opening_book(path=BOOK_PATH):
    """读取可选开局库 JSON，供后续匹配局面使用。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def opening_book_action(board, player, game, book=None):
    """按当前棋盘查找开局库动作，并确认落点合法；未命中或无法使用时返回空值。"""
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
