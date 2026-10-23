
from typing import Optional
from typing_extensions import Protocol

import numpy as np


class SearchableEnv(Protocol):

    def reset(self, *args, **kwargs) -> None:
        """开始新对局。"""
        ...

    def seed(self, seed: Optional[int]) -> None:
        """设置随机种子。"""
        ...

    def step(self, action: int) -> None:
        """在游戏中执行一步。"""
        ...

    @property
    def state(self) -> 'GameState':
        """获取当前游戏状态。"""
        ...

    def snapshot(self) -> None:
        """保存当前状态快照。"""
        ...

    def restore(self) -> None:
        """从快照恢复游戏状态。"""
        ...


class GameState(Protocol):
    color: int
    legal_moves: np.ndarray
    result: int
    board: np.ndarray
