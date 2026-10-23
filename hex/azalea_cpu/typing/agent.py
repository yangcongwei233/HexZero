
from enum import IntEnum
from typing import Any, Mapping, Optional
from typing_extensions import Protocol


class Agent(Protocol):
    """对弈智能体接口。"""

    def reset(self, *args, **kwargs) -> None:
        """开始新对局。"""
        ...

    def seed(self, seed: Optional[int]) -> None:
        """设置随机种子。"""
        ...

    @property
    def settings(self) -> Mapping[str, Any]:
        """读写智能体设置。"""
        ...

    def choose_action(self) -> int:
        """规划下一步。"""
        ...

    def execute_action(self, action: int) -> 'GameResult':
        """执行动作并返回游戏结果。"""
        ...


class GameResult(IntEnum):
    """游戏结果，胜负统一采用先手视角。"""
    ONGOING = 0  # 游戏尚未结束。
    LOSS = 1     # 先手失败。
    DRAW = 2
    WIN = 3      # 先手获胜。
