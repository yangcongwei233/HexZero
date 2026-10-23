# 派生自 Azalea（Apache-2.0），于 2026-10-15 适配 NumPy/ONNX 推理。
from __future__ import annotations

import dataclasses as dc
from typing import Any, Dict, Optional, Sequence, Union, overload

import numpy as np

from .typing import GameState


ScalarOrArray = Union[float, np.ndarray]


def batch_games(seq: Sequence[GameState]) -> Dict[str, np.ndarray]:
    """将含标量或数组字段的数据类序列合并为批量数组字典。"""
    df = transpose_dataclass(seq)
    return {name: pad(df[name]) for name in df}


def transpose_dataclass(seq: Sequence) \
        -> Dict[str, Sequence[ScalarOrArray]]:
    """将数据类序列转置为按字段分组的列表字典。"""
    names = [f.name for f in dc.fields(seq[0])]
    values = zip(*map(dc.astuple, seq))
    # 按字段归并，例如棋盘列表和合法动作列表。
    return dict(zip(names, values))


@overload
def pad(mats: Sequence[float]) -> np.ndarray:
    """将标量列表转换为数组。"""
    ...


@overload
def pad(mats: Sequence[np.ndarray],
        size: Optional[np.ndarray]) -> np.ndarray:
    """对多维矩阵补零。"""
    ...


# 此函数暂不启用即时编译。
def pad(mats, size=None):
    """对多维矩阵补零。"""
    if isinstance(mats[0], (int, float)):
        assert size is None
        return np.array(mats)
    max_size = np.amax([m.shape for m in mats], 0)
    if size is None:
        size = max_size
    else:
        assert all(max_size <= size)
    padded_size = (len(mats),) + tuple(size)
    padded = np.zeros(padded_size, dtype=mats[0].dtype)
    for i, m in enumerate(mats):
        ids = [i] + [slice(s) for s in m.shape]
        padded[tuple(ids)] = m
    return padded
