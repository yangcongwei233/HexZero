import logging

import numpy as np


log = logging.getLogger(__name__)


def value_stats(values):
    """汇总价值数组的范围、均值、方差和标签分布，用于检查目标偏置或预测饱和。"""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "unique": {},
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    unique, counts = np.unique(values, return_counts=True)
    return {
        "count": int(values.size),
        "unique": {float(k): int(v) for k, v in zip(unique, counts)},
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def example_value_stats(examples):
    """从训练样本中提取价值标签，汇总其数量、分布和范围。"""
    return value_stats([example[2] for example in examples])


def format_value_stats(stats):
    """将样本价值统计格式化为简短日志，方便对比不同轮次。"""
    return (
        "count={count} unique={unique} mean={mean:.6f} std={std:.6f} "
        "min={min:.6f} max={max:.6f}"
    ).format(**stats)


def format_numeric_stats(stats):
    """将一般数值分布统计格式化，统一诊断日志字段的展示。"""
    return (
        "count={count} mean={mean:.6f} std={std:.6f} "
        "min={min:.6f} max={max:.6f}"
    ).format(**stats)


def log_example_value_stats(label, examples):
    """计算并记录一批训练样本的价值统计，附带调用方提供的阶段名称。"""
    log.info("%s value targets: %s", label, format_value_stats(example_value_stats(examples)))
