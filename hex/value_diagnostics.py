import logging

import numpy as np


log = logging.getLogger(__name__)


def value_stats(values):
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
    return value_stats([example[2] for example in examples])


def format_value_stats(stats):
    return (
        "count={count} unique={unique} mean={mean:.6f} std={std:.6f} "
        "min={min:.6f} max={max:.6f}"
    ).format(**stats)


def format_numeric_stats(stats):
    return (
        "count={count} mean={mean:.6f} std={std:.6f} "
        "min={min:.6f} max={max:.6f}"
    ).format(**stats)


def log_example_value_stats(label, examples):
    log.info("%s value targets: %s", label, format_value_stats(example_value_stats(examples)))
