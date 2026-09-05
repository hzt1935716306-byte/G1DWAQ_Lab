"""Simulator-independent metadata math for matched continuous-plane terrain."""

from __future__ import annotations

import math

import numpy as np


MATCHED_ROWS = 10
MATCHED_COLS = 20
MATCHED_SEED = 42
MATCHED_SLOPE_RANGE = (0.0, 0.2679491924311227)  # tan(15 deg)


def slope_coefficient(
    difficulty: float,
    slope_range: tuple[float, float] = MATCHED_SLOPE_RANGE,
    *,
    inverted: bool = False,
) -> float:
    low, high = (float(value) for value in slope_range)
    value = low + float(difficulty) * (high - low)
    return -value if inverted else value


def column_types(num_cols: int = MATCHED_COLS) -> tuple[str, ...]:
    proportions = np.asarray((0.4, 0.3, 0.3), dtype=np.float64)
    cumulative = np.cumsum(proportions / proportions.sum())
    labels = ("flat", "uphill", "downhill")
    return tuple(
        labels[int(np.min(np.where(index / num_cols + 0.001 < cumulative)[0]))]
        for index in range(num_cols)
    )


def slope_table(
    seed: int = MATCHED_SEED,
    *,
    num_rows: int = MATCHED_ROWS,
    num_cols: int = MATCHED_COLS,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    labels = column_types(num_cols)
    table = np.zeros((num_rows, num_cols), dtype=np.float64)
    lower, upper = (float(value) for value in difficulty_range)
    for col, label in enumerate(labels):
        for row in range(num_rows):
            difficulty = (row + rng.uniform()) / num_rows
            difficulty = lower + (upper - lower) * difficulty
            if label != "flat":
                table[row, col] = math.atan(
                    slope_coefficient(difficulty, inverted=(label == "downhill"))
                )
    return table


def upward_normal(alpha: float) -> np.ndarray:
    return np.asarray((-math.sin(alpha), 0.0, math.cos(alpha)), dtype=np.float64)


__all__ = [
    "MATCHED_COLS",
    "MATCHED_ROWS",
    "MATCHED_SEED",
    "MATCHED_SLOPE_RANGE",
    "column_types",
    "slope_coefficient",
    "slope_table",
    "upward_normal",
]
