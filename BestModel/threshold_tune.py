from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class ThresholdResult:
    threshold: float
    f1: float


def tune_threshold(y_true: Iterable[int], y_prob: Iterable[float]) -> ThresholdResult:
    y_true = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(list(y_prob), dtype=float)

    best_t = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return ThresholdResult(threshold=best_t, f1=best_f1)