from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def make_tfidf(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int = 50000,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        strip_accents="unicode",
    )


def make_logreg(
    C: float = 2.0,
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1,
    )


def make_linear_svm(
    C: float = 1.0,
) -> LinearSVC:
    return LinearSVC(
        C=C,
        class_weight="balanced",
    )


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Metrics:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def metrics_to_dict(m: Metrics) -> Dict[str, float]:
    return asdict(m)