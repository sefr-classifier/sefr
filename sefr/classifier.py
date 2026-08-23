"""
SEFR: Scalable, Efficient, and Fast classifieR
==============================================

Implementation of the algorithm in:

  Keshavarz, Abadeh, Rawassizadeh (2020)
  "SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices"
  arXiv:2006.04620

Features should be scaled to [0, 1] (or [0, n]) before fit/predict.
Binary labels must be 0 / 1.
"""

from __future__ import annotations

import numpy as np


class SEFR:
    """Binary SEFR classifier.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        Feature weights in [-1, 1] (Eq. 5).
    bias : float
        Decision threshold (Eq. 9).
    """

    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def fit(self, train_predictors, train_target):
        """Train on binary labels in {0, 1}."""
        X = np.asarray(train_predictors, dtype=np.float64)
        y = np.asarray(train_target, dtype=np.int32).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("Binary SEFR requires labels in {0, 1}")

        pos_labels = y == 1
        neg_labels = ~pos_labels
        if not np.any(pos_labels) or not np.any(neg_labels):
            raise ValueError("Both classes must be present in the training set")

        avg_pos = np.mean(X[pos_labels], axis=0)
        avg_neg = np.mean(X[neg_labels], axis=0)
        self.weights = (avg_pos - avg_neg) / (avg_pos + avg_neg + 1e-7)

        scores = X @ self.weights
        pos_label_count = int(np.count_nonzero(y))
        neg_label_count = y.shape[0] - pos_label_count
        pos_score_avg = float(np.mean(scores[pos_labels]))
        neg_score_avg = float(np.mean(scores[neg_labels]))
        self.bias = (
            neg_label_count * pos_score_avg + pos_label_count * neg_score_avg
        ) / (neg_label_count + pos_label_count)
        return self

    def decision_function(self, test_predictors):
        """Signed score; positive values favor class 1."""
        X = np.asarray(test_predictors, dtype=np.float64)
        return X @ self.weights - self.bias

    def predict(self, test_predictors):
        """Predict binary labels in {0, 1}."""
        X = np.asarray(test_predictors, dtype=np.float64)
        scores = X @ self.weights
        return np.where(scores <= self.bias, 0, 1).astype(np.int32)


class SEFRMulticlass:
    """One-against-all SEFR (paper Sec. 3.4)."""

    def __init__(self):
        self.classes_ = None
        self.models_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("Need at least two classes")

        if self.classes_.size == 2:
            mapping = {self.classes_[0]: 0, self.classes_[1]: 1}
            y_bin = np.vectorize(mapping.get)(y)
            self.models_ = [SEFR().fit(X, y_bin)]
            return self

        self.models_ = []
        for c in self.classes_:
            y_bin = np.where(y == c, 0, 1).astype(np.int32)
            self.models_.append(SEFR().fit(X, y_bin))
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.classes_.size == 2:
            return self.models_[0].decision_function(X)
        cols = [m.decision_function(X) for m in self.models_]
        return np.column_stack(cols)

    def predict(self, X):
        if self.classes_.size == 2:
            pred01 = self.models_[0].predict(X)
            return np.where(pred01 == 0, self.classes_[0], self.classes_[1])
        scores = self.decision_function(X)
        return self.classes_[np.argmin(scores, axis=1)]
