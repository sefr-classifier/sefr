#!/usr/bin/env python3
"""Minimal SEFR usage example (binary classification on Sonar)."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

from sefr import SEFR

X, y = load_breast_cancer(return_X_y=True)
X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

clf = SEFR()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"F1:         {f1_score(y_test, pred):.3f}")
print(f"Model size: {clf.weights.size + 1} floats (weights + bias)")
