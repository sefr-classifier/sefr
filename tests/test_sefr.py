import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from sefr import SEFR, SEFRMulticlass


def test_binary_fit_predict():
    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    X = MinMaxScaler().fit_transform(X)
    y = (y > 0).astype(np.int32)

    clf = SEFR()
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (200,)
    assert set(np.unique(pred)) <= {0, 1}
    assert clf.weights.shape == (10,)
    assert isinstance(clf.bias, float)


def test_original_api_names():
    """Original repo used train_predictors / train_target argument names."""
    X = np.random.rand(50, 4).astype(np.float64)
    y = np.array([0] * 25 + [1] * 25, dtype=np.int32)
    clf = SEFR()
    clf.fit(train_predictors=X, train_target=y)
    assert clf.predict(X).shape == (50,)


def test_multiclass():
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=5, n_classes=3, random_state=1
    )
    X = MinMaxScaler().fit_transform(X)
    clf = SEFRMulticlass().fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (150,)
    assert len(np.unique(pred)) == 3


def test_sefr_py_shim():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("SEFR_shim", root / "SEFR.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SEFR is SEFR
