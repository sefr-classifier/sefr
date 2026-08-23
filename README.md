# SEFR

[![arXiv](https://img.shields.io/badge/arXiv-2006.04620-b31b1b.svg)](https://arxiv.org/abs/2006.04620)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**SEFR** (Scalable, Efficient, and Fast classifieR) is a linear-time binary classifier designed for ultra-low power and resource-constrained devices. Training and inference are a single pass over the data — no iterative optimization, no hyperparameters.

> Keshavarz, Abadeh, Rawassizadeh — [*SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices*](https://arxiv.org/abs/2006.04620) (2020)

<p align="center">
  <em>Train on a microcontroller. Classify in microseconds. Model size = n_features + 1.</em>
</p>

---

## Why SEFR?

| | SEFR | Typical sklearn pipeline |
|---|------|--------------------------|
| Hyperparameters | **None** | Often many |
| Training complexity | **O(n·m)** | Varies (trees, SVM, …) |
| Model size | **m + 1 floats** | Often much larger |
| On-device training | **Yes** ([Arduino demo](arduino-c)) | Usually no |

SEFR assigns each feature a weight in \([-1, 1]\) from class-conditional means (Eqs. 3–9 in the paper), then classifies with one dot product and a bias.

---

## Install

```bash
pip install numpy
pip install git+https://github.com/sefr-classifier/sefr.git
```

Or from a checkout:

```bash
git clone https://github.com/sefr-classifier/sefr.git
cd sefr
pip install -e .
```

---

## Quick start

Scale features to **[0, 1]** (Min–Max, as in the paper). Labels must be **0** and **1**.

```python
from sefr import SEFR
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SEFR()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

Multiclass (one-vs-all, paper Sec. 3.4):

```python
from sefr import SEFRMulticlass

clf = SEFRMulticlass().fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

Run the included example:

```bash
python examples/quickstart.py
```

### Original API (still supported)

The [original repository](https://github.com/sefr-classifier/sefr) exposed a single `SEFR.py` module:

```python
from SEFR import SEFR   # or: from sefr import SEFR

clf = SEFR()
clf.fit(X_train, y_train)
clf.predict(X_test)
```

After training, `clf.weights` and `clf.bias` hold the learned model (same as the C/Arduino port).

---

## Arduino / embedded

The file [`arduino-c`](arduino-c) is the reference C implementation used in the paper’s Arduino Uno experiments (Gisette subsamples, sub-millisecond inference). See Section 4.6 of the [paper](https://arxiv.org/abs/2006.04620).

---

## Project layout

```
sefr/
├── sefr/              # Python package
│   └── classifier.py  # SEFR & SEFRMulticlass
├── SEFR.py            # backward-compatible shim
├── arduino-c          # MCU reference implementation
├── examples/          # quickstart
└── tests/
```

---

## Citation

```bibtex
@article{keshavarz2020sefr,
  title={SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices},
  author={Keshavarz, Hamidreza and Abadeh, Mohammad Saniee and Rawassizadeh, Reza},
  journal={arXiv preprint arXiv:2006.04620},
  year={2020}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Authors

- Hamidreza Keshavarz — keshavarz.h@modares.ac.ir  
- Mohammad Saniee Abadeh  
- Reza Rawassizadeh  
