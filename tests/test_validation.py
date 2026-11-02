import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scorecard import numeric_bins, woe_table
from validation import gini, ks, psi


def test_gini_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1
    assert gini(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0


def test_ks_perfect_split():
    y = np.array([0, 0, 1, 1])
    assert ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1


def test_psi_same_sample_is_zero():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x, np.array([-np.inf, -1, 0, 1, np.inf])) < 1e-9


def test_bins_have_monotonic_bad_rate():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=20000))
    y = pd.Series((rng.random(20000) < 1 / (1 + np.exp(-x))).astype(int))
    edges = numeric_bins(x, y)
    rates = woe_table(pd.cut(x, edges).astype(str), y)["bad_rate"].to_numpy()
    assert (np.diff(rates) >= 0).all()
