"""Measures used to validate a PD model."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def gini(y, risk):
    """Gini = 2 * AUC - 1. `risk` should be higher for riskier clients."""
    return 2 * roc_auc_score(y, risk) - 1


def ks(y, risk):
    """Largest gap between the cumulative distributions of bads and goods."""
    y = np.asarray(y)
    risk = np.asarray(risk)
    order = np.argsort(-risk)
    cum_bad = np.cumsum(y[order]) / y.sum()
    cum_good = np.cumsum(1 - y[order]) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def calibration(y, pd_pred, n_groups=10):
    """Predicted vs observed bad rate in groups of clients sorted by PD, with a binomial test."""
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_pred)})
    df["group"] = pd.qcut(df["pd"].rank(method="first"), n_groups, labels=False) + 1
    t = df.groupby("group").agg(count=("y", "size"), bads=("y", "sum"), pd_mean=("pd", "mean"))
    t["bad_rate"] = t["bads"] / t["count"]
    # p-value of observing this many bads if the predicted PD were right
    t["p_value"] = [stats.binomtest(int(r.bads), int(r.count), r.pd_mean).pvalue for r in t.itertuples()]
    return t


def psi(expected, actual, edges):
    """Population stability index between two samples, using the same bins."""
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categories(expected, actual):
    """PSI for a variable that is already split into bins (e.g. WoE bins)."""
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = e.reindex(idx).fillna(0).clip(lower=1e-6)
    a = a.reindex(idx).fillna(0).clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())
