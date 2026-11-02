"""Scorecard: binning, WoE, information value, logistic regression and points.

WoE (weight of evidence) of a bin = ln(share of goods in the bin / share of bads in the bin).
Positive WoE means the bin is safer than average.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MISSING = "missing"


def numeric_bins(x, y, max_bins=10, min_share=0.05):
    """Cut points for a numeric variable: quantile bins, merged until the bad rate is monotonic."""
    ok = x.notna()
    qs = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, max_bins + 1)[1:-1]))
    edges = [-np.inf, *qs, np.inf]

    def table(edges):
        b = pd.cut(x[ok], edges, right=True)
        return y[ok].groupby(b, observed=False).agg(["mean", "count"])

    # merge bins that are too small
    t = table(edges)
    while len(edges) > 2 and t["count"].min() < min_share * ok.sum():
        i = int(np.argmin(t["count"].to_numpy()))
        edges.pop(i + 1 if i < len(t) - 1 else i)
        t = table(edges)

    # merge neighbours until the bad rate only goes up or only goes down
    direction = np.sign(np.corrcoef(x[ok].rank(), y[ok])[0, 1]) or 1
    while len(edges) > 2:
        rates = t["mean"].to_numpy()
        steps = np.diff(rates) * direction
        if (steps >= 0).all():
            break
        i = int(np.argmin(steps))           # the worst step: merge bins i and i+1
        edges.pop(i + 1)
        t = table(edges)
    return edges


def assign_bins(x, edges):
    out = pd.cut(x, edges, right=True).astype(str)
    return out.where(x.notna(), MISSING)


def category_bins(x, y, min_share=0.01):
    """Rare categories are grouped as 'other'."""
    share = x.value_counts(normalize=True)
    keep = set(share[share >= min_share].index)
    return lambda s: s.where(s.isin(keep), "other").astype(str)


def bin_order(name):
    """Sort bins like '(-inf, 5.0]' by their left edge; 'missing' goes last."""
    if name == MISSING:
        return np.inf
    try:
        return float(name.strip("([").split(",")[0])
    except ValueError:
        return 0.0


def woe_table(bins, y):
    t = y.groupby(bins).agg(bad="sum", count="count")
    t = t.loc[sorted(t.index, key=bin_order)]
    t["good"] = t["count"] - t["bad"]
    # +0.5 avoids dividing by zero in bins without bads or goods
    t["good_share"] = (t["good"] + 0.5) / (t["good"].sum() + 0.5)
    t["bad_share"] = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5)
    t["woe"] = np.log(t["good_share"] / t["bad_share"])
    t["iv"] = (t["good_share"] - t["bad_share"]) * t["woe"]
    t["bad_rate"] = t["bad"] / t["count"]
    return t


class Scorecard:
    def __init__(self, numeric, categorical, min_iv=0.02, max_corr=0.7):
        self.numeric = numeric
        self.categorical = categorical
        self.min_iv = min_iv
        self.max_corr = max_corr

    def fit(self, X, y):
        self.binners, self.tables = {}, {}
        for col in self.numeric:
            edges = numeric_bins(X[col], y)
            self.binners[col] = lambda s, e=edges: assign_bins(s, e)
        for col in self.categorical:
            self.binners[col] = category_bins(X[col], y)
        for col, f in self.binners.items():
            self.tables[col] = woe_table(f(X[col]), y)
        self.iv = pd.Series({c: t["iv"].sum() for c, t in self.tables.items()}).sort_values(ascending=False)

        # keep variables with enough IV; of two strongly correlated ones keep the one with higher IV
        W = self.woe(X, list(self.iv.index))
        chosen = []
        for col in self.iv.index:
            if self.iv[col] < self.min_iv:
                continue
            if all(abs(np.corrcoef(W[col], W[c])[0, 1]) < self.max_corr for c in chosen):
                chosen.append(col)

        # every coefficient should have the same sign (higher WoE = lower risk), otherwise drop it
        while True:
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(W[chosen], y)
            coefs = pd.Series(self.model.coef_[0], index=chosen)
            wrong = coefs[coefs > 0]
            if wrong.empty:
                break
            chosen.remove(wrong.idxmax())
        self.features = chosen
        return self

    def woe(self, X, cols=None):
        cols = cols or self.features
        out = {}
        for col in cols:
            b = self.binners[col](X[col])
            # a bin not seen in training gets WoE 0 (average risk)
            out[col] = b.map(self.tables[col]["woe"]).astype(float).fillna(0.0)
        return pd.DataFrame(out, index=X.index)

    def predict_pd(self, X):
        return self.model.predict_proba(self.woe(X))[:, 1]

    def points_table(self, base_points=600, base_odds=50, pdo=20):
        """Points for each bin: 600 points means good:bad odds of 50:1, +20 points doubles the odds."""
        factor = pdo / np.log(2)
        offset = base_points - factor * np.log(base_odds)
        n = len(self.features)
        rows = []
        for col, b in zip(self.features, self.model.coef_[0]):
            for bin_name, r in self.tables[col].iterrows():
                pts = -(b * r["woe"] + self.model.intercept_[0] / n) * factor + offset / n
                rows.append({"variable": col, "bin": bin_name, "count": int(r["count"]),
                             "bad_rate": round(r["bad_rate"], 4), "woe": round(r["woe"], 3),
                             "points": round(pts)})
        return pd.DataFrame(rows)

    def score(self, X, base_points=600, base_odds=50, pdo=20):
        factor = pdo / np.log(2)
        offset = base_points - factor * np.log(base_odds)
        p = self.predict_pd(X)
        return offset + factor * np.log((1 - p) / p)
