"""Build the scorecard on loans from 2010-2013, test it on 2014, save results and charts.

    python3 src/run.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from prepare import CATEGORICAL, NUMERIC, load
from scorecard import Scorecard
from validation import calibration, gini, ks, psi, psi_categories

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"

NAMES = {"annual_inc": "dochód roczny", "loan_to_income": "kwota kredytu / dochód", "dti": "zadłużenie / dochód (DTI)",
         "revol_util": "wykorzystanie limitów kart", "home_ownership": "mieszkanie (własne, kredyt, najem)",
         "inq_last_6mths": "zapytania kredytowe (6 mies.)", "purpose": "cel kredytu",
         "credit_history_months": "długość historii kredytowej", "emp_years": "staż pracy",
         "revol_bal": "saldo na kartach", "total_acc": "liczba rachunków", "loan_amnt": "kwota kredytu",
         "verification_status": "weryfikacja dochodu", "months_since_delinq": "mies. od ostatniego opóźnienia",
         "pub_rec": "wpisy w rejestrach", "pub_rec_bankruptcies": "upadłości", "open_acc": "otwarte rachunki",
         "delinq_2yrs": "opóźnienia (2 lata)"}

DARK, GREY, BLUE, RED = "#1f2a33", "#7b8794", "#2459d6", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e8ec", "axes.axisbelow": True,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def gradient_boosting(dev, oot):
    """Challenger model: gradient boosting on the raw variables."""
    cols = NUMERIC + CATEGORICAL
    Xd, Xo = dev[cols].copy(), oot[cols].copy()
    for c in CATEGORICAL:
        cats = pd.Categorical(Xd[c]).categories
        Xd[c] = pd.Categorical(Xd[c], categories=cats).codes
        Xo[c] = pd.Categorical(Xo[c], categories=cats).codes
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, categorical_features=CATEGORICAL,
                                           random_state=0)
    model.fit(Xd, dev["bad"])
    return model.predict_proba(Xo)[:, 1]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    loans = load()
    dev = loans[loans.year <= 2013]
    oot = loans[loans.year == 2014]

    sc = Scorecard(NUMERIC, CATEGORICAL).fit(dev, dev["bad"])
    pd_dev = sc.predict_pd(dev)
    pd_oot = sc.predict_pd(oot)
    score_dev = sc.score(dev)
    score_oot = sc.score(oot)
    pd_gb = gradient_boosting(dev, oot)

    results = {
        "scorecard_dev": {"gini": gini(dev.bad, pd_dev), "ks": ks(dev.bad, pd_dev)},
        "scorecard_oot": {"gini": gini(oot.bad, pd_oot), "ks": ks(oot.bad, pd_oot)},
        "gradient_boosting_oot": {"gini": gini(oot.bad, pd_gb), "ks": ks(oot.bad, pd_gb)},
        "lending_club_grade_oot": {"gini": gini(oot.bad, oot.grade_num), "ks": ks(oot.bad, oot.grade_num)},
    }
    results = {k: {m: round(v, 3) for m, v in r.items()} for k, r in results.items()}

    calib_oot = calibration(oot.bad, pd_oot)
    calib_dev = calibration(dev.bad, pd_dev)

    # stability: score distribution and each variable, 2014 vs 2010-2013
    edges = np.unique(np.quantile(score_dev, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    psi_score = psi(score_dev, score_oot, edges)
    psi_vars = {c: psi_categories(sc.binners[c](dev[c]), sc.binners[c](oot[c])) for c in sc.features}

    # loss given default and exposure at default, measured on bad loans from 2010-2013
    bad_dev = dev[dev.bad == 1]
    ead = (bad_dev.funded_amnt - bad_dev.total_rec_prncp).clip(lower=0)
    lgd = 1 - bad_dev.recoveries.sum() / ead.sum()
    ead_share = ead.sum() / bad_dev.funded_amnt.sum()

    # expected loss in 2014 = PD x LGD x EAD, compared with the loss that really happened
    el_pred = float((pd_oot * lgd * ead_share * oot.funded_amnt).sum())
    bad_oot = oot[oot.bad == 1]
    loss_real = float(((bad_oot.funded_amnt - bad_oot.total_rec_prncp).clip(lower=0) - bad_oot.recoveries).sum())

    # return on money lent in 2014 (money received - money lent) / money lent, by risk group
    oot = oot.assign(pd=pd_oot, profit=oot.total_pymnt - oot.funded_amnt)
    oot["decile"] = pd.qcut(oot["pd"].rank(method="first"), 10, labels=False) + 1
    by_decile = oot.groupby("decile").agg(bad_rate=("bad", "mean"), interest=("int_rate", "mean"),
                                          profit=("profit", "sum"), lent=("funded_amnt", "sum"))
    by_decile["return"] = by_decile["profit"] / by_decile["lent"]

    # inside each Lending Club grade (same price): safer and riskier half according to the scorecard
    oot["half"] = np.where(oot.groupby("grade")["pd"].rank(pct=True) <= 0.5, "safer", "riskier")
    by_grade = oot.groupby(["grade", "half"]).agg(loans=("bad", "size"), bad_rate=("bad", "mean"),
                                                  profit=("profit", "sum"), lent=("funded_amnt", "sum"))
    by_grade["return"] = by_grade["profit"] / by_grade["lent"]

    summary = {
        "development": {"years": "2010-2013", "loans": len(dev), "bad_rate": round(dev.bad.mean(), 4)},
        "out_of_time": {"year": 2014, "loans": len(oot), "bad_rate": round(oot.bad.mean(), 4),
                        "mean_pd": round(float(pd_oot.mean()), 4)},
        "information_value": sc.iv.round(4).to_dict(),
        "features": sc.features,
        "coefficients": dict(zip(sc.features, np.round(sc.model.coef_[0], 3))),
        "results": results,
        "calibration_oot": calib_oot.round(4).to_dict(orient="index"),
        "psi_score": round(psi_score, 4),
        "psi_variables": {k: round(v, 4) for k, v in psi_vars.items()},
        "lgd": round(float(lgd), 3),
        "ead_share_of_amount": round(float(ead_share), 3),
        "expected_loss_2014_usd": round(el_pred),
        "real_loss_2014_usd": round(loss_real),
        "return_all_2014": round(float(oot.profit.sum() / oot.funded_amnt.sum()), 4),
        "by_decile_2014": by_decile[["bad_rate", "interest", "return"]].round(4).to_dict(orient="index"),
        "by_grade_2014": {f"{g}_{h}": r for (g, h), r in
                          by_grade[["loans", "bad_rate", "return"]].round(4).to_dict(orient="index").items()},
    }
    summary["coefficients"] = {k: float(v) for k, v in summary["coefficients"].items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    sc.points_table().to_csv(OUT / "scorecard_points.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 1. information value
    iv = sc.iv[sc.iv > 0.001].sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh([NAMES[c] for c in iv.index], iv.values,
            color=[BLUE if c in sc.features else GREY for c in iv.index])
    ax.axvline(0.02, color=DARK, lw=0.8, ls="--")
    ax.set_xlabel("Information value (IV)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Siła zmiennych (niebieskie weszły do modelu)", loc="left")
    fig.savefig(FIG / "information_value.png")
    plt.close(fig)

    # 2. Gini of the models
    labels = ["Karta scoringowa\n2010–2013", "Karta scoringowa\n2014", "Gradient boosting\n2014",
              "Rating Lending Club\n2014"]
    keys = ["scorecard_dev", "scorecard_oot", "gradient_boosting_oot", "lending_club_grade_oot"]
    vals = [results[k]["gini"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.bar(labels, vals, color=[GREY, BLUE, DARK, GREY], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center")
    ax.set_ylabel("Gini [%]")
    ax.grid(axis="x", visible=False)
    ax.set_title("Moc dyskryminacyjna modeli", loc="left")
    fig.savefig(FIG / "gini.png")
    plt.close(fig)

    # 3. calibration
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = calib_oot.index
    ax.plot(x, calib_dev["bad_rate"] * 100, "o-", color=GREY, label="2010–2013: rzeczywisty")
    ax.plot(x, calib_oot["bad_rate"] * 100, "o-", color=RED, label="2014: rzeczywisty")
    ax.plot(x, calib_oot["pd_mean"] * 100, "s--", color=BLUE, label="2014: przewidywany przez model")
    ax.set_xticks(x)
    ax.set_xlabel("Grupa klientów (1 = najbezpieczniejsi, 10 = najbardziej ryzykowni)")
    ax.set_ylabel("Odsetek niespłaconych [%]")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Kalibracja: przewidywane i rzeczywiste PD", loc="left")
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)

    # 4. return by risk group
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.2))
    a1.bar(by_decile.index, by_decile["return"] * 100, color=BLUE, width=0.7)
    for i, v in zip(by_decile.index, by_decile["return"] * 100):
        a1.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8.5)
    a1.set_xticks(by_decile.index)
    a1.set_xlabel("Grupa klientów wg modelu (10 = najbardziej ryzykowni)")
    a1.set_ylabel("Zwrot z pożyczonych pieniędzy [%]")
    a1.grid(axis="x", visible=False)
    a1.set_title("Zwrot z kredytów udzielonych w 2014", loc="left")

    grades = [g for g in "ABCDE"]
    x = np.arange(len(grades))
    safer = [by_grade.loc[(g, "safer"), "bad_rate"] * 100 for g in grades]
    riskier = [by_grade.loc[(g, "riskier"), "bad_rate"] * 100 for g in grades]
    a2.bar(x - 0.2, safer, width=0.38, color=BLUE, label="bezpieczniejsza połowa wg modelu")
    a2.bar(x + 0.2, riskier, width=0.38, color=RED, label="ryzykowniejsza połowa wg modelu")
    a2.set_xticks(x, grades)
    a2.set_xlabel("Rating Lending Club (podobna cena kredytu)")
    a2.set_ylabel("Odsetek niespłaconych [%]")
    a2.legend(frameon=False, fontsize=8.5)
    a2.grid(axis="x", visible=False)
    a2.set_title("Model rozdziela klientów w tym samym ratingu", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "return_by_risk.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
