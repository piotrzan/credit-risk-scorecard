"""Read data/loans.csv, define the target and build the application features."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "loans.csv"

BAD = ["Charged Off", "Default", "Late (31-120 days)"]
GOOD = ["Fully Paid"]

NUMERIC = ["loan_amnt", "annual_inc", "loan_to_income", "emp_years", "dti", "delinq_2yrs",
           "credit_history_months", "inq_last_6mths", "months_since_delinq", "open_acc", "pub_rec",
           "revol_bal", "revol_util", "total_acc", "pub_rec_bankruptcies"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]


def load(path=DATA):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run src/download_data.py first")
    d = pd.read_csv(path, parse_dates=["issue_d", "earliest_cr_line"])

    # 36-month loans with a final status only (loans still being repaid are left out)
    d = d[d["term"] == 36]
    d = d[d["loan_status"].isin(BAD + GOOD)].copy()
    d["bad"] = d["loan_status"].isin(BAD).astype(int)
    d["year"] = d["issue_d"].dt.year

    d["loan_to_income"] = d["loan_amnt"] / d["annual_inc"].replace(0, np.nan)
    d["emp_years"] = (d["emp_length"].astype(str).str.replace("< 1", "0", regex=False)
                      .str.extract(r"(\d+)")[0].astype(float))
    d["credit_history_months"] = ((d["issue_d"] - d["earliest_cr_line"]).dt.days / 30.44).round()
    # no delinquency in the past is a separate group, not a missing value
    d["months_since_delinq"] = d["mths_since_last_delinq"]

    d["home_ownership"] = d["home_ownership"].replace({"ANY": "OTHER", "NONE": "OTHER"})
    d["grade_num"] = d["grade"].map({g: i for i, g in enumerate("ABCDEFG", start=1)})
    return d.reset_index(drop=True)


if __name__ == "__main__":
    d = load()
    print(d.groupby("year")["bad"].agg(["count", "mean"]).round(3))
