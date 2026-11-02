"""Download Lending Club loans issued in 2010-2014 and save the columns used in the project.

    python3 src/download_data.py

Source: github.com/kuhnrl30/LendingClubData (Lending Club loan data saved as R files, MIT license).
Saves data/loans.csv. Needs the pyreadr package to read R files.
"""
import urllib.request
from pathlib import Path

import pandas as pd
import pyreadr

URL = "https://raw.githubusercontent.com/kuhnrl30/LendingClubData/master/data/LendingClubData_{}.rda"
YEARS = [2010, 2011, 2012, 2013, 2014]
DATA = Path(__file__).resolve().parents[1] / "data"

COLUMNS = [
    # known when the client applies
    "issue_d", "term", "loan_amnt", "annual_inc", "emp_length", "home_ownership",
    "verification_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "pub_rec_bankruptcies",
    # Lending Club's own rating and price, used only for comparison
    "grade", "int_rate",
    # what happened to the loan
    "loan_status", "funded_amnt", "total_pymnt", "total_rec_prncp", "recoveries",
]


def main():
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in YEARS:
        path = raw / f"LendingClubData_{year}.rda"
        if not path.exists():
            print("downloading", year)
            urllib.request.urlretrieve(URL.format(year), path)
        df = next(iter(pyreadr.read_r(path).values()))
        frames.append(df[COLUMNS])
    loans = pd.concat(frames, ignore_index=True)
    loans.to_csv(DATA / "loans.csv", index=False)
    print(len(loans), "loans saved to", DATA / "loans.csv")


if __name__ == "__main__":
    main()
