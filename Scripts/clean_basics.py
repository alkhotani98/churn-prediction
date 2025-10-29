# Scripts/clean_basics.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "customer_churn_mini.json"
OUT = ROOT / "data" / "processed" / "clean_mini.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_json(RAW, lines=True)

    # ts -> datetime
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    # valid users
    df = df[df["userId"].notnull()]
    df = df[df["userId"] != ""]
    df["userId"] = df["userId"].astype(str)

    df.to_parquet(OUT, index=False)
    print("clean_basics: rows:", len(df))
    print("clean_basics: saved:", OUT)

if __name__ == "__main__":
    main()
