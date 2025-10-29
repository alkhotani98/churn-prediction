from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(ref, cur, bins=10):
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    r_hist, _ = np.histogram(ref, bins=qs)
    c_hist, _ = np.histogram(cur, bins=qs)
    r_pct = np.clip(r_hist / max(1, r_hist.sum()), 1e-6, 1)
    c_pct = np.clip(c_hist / max(1, c_hist.sum()), 1e-6, 1)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


def ks(ref, cur):
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    return float(ks_2samp(ref, cur).pvalue)


def build_features(df):
    df = df[df["userId"].notnull() & (df["userId"] != "")]
    df["userId"] = df["userId"].astype(str)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    feats = (
        df.groupby("userId")
        .agg(
            songs_played=("page", lambda x: (x == "NextSong").sum()),
            thumbs_up=("page", lambda x: (x == "Thumbs Up").sum()),
            thumbs_down=("page", lambda x: (x == "Thumbs Down").sum()),
            add_playlist=("page", lambda x: (x == "Add to Playlist").sum()),
            days_active=("ts", lambda x: x.dt.date.nunique()),
            last_seen=("ts", "max"),
        )
        .reset_index()
    )
    max_ts = df["ts"].max()
    feats["recency_days"] = (max_ts - feats["last_seen"]).dt.days.fillna(999)
    return feats.drop(columns=["last_seen"])


# ----- load data -----
ref = pd.read_parquet("data/processed/train_mini.parquet")
cur = pd.read_parquet("data/processed/clean_mini.parquet")

# if data is raw events (has "page"), build features
if "page" in ref.columns:
    ref = build_features(ref)
if "page" in cur.columns:
    cur = build_features(cur)

cols = ["songs_played", "thumbs_up", "thumbs_down", "add_playlist", "days_active", "recency_days"]
rows = []
for c in cols:
    r = ref[c].astype(float).to_numpy()
    k = cur[c].astype(float).to_numpy()
    rows.append(
        {
            "feature": c,
            "mean_ref": r.mean(),
            "mean_cur": k.mean(),
            "ks_pvalue": ks(r, k),
            "psi": psi(r, k),
        }
    )

report = pd.DataFrame(rows)
Path("artifacts").mkdir(exist_ok=True)
html = "artifacts/data_drift_simple.html"
report.to_html(html, index=False)
print(f"✅ Drift report saved: {html}")
