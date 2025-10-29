# Scripts/make_dataset.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "clean_mini.parquet"
OUT_PATH = ROOT / "data" / "processed" / "train_mini.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

INACTIVE_DAYS = 30
FEAT_WINDOW_DAYS = 30

def main():
    df = pd.read_parquet(IN_PATH)

    # valid users (defensive)
    df = df[df["userId"].notnull()]
    df = df[df["userId"] != ""]
    df["userId"] = df["userId"].astype(str)

    # time split
    global_last = df["ts"].max()
    split_time = global_last - pd.Timedelta(days=INACTIVE_DAYS)

    # past window for features
    win_start = split_time - pd.Timedelta(days=FEAT_WINDOW_DAYS)
    past = df[(df["ts"] >= win_start) & (df["ts"] < split_time)].copy()

    feats = (
        past.groupby("userId")
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

    feats["recency_days"] = (split_time - feats["last_seen"]).dt.days.fillna(999)
    feats = feats.drop(columns=["last_seen"])

    # labels from future window
    future = df[df["ts"] >= split_time].copy()
    canceled = set(
        future.loc[future["page"] == "Cancellation Confirmation", "userId"].astype(str)
    )
    active_future = (
        future.groupby("userId")["ts"].max().rename("last_future").reset_index()
    )

    labels = feats[["userId"]].merge(active_future, on="userId", how="left")
    labels["churn"] = 0
    labels.loc[labels["userId"].isin(canceled), "churn"] = 1
    labels.loc[labels["last_future"].isna(), "churn"] = 1
    labels = labels[["userId", "churn"]].astype({"churn": int})

    train = feats.merge(labels, on="userId", how="inner")
    train.to_parquet(OUT_PATH, index=False)

    print("make_dataset: users in features:", len(feats))
    print("make_dataset: final rows:", len(train))
    print("make_dataset: churn dist:\n", train["churn"].value_counts())
    print("make_dataset: saved:", OUT_PATH)

if __name__ == "__main__":
    main()