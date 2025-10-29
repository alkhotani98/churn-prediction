import pandas as pd

df = pd.read_parquet("data/processed/clean_mini.parquet")

df = df[df["userId"] != 0]
# Simple features for each user
features = (
    df.groupby("userId")
    .agg(
        songs_played=("page", lambda x: (x == "NextSong").sum()),
        thumbs_up=("page", lambda x: (x == "Thumbs Up").sum()),
        thumbs_down=("page", lambda x: (x == "Thumbs Down").sum()),
        add_playlist=("page", lambda x: (x == "Add to Playlist").sum()),
        days_active=("ts", lambda x: x.dt.date.nunique()),
    )
    .reset_index()
)

print(features.head())
print("Rows:", len(features))
