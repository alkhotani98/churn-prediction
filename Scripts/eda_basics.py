import pandas as pd

df = pd.read_parquet("data/processed/clean_mini.parquet")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\n--- Unique pages ---")
print(df["page"].unique()[:20])
