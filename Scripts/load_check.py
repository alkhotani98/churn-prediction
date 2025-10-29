from pathlib import Path

import pandas as pd

path = Path("data/raw/customer_churn_mini.json")

# Try JSON Lines first
try:
    df = pd.read_json(path, lines=True)
except ValueError:
    # Fallback if it's a regular JSON array
    df = pd.read_json(path)

print("Loaded shape:", df.shape)
print("Columns:", list(df.columns)[:20])
print(df.head(3))
