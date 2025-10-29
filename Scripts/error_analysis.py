# Scripts/error_analysis.py
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

DATA = Path("data/processed/train_mini.parquet")
MODEL_PKL = Path("models/xgb_model.pkl")      # if exists, we use it
OUT_DIR = Path("artifacts/error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_xy():
    df = pd.read_parquet(DATA)
    y = df["churn"].astype(int)
    X = df.drop(columns=["userId", "churn"])
    users = df["userId"].astype(str)
    return X, y, users

def main():
    X, y, users = load_xy()
    Xtr, Xte, ytr, yte, u_tr, u_te = train_test_split(
        X, y, users, test_size=0.2, random_state=42, stratify=y
    )

    # use trained model if present; otherwise fit a quick one
    if MODEL_PKL.exists():
        model = joblib.load(MODEL_PKL)
    else:
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42,
        )
        model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # metrics
    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "report": classification_report(yte, pred, output_dict=True),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # false negatives (actual 1, predicted 0) — most dangerous
    fn_mask = (yte == 1) & (pred == 0)
    fn_table = pd.DataFrame({
        "userId": u_te[fn_mask],
        "true_label": yte[fn_mask].values,
        "pred_prob": proba[fn_mask],
    }).sort_values("pred_prob", ascending=False)
    fn_table.to_csv(OUT_DIR / "false_negatives.csv", index=False)

    # false positives (actual 0, predicted 1)
    fp_mask = (yte == 0) & (pred == 1)
    fp_table = pd.DataFrame({
        "userId": u_te[fp_mask],
        "true_label": yte[fp_mask].values,
        "pred_prob": proba[fp_mask],
    }).sort_values("pred_prob", ascending=False)
    fp_table.to_csv(OUT_DIR / "false_positives.csv", index=False)

    print(f"[OK] Saved error analysis to {OUT_DIR}")

if __name__ == "__main__":
    main()