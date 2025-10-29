# Scripts/retrain.py
"""
End-to-end periodic retraining:
1) Run clean_basics.py  -> data/processed/clean_mini.parquet
2) Run make_dataset.py  -> data/processed/train_mini.parquet
3) Train XGBoost        -> models/xgb_model.pkl
4) Log to MLflow and register model "xgb_churn_model"
"""

from __future__ import annotations
from pathlib import Path
import sys
import runpy
import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import subprocess

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed" / "train_mini.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODEL_DIR / "xgb_model.pkl"

MLRUNS = ROOT / "mlruns"  # local tracking folder
MLRUNS.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def run_script(rel_path: str) -> None:
    """Execute another script relative to project root."""
    script = ROOT / rel_path
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    # Ensure scripts can import modules placed alongside them
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")

# ---------- Pipeline ----------
def main() -> None:
    print("== Retraining pipeline START ==")

    # 1) Clean raw -> clean_mini.parquet
    run_script("Scripts/clean_basics.py")

    # 2) Build features + labels -> train_mini.parquet
    run_script("Scripts/make_dataset.py")

    # 3) Load training table
    if not DATA_PROC.exists():
        raise FileNotFoundError(f"Expected dataset not found: {DATA_PROC}")
    df = pd.read_parquet(DATA_PROC)

    y = df["churn"].astype(int)
    X = df.drop(columns=["userId", "churn"])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model = XGBClassifier(**params)
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    print("ROC-AUC:", auc)
    print(classification_report(yte, (proba >= 0.5).astype(int)))

    # 4) Save model file used by FastAPI
    joblib.dump(model, MODEL_OUT)
    print(f"Saved model -> {MODEL_OUT}")

    # 5) Log to MLflow (local ./mlruns) and register
    mlflow.set_tracking_uri(MLRUNS.as_uri())
    mlflow.set_experiment("churn_retrain")

    with mlflow.start_run(run_name="retrain_xgb"):
        # params and metrics
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("data_path", str(DATA_PROC))
        mlflow.log_metric("roc_auc", auc)

        # log pickle artifact and an MLflow model flavor
        mlflow.log_artifact(str(MODEL_OUT))
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="xgb_churn_model",
        )

    # (optional) move latest registered version to STAGING
    try:
        client = MlflowClient()
        versions = client.get_latest_versions("xgb_churn_model")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                name="xgb_churn_model",
                version=latest.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"Registered xgb_churn_model v{latest.version} -> STAGING")
    except Exception as e:
        print("WARN: could not transition model stage:", e)

    print("== Retraining pipeline DONE ==")

        # --- Run concept drift check automatically ---
    try:
        print("[INFO] Running concept drift check...")
        result = subprocess.run(
            ["python", "Scripts/monitor_concept_drift.py"],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode == 3:
            print("[ALERT] Concept drift detected. Consider promoting previous model or reviewing data.")
        else:
            print("[OK] No concept drift detected.")
    except Exception as e:
        print("[WARN] Could not execute concept drift script:", e)
        

if __name__ == "__main__":
    # run relative to project root to keep paths stable
    sys.path.insert(0, str(ROOT / "Scripts"))
    main()