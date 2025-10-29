# Scripts/train_xgb_mlflow.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "train_mini.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_OUT = MODEL_DIR / "xgb_model.pkl"

def load_data():
    df = pd.read_parquet(DATA)
    y = df["churn"].astype(int)
    X = df.drop(columns=["userId", "churn"])
    return X, y

def train(X, y):
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
    auc = roc_auc_score(yte, proba)
    print("train_xgb_mlflow: ROC-AUC:", auc)
    print(classification_report(yte, (proba >= 0.5).astype(int)))
    return model, float(auc), params

def save_and_log(model, auc, params):
    # save model file
    joblib.dump(model, MODEL_OUT)

    # MLflow logging
    mlflow.set_experiment("churn_xgb")
    with mlflow.start_run(run_name="xgb_baseline"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_param("data_path", str(DATA))

        # log serialized model file and an MLflow model flavor
        mlflow.log_artifact(str(MODEL_OUT))
        mlflow.xgboost.log_model(model, artifact_path="model")

    print("train_xgb_mlflow: model saved ->", MODEL_OUT)

if __name__ == "__main__":
    X, y = load_data()
    m, auc, p = train(X, y)
    save_and_log(m, auc, p)