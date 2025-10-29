# Scripts/monitor_concept_drift.py
from __future__ import annotations
import sys
from typing import Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "xgb_churn_model"   # <-- your registered model name
AUC_METRIC = "roc_auc"
THRESHOLD = 0.05                 # drift if latest_auc <= prev_auc - THRESHOLD

def get_auc_from_version(client: MlflowClient, model_name: str, version: str) -> Optional[float]:
    """Fetch roc_auc metric from the run that produced a specific model version."""
    mv = client.get_model_version(name=model_name, version=version)
    run = client.get_run(mv.run_id)
    auc = run.data.metrics.get(AUC_METRIC)
    return float(auc) if auc is not None else None

def get_latest_two_versions(client: MlflowClient, model_name: str) -> Tuple[str, Optional[str]]:
    versions = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda v: int(v.version),
        reverse=True,
    )
    latest = versions[0].version if versions else None
    prev   = versions[1].version if len(versions) > 1 else None
    if latest is None:
        raise RuntimeError(f"No versions found for registered model '{model_name}'.")
    return latest, prev

def main() -> int:
    client = MlflowClient()
    latest_v, prev_v = get_latest_two_versions(client, MODEL_NAME)

    latest_auc = get_auc_from_version(client, MODEL_NAME, latest_v)
    if latest_auc is None:
        print(f"[ERROR] '{AUC_METRIC}' not logged for {MODEL_NAME} v{latest_v}.")
        return 2

    if not prev_v:
        print(f"[INFO] Only one version found (v{latest_v}). AUC={latest_auc:.4f}. No drift check yet.")
        return 0

    prev_auc = get_auc_from_version(client, MODEL_NAME, prev_v)
    if prev_auc is None:
        print(f"[WARN] Previous version (v{prev_v}) has no '{AUC_METRIC}'. Skipping drift test.")
        return 0

    drop = prev_auc - latest_auc
    print(f"[INFO] {MODEL_NAME}: prev AUC (v{prev_v})={prev_auc:.4f} | latest (v{latest_v})={latest_auc:.4f} | drop={drop:.4f}")

    if drop >= THRESHOLD:
        print(f"[ALERT] Concept drift detected (drop >= {THRESHOLD:.2f}).")
        return 3

    print("[OK] No concept drift.")
    return 0

if __name__ == "__main__":
    sys.exit(main())