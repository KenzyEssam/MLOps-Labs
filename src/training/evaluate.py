import os
import json
import yaml
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
import dagshub

from mlflow.tracking import MlflowClient


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(X_test, y_test, model, logger):

    preds = model.predict(X_test)

    report = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }

    logger.info(f"Evaluation results: {report}")

    return report


# =========================
# MAIN ENTRY POINT
# =========================
if __name__ == "__main__":

    # -------------------------
    # LOGGER
    # -------------------------
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # -------------------------
    # LOAD CONFIG
    # -------------------------
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    target_col = cfg["data"]["target_column"]
    test_path = cfg["data"]["test_path"]

    # -------------------------
    # LOAD TEST DATA
    # -------------------------
    test_data = pd.read_parquet(test_path)

    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # -------------------------
    # INIT DAGSHUB + MLFLOW
    # -------------------------
    load_dotenv()

    dagshub.init(
        repo_owner=cfg["experiment"]["DAGSHUB_USERNAME"],
        repo_name=cfg["experiment"]["DAGSHUB_REPO"],
        mlflow=True
    )

    mlflow.set_experiment(cfg["experiment"]["name"])

    # -------------------------
    # START MLFLOW RUN
    # -------------------------
    with mlflow.start_run():

        # -------------------------
        # LOAD MODEL FROM REGISTRY
        # -------------------------
        client = MlflowClient()

        # Get latest model version in Staging
        latest = client.get_latest_versions(
            name="my-titanic-model",
            stages=["Staging"]
        )[0]

        model_version = latest.version
        model_uri = f"models:/my-titanic-model/{model_version}"

        # Load exact version (NOT stage)
        model = mlflow.sklearn.load_model(model_uri)

        # -------------------------
        # LOG MODEL LINEAGE
        # -------------------------
        mlflow.set_tag("model_name", "my-titanic-model")
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("model_uri", model_uri)

        # -------------------------
        # RUN EVALUATION
        # -------------------------
        report = evaluate(X_test, y_test, model, logger)

        # -------------------------
        # LOG TO MLFLOW
        # -------------------------
        mlflow.log_metrics(report)

        # Log as artifact directly (no local file)
        mlflow.log_dict(report, "metrics.json")
        # Optional but useful tags
        mlflow.set_tag("model_name", "my-titanic-model")
