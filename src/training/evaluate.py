import os
import sys
import json
import yaml
import pickle
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score
import re
import pandas as pd
import mlflow.sklearn
import mlflow

from dotenv import load_dotenv
import dagshub

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(X_test, y_test, model, metrics_path, logger):

    preds = model.predict(X_test)

    report = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }

    # ensure directory exists
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)

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
    # LOAD EXPERIMENT NAME
    # -------------------------

    # -------------------------
    # LOAD CONFIG (params.yaml)
    # -------------------------
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["output"]["model_path"]
    metrics_path = cfg["output"]["metrics_path"]

    target_col = cfg["data"]["target_column"]

    # -------------------------
    # LOAD TEST DATA
    # -------------------------
    

    test_path = cfg["data"]["test_path"]
        


    test_data = pd.read_parquet(test_path)

    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # -------------------------
    # LOAD MODEL
    # -------------------------
    
    load_dotenv()

    dagshub.init(
        repo_owner=cfg["experiment"]["DAGSHUB_USERNAME"],
        repo_name=cfg["experiment"]["DAGSHUB_REPO"],
        mlflow=True
    )

    mlflow.set_experiment(cfg["experiment"]["name"])
    
    
    model = mlflow.sklearn.load_model(
        "models:/my-titanic-model/Staging"
    )
    
    # -------------------------
    # RUN EVALUATION
    # -------------------------
    evaluate(X_test, y_test, model, metrics_path, logger)