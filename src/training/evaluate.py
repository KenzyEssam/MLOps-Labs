import os
import sys
import json
import yaml
import pickle
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score


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
    experiment = sys.argv[1]  # titanic or titanic2

    # -------------------------
    # LOAD CONFIG (params.yaml)
    # -------------------------
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)[experiment]

    model_path = cfg["output"]["model_path"]
    metrics_path = cfg["output"]["metrics_path"]

    target_col = cfg["data"]["target_column"]

    # -------------------------
    # LOAD TEST DATA
    # -------------------------
    test_path = "data/processed/train-test.parquet"

    # handle second experiment naming automatically
    if experiment == "titanic2":
        test_path = "data/processed/train-test-2.parquet"

    import pandas as pd

    test_data = pd.read_parquet(test_path)

    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # -------------------------
    # LOAD MODEL
    # -------------------------
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # -------------------------
    # RUN EVALUATION
    # -------------------------
    evaluate(X_test, y_test, model, metrics_path, logger)