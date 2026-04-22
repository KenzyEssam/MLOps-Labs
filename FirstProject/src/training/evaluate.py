import os
import pickle
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODEL_PATH = "models"
REPORT_PATH = "reports"


def evaluate(X_test, y_test, logger):

    logger.info("Loading model")

    with open(os.path.join(MODEL_PATH, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)

    report = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
    }

    os.makedirs(REPORT_PATH, exist_ok=True)

    with open(os.path.join(REPORT_PATH, "metrics.json"), "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Evaluation saved: {report}")