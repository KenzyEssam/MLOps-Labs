import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(X_test, y_test, logger, cfg):

    with open(cfg.pipeline.output.model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)

    report = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }

    os.makedirs("reports", exist_ok=True)

    with open(cfg.pipeline.output.metrics_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(report)