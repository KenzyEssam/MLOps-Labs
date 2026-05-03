import os
import sys
import yaml
import pickle
import logging

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from process_data import build_preprocessor
from pathlib import Path
import yaml



# =========================
# LOAD CONFIG
# =========================

def load_config():
    experiment = sys.argv[1]  # titanic or titanic2

    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")

    with open(params_path, "r") as f:
        cfg = yaml.safe_load(f)

    if experiment not in cfg:
        raise KeyError(f"Experiment '{experiment}' not found in params.yaml")

    return cfg[experiment]


# =========================
# TRAIN MODELS
# =========================
def train_models(X, y, logger, cfg):

    logger.info("Starting training stage")

    preprocessor = build_preprocessor()

    models = {
        "logreg": LogisticRegression(
            **cfg["model"]["logistic"]
        ),

        "rf": RandomForestClassifier(
            **cfg["model"]["random_forest"]
        )
    }

    best_model = None
    best_score = -1
    best_model_name = None

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        score = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="accuracy"
        ).mean()

        logger.info(f"{name} accuracy: {score}")

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_model_name = name

    # =========================
    # FIT BEST MODEL
    # =========================
    best_model.fit(X, y)

    # =========================
    # SAVE MODEL
    # =========================
    model_path = cfg["output"]["model_path"]

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    logger.info(f"Saving model → {model_path}")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    logger.info(f"Best model selected: {best_model_name}")
    logger.info(f"Best accuracy score: {best_score}")

    return best_model


# =========================
# LOAD DATA
# =========================
def load_data(cfg):
    processed_path = cfg["data"]["processed_path"]

    train_file = os.path.join(processed_path, "train-train.parquet")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Processed train file not found: {train_file}")

    df = pd.read_parquet(train_file)

    target = cfg["data"]["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


# =========================
# MAIN FUNCTION
# =========================
def main():

    cfg = load_config()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")

    X, y = load_data(cfg)

    logger.info(f"Data shape: {X.shape}")

    train_models(X, y, logger, cfg)


# =========================
# ENTRY POINT (IMPORTANT FIX)
# =========================
if __name__ == "__main__":
    main()