import sys
import pickle
import logging

import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from process_data import build_preprocessor
from pathlib import Path
import yaml
from sklearn.model_selection import cross_validate

import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score
import os
import dagshub
import mlflow
from dotenv import load_dotenv

    
# =========================
# LOAD CONFIG
# =========================

def load_config():
    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"

    with open(params_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


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
        
        mlflow.log_params({f"{name}_param_{k}": v for k, v in model.get_params().items()})

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        }

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=5,
            scoring=scoring
        )
        
        accuracy = scores["test_accuracy"].mean()
        precision = scores["test_precision"].mean()
        recall = scores["test_recall"].mean()
        f1 = scores["test_f1"].mean()
        
        logger.info(f"{name} accuracy: {accuracy}")
        logger.info(f"{name} precision: {precision}")
        logger.info(f"{name} recall: {recall}")
        logger.info(f"{name} f1: {f1}")
        
        ##Log Metrics
        mlflow.log_metric(f"{name}_cv_accuracy", accuracy)
        mlflow.log_metric(f"{name}_cv_precision", precision)
        mlflow.log_metric(f"{name}_cv_recall", recall)
        mlflow.log_metric(f"{name}_cv_f1", f1)

        if accuracy > best_score:
            best_score = accuracy
            best_model = pipeline
            best_model_name = name

    # =========================
    # FIT BEST MODEL
    # =========================
    best_model.fit(X, y)
    
    mlflow.log_param("best_model", best_model_name) ##Log Best Model
    mlflow.log_metric("best_accuracy", best_score) ##Log Best Accuracy
        
            
    # create predictions for signature
    train_preds = best_model.predict(X)

    # infer signature
    signature = infer_signature(X, train_preds)

    # log model WITH signature
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="my-titanic-model",
        signature=signature
    )
    
    
  
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
    
    load_dotenv()
    dagshub.init(
        repo_owner=cfg["experiment"]["DAGSHUB_USERNAME"],
        repo_name=cfg["experiment"]["DAGSHUB_REPO"],
        mlflow=True
    )
    
    # Set experiment (Dagshub handles tracking backend automatically)
    mlflow.set_experiment(cfg["experiment"]["name"])
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    
    
    
    X, y = load_data(cfg)
    

    with mlflow.start_run():
        train_models(X, y, logger, cfg)

        client = MlflowClient()

        latest_version = client.get_latest_versions(
            name="my-titanic-model",
            stages=["None"]
        )[0].version

        client.transition_model_version_stage(
            name="my-titanic-model",
            version=latest_version,
            stage="Staging"
        )


# =========================
# ENTRY POINT (IMPORTANT FIX)
# =========================
if __name__ == "__main__":
    main()