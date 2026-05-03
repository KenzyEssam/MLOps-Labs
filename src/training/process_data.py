import os
import sys
import yaml
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import re

# =========================
# Logger setup
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================
# Preprocessing function
# =========================
def build_preprocessor():

    num_cols = ["Age", "Fare", "SibSp", "Parch"]
    cat_cols = ["Pclass", "Sex", "Embarked"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])


# =========================
# Data processing function
# =========================
def process_data(cfg):

    logger.info("Starting process_data stage")

    raw_path = os.path.join(
        cfg["data"]["raw_path"],
        f"{cfg['data']['file_name']}.csv"
    )

    logger.info(f"Loading data from: {raw_path}")

    df = pd.read_csv(raw_path)

    logger.info(f"Data loaded. Shape: {df.shape}")

    df = df.drop(
        ["PassengerId", "Name", "Ticket", "Cabin"],
        axis=1
    )

    logger.info("Dropped unnecessary columns")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=df[cfg["data"]["target_column"]],
    )

    logger.info(
        f"Data split: train={train_df.shape}, test={test_df.shape}"
    )

    processed_dir = cfg["data"]["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)

    # =========================
    # FIX: handle multiple experiments cleanly
    # =========================
    
    train_path = os.path.join(
        processed_dir,
        "train-train.parquet"
    )

    test_path = os.path.join(
        processed_dir,
        "train-test.parquet"
    )

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    logger.info(f"Saved train file: {train_path}")
    logger.info(f"Saved test file: {test_path}")

    logger.info("process_data stage finished successfully")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    
    logger.info("Loading params.yaml")

    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)

    process_data(cfg)