from src.training.process_data import process_data
from src.training.train import train_models
from src.training.evaluate import evaluate

import pandas as pd
import os


def main(logger):

    logger.info("Pipeline started")

    # Step 1: process data
    process_data("train", "Survived", logger)

    # Step 2: load processed data
    train_df = pd.read_parquet("data/processed/train-train.parquet")
    test_df = pd.read_parquet("data/processed/train-test.parquet")

    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]

    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    # Step 3: train
    model = train_models(X_train, y_train, logger)

    # Step 4: evaluate
    evaluate(X_test, y_test, logger)

    logger.info("Pipeline finished")


if __name__ == "__main__":
    from src.logger import ExecutorLogger

    logger = ExecutorLogger("titanic_pipeline")
    main(logger)