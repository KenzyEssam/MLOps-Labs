import os
import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE = os.path.join("data", "raw")
DESTINATION = os.path.join("data", "processed")


def process_data(file_name: str, target_col: str, logger) -> None:
    logger.info("Loading raw Titanic data")

    df = pd.read_csv(os.path.join(SOURCE, f"{file_name}.csv"))

    # Basic cleanup
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    logger.info("Splitting train/test")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col],
    )

    os.makedirs(DESTINATION, exist_ok=True)

    train_df.to_parquet(os.path.join(DESTINATION, f"{file_name}-train.parquet"))
    test_df.to_parquet(os.path.join(DESTINATION, f"{file_name}-test.parquet"))

    logger.info("Data processed and saved")