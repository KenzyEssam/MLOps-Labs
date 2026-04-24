import os
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


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
def process_data(cfg, logger):

    logger.info("Loading raw data")

    df = pd.read_csv(
        os.path.join(
            cfg.pipeline.data.raw_path,
            f"{cfg.pipeline.data.file_name}.csv"
        )
    )

    df = df.drop(
        ["PassengerId", "Name", "Ticket", "Cabin"],
        axis=1
    )

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.pipeline.split.test_size,
        random_state=cfg.pipeline.split.random_state,
        stratify=df[cfg.pipeline.data.target_column],
    )

    os.makedirs(cfg.pipeline.data.processed_path, exist_ok=True)

    train_df.to_parquet(
        os.path.join(
            cfg.pipeline.data.processed_path,
            f"{cfg.pipeline.data.file_name}-train.parquet"
        )
    )

    test_df.to_parquet(
        os.path.join(
            cfg.pipeline.data.processed_path,
            f"{cfg.pipeline.data.file_name}-test.parquet"
        )
    )

    logger.info("Data processed successfully")