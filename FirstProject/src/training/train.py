import os
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"


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


def train_models(X, y, logger):

    preprocessor = build_preprocessor()

    models = {
        "logreg": LogisticRegression(max_iter=200),
        "rf": RandomForestClassifier(random_state=42)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        score = scores.mean()

        logger.info(f"{name} accuracy: {score}")

        if score > best_score:
            best_score = score
            best_model = pipeline

    best_model.fit(X, y)

    os.makedirs(MODEL_PATH, exist_ok=True)

    with open(os.path.join(MODEL_PATH, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    logger.info("Best model saved")

    return best_model