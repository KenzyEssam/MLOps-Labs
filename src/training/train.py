import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.training.process_data import build_preprocessor


def train_models(X, y, logger, cfg):

    preprocessor = build_preprocessor()

    models = {
        "logreg": LogisticRegression(
            max_iter=cfg.pipeline.model.logistic.max_iter
        ),

        "rf": RandomForestClassifier(
            random_state=cfg.pipeline.model.random_forest.random_state,
            n_estimators=cfg.pipeline.model.random_forest.n_estimators
        )
    }

    best_model = None
    best_score = 0
    best_model_name = None   # ✅ NEW

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

        logger.info(f"{name}: {score}")

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_model_name = name   # ✅ NEW

    best_model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    # Save model
    with open(cfg.pipeline.output.model_path, "wb") as f:
        pickle.dump(best_model, f)

    # ✅ LOG BEST MODEL NAME
    logger.info(f"Best model selected: {best_model_name}")
    logger.info(f"Best accuracy score: {best_score}")

    return best_model