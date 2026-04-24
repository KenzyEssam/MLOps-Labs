import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import os

from src.training.process_data import process_data
from src.training.train import train_models
from src.training.evaluate import evaluate



@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config"
)
def main(cfg: DictConfig):
    
    print("FULL CONFIG:")
    print(OmegaConf.to_yaml(cfg))

    from src.logger import ExecutorLogger
    logger = ExecutorLogger("titanic_pipeline")

    logger.info("Pipeline started")

    process_data(cfg, logger)

    train_df = pd.read_parquet(
        os.path.join(
            cfg.pipeline.data.processed_path,
            f"{cfg.pipeline.data.file_name}-train.parquet"
        )
    )

    test_df = pd.read_parquet(
        os.path.join(
            cfg.pipeline.data.processed_path,
            f"{cfg.pipeline.data.file_name}-test.parquet"
        )
    )

    X_train = train_df.drop(cfg.pipeline.data.target_column, axis=1)
    y_train = train_df[cfg.pipeline.data.target_column]

    X_test = test_df.drop(cfg.pipeline.data.target_column, axis=1)
    y_test = test_df[cfg.pipeline.data.target_column]

    train_models(X_train, y_train, logger, cfg)

    evaluate(X_test, y_test, logger, cfg)

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()