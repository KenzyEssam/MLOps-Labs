""" import pandas as pd


def download_titanic_data(logger):

    logger.info("Loading Titanic dataset...")

    df = pd.read_csv("../data/titanic/train.csv")

    logger.info(f"Data loaded with shape: {df.shape}")

    return df """