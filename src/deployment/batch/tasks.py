from prefect import task
import duckdb
import mlflow

@task
def load_data():
    con = duckdb.connect("md:my_db")
    return con.execute("SELECT * FROM titanic_test").df()

@task
def load_model():
    mlflow.set_tracking_uri("https://dagshub.com/kenzy.3ab7alim/MLOps-Labs.mlflow")
    return mlflow.sklearn.load_model("models:/my-titanic-model/Production")

@task
def predict(model, df):
    df["prediction"] = model.predict(df)
    return df

@task
def save_results(df):
    con = duckdb.connect("md:my_db")
    con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM df")