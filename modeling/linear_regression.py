import boto3.session
from mlflow.data import Dataset as MlflowDataset
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys
import dvc.api
import yaml
from mlflow.data.pandas_dataset import PandasDataset

from prophet import Prophet
import numpy as np

import dvc.api
import pandas as pd

import os
from commons.common_functions import get_best_result
import tempfile, pickle

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def create_experiment():
    client = mlflow.MlflowClient()

    print(client.search_experiments())

    experiment_description = (
        "This is the grocery forecasting project. "
        "This experiment contains the produce models for apples."
    )

    experiment_tags = {
        "project_name": "demand-forecasting",
        "project_quarter": "Q1-2025",
        "mlflow.note.content": experiment_description,
    }


if __name__ == "__main__":

    create_experiment()

    mlflow.set_experiment("tirtir_sku_002_linear_regression")

    try:
        data = pd.read_csv('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')
        
    except:
        print("you alrady deleted the data in local!")
        print("I will get the most recent data you did dvc push")
        with dvc.api.open(
            path="../data_temp_storage/final_data.csv",
            repo=".",         # current repo
            rev=None          # ← None = workspace (latest even if uncommitted)
        ) as fd:
            data = pd.read_csv(fd)


    
    data["order_created_at"] = pd.to_datetime(data["order_created_at"], unit="ms")

    # data_beauty = data[data["customer_id"] == "beautyofjoseon"]

    data_daily = data.groupby(data["order_created_at"].dt.date).size().reset_index(name="order_count")

    X = data_daily.drop(columns=["order_created_at"])
    y = data_daily['order_count']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.start_run()

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)

    print(mlflow.get_artifact_uri())

    mlflow.log_param('model_type', 'linear_regression')
    mlflow.log_metric("mape", mse)

    original_value = os.getenv("AWS_PROFILE")
    os.environ["AWS_PROFILE"] = "axb-dev-general"

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pkl") as tmp_file:
        model_path = tmp_file.name

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open("/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv.dvc", "r") as file:
        dvc_data = yaml.safe_load(file)

    dataset_info = dvc_data["outs"][0]
    dataset_md5 = dataset_info["md5"]
    dataset_path = dataset_info["path"]

    print(f"Dataset Path: {dataset_path}")
    print(f"MD5 Hash: {dataset_md5}")

    folder_name = dataset_md5[:2]
    file_name = dataset_md5[2:]

    mlflow.pyfunc.log_model("prophet_model", python_model=CustomModelWrapper(model))

    dataset_url = 's3://data-pipeline.prod.acrossb.net/tmp/mlops_test/dvc/files/md5/' + folder_name + '/' + file_name 
    dataset = mlflow.data.from_pandas(data, source=dataset_url)

    mlflow.log_input(dataset, context="training")
    mlflow.log_param("dataset_md5", "b1227b9cd58931300e31bbc3c640f0")

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(sys.argv[0])

    if original_value is None:
        del os.environ["AWS_PROFILE"]
    else:
        os.environ["AWS_PROFILE"] = original_value

    print(os.environ["AWS_PROFILE"])

    mlflow.end_run()

    print(f"✅ MAPE for Last 5 Days: {mse:.2f}%")