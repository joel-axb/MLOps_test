import boto3.session
import mlflow
import mlflow.artifacts
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import sys
import yaml

from prophet import Prophet
import numpy as np

import dvc.api
import pandas as pd

import os
# from commons.common_functions import get_best_result
import tempfile, pickle


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from commons.common_functions import CustomModelWrapper


# def create_experiment():
#     client = mlflow.MlflowClient(
#         tracking_uri="http://15.164.97.14:5000",
#         # registry_uri=""
#     )
    
#     print(client.search_experiments())
    
#     # create an experiment
#     experiment_description = (
#         "This is the grocery forecasting project. "
#         "This experiment contains the produce models for apples."
#     )

#     experiment_tags = {
#         "project_name": "demand-forecasting",
#         "project_quarter": "Q1-2025",
#         "mlflow.note.content": experiment_description,
#     }    

    
class Model:

    def __init__(self, X_train, X_val, y_train, y_val, data, 
                exp_name, customer, store_id, sku, PREPROCESSING_PATH):
        
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.data = data
        self.dataset_dvc_path = '/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv.dvc'
        self.experiment_name = exp_name
        self.customer_id = customer
        self.store_id = store_id
        self.sku = sku
        self.PREPROCESSING_PATH = PREPROCESSING_PATH

    # if __name__ == "__main__":
    def run(self):
        mlflow.set_experiment(self.experiment_name)


        train_start_dt = self.X_train['forecast_dt'].min()
        train_end_dt = self.X_train['forecast_dt'].max()
        test_start_dt = self.X_val['forecast_dt'].min()
        test_end_dt = self.X_val['forecast_dt'].max()

        # Use the fluent API to set the tracking uri and the active experiment
        # mlflow.set_tracking_uri("http://15.164.97.14:5000")

        mlflow.start_run()

        # Sets the current active experiment to the "Apple_Models" experiment and returns the Experiment metadata
        # apple_experiment = mlflow.set_experiment("beauty_of_joseon_prophet2")


        # ✅ Load the dataset
        # data = pd.read_csv('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')
        # data["order_created_at"] = pd.to_datetime(data["order_created_at"], unit="ms")

        # # ✅ Filter for customer_id = 'beautyofjoseon'
        # data_cosrx = data[data["customer_id"] == "beautyofjoseon"]

        # ✅ Aggregate order count per day
        # data_daily = data_cosrx.groupby(data_cosrx["order_created_at"].dt.date).size().reset_index(name="order_count")

        # ✅ Rename columns for Prophet
        # data_daily.columns = ["ds", "y"]
        # data_daily["ds"] = pd.to_datetime(data_daily["ds"])

        # ✅ Use the last 5 days as test data
        # test_days = 5
        # train_df = data_daily.iloc[:-test_days]  # Train on all except last 5 days
        # test_df = data_daily.iloc[-test_days:]   # Test on last 5 days

        train_df = pd.concat([self.X_train, self.y_train], axis=1)
        test_df = pd.concat([self.X_val, self.y_val], axis=1)

        train_df = train_df[['forecast_dt', 'sellout_raw']]
        test_df = test_df[['forecast_dt', 'sellout_raw']]

        train_df.columns = ["ds", "y"]
        train_df["ds"] = pd.to_datetime(train_df["ds"])
        test_df.columns = ["ds", "y"]
        test_df["ds"] = pd.to_datetime(test_df["ds"])

    
        # ✅ Start MLflow run
        # mlflow.start_run()

        # ✅ Train Prophet Model
        model = Prophet()
        model.fit(train_df)

        # ✅ Predict ONLY for the last 5 days we have actual labels for
        future = test_df[["ds"]]  # Use actual test dates, NOT future dates
        forecast = model.predict(future)

        # ✅ Merge predictions with actual values
        eval_df = test_df.merge(forecast[["ds", "yhat"]], on="ds", how="left")

        # ✅ Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((eval_df["y"] - eval_df["yhat"]) / eval_df["y"])) * 100

        # ✅ Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param('model_type', 'prophet')
        mlflow.log_metric("mape", mape)


        original_value = os.getenv("AWS_PROFILE")
        os.environ["AWS_PROFILE"] = "axb-dev-general"


        # Save the Model in MLflow

        # Use a temporary file to save and log the model
        with open(self.dataset_dvc_path, "r") as file:
            dvc_data = yaml.safe_load(file)
            dataset_info = dvc_data["outs"][0]
            dataset_md5 = dataset_info["md5"]
            folder_name = dataset_md5[:2]
            file_name = dataset_md5[2:]

        dataset_url = f's3://data-pipeline.prod.acrossb.net/tmp/mlops_test/dvc/files/md5/{folder_name}/{file_name}'
        dataset = mlflow.data.from_pandas(self.data, source=dataset_url)

        mlflow.pyfunc.log_model("random_forest_model", python_model=CustomModelWrapper(model))
        mlflow.log_input(dataset, context="training")
        mlflow.log_param("dataset_md5", dataset_md5)
        mlflow.log_param("sku", self.sku)

        mlflow.log_param("train_start_dt", train_start_dt)
        mlflow.log_param("train_end_dt", train_end_dt)
        mlflow.log_param("test_start_dt", test_start_dt)
        mlflow.log_param("test_end_dt", test_end_dt)

        mlflow.log_param("customer_id", self.customer_id)
        mlflow.log_param("store_id", self.store_id)

        # mlflow.log_artifact(model_path)

        if original_value is None:
            del os.environ["AWS_PROFILE"]
        else:
            os.environ["AWS_PROFILE"] = original_value

        mlflow.end_run()
        print(f"✅ MAPE: {mape:.2f}%")


        return forecast['yhat'].values