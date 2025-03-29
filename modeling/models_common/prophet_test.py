import boto3.session
import mlflow
import mlflow.artifacts
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

from prophet import Prophet
import numpy as np

import dvc.api
import pandas as pd

import os
from commons.common_functions import get_best_result
import tempfile, pickle


def create_experiment():
    client = mlflow.MlflowClient(
        tracking_uri="http://15.164.97.14:5000",
        # registry_uri=""
    )
    
    print(client.search_experiments())
    
    # create an experiment
    experiment_description = (
        "This is the grocery forecasting project. "
        "This experiment contains the produce models for apples."
    )

    experiment_tags = {
        "project_name": "demand-forecasting",
        "project_quarter": "Q1-2025",
        "mlflow.note.content": experiment_description,
    }    

    



# if __name__ == "__main__":
def run_prophet():

    create_experiment()

    # Use the fluent API to set the tracking uri and the active experiment
    mlflow.set_tracking_uri("http://15.164.97.14:5000")

    # Sets the current active experiment to the "Apple_Models" experiment and returns the Experiment metadata
    apple_experiment = mlflow.set_experiment("beauty_of_joseon_prophet2")


    # ✅ Load the dataset
    data = pd.read_csv('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')
    data["order_created_at"] = pd.to_datetime(data["order_created_at"], unit="ms")

    # ✅ Filter for customer_id = 'beautyofjoseon'
    data_cosrx = data[data["customer_id"] == "beautyofjoseon"]

    # ✅ Aggregate order count per day
    data_daily = data_cosrx.groupby(data_cosrx["order_created_at"].dt.date).size().reset_index(name="order_count")

    # ✅ Rename columns for Prophet
    data_daily.columns = ["ds", "y"]
    data_daily["ds"] = pd.to_datetime(data_daily["ds"])

    # ✅ Use the last 5 days as test data
    test_days = 5
    train_df = data_daily.iloc[:-test_days]  # Train on all except last 5 days
    test_df = data_daily.iloc[-test_days:]   # Test on last 5 days


    # ✅ Start MLflow run
    mlflow.start_run()

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
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pkl") as tmp_file:
        model_path = tmp_file.name  # Get the temporary file path

    # Save the model to the temporary file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Log the temporary file as an artifact
    mlflow.log_artifact(model_path)
    # Log script as an artifact
    mlflow.log_artifact(sys.argv[0])

    # ✅ End MLflow Run
    mlflow.end_run()

    # ✅ Print Evaluation Results
    print(f"✅ MAPE for Last 5 Days: {mape:.2f}%")
