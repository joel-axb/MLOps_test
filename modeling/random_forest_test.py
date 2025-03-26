import boto3.session
from mlflow.data import Dataset as MlflowDataset
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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

# mlflow.set_tracking_uri(uri="http://15.164.97.14:5000")
# mlflow.set_experiment("Learning Fashion MNIST Dataset with Resnet")

# export MLFLOW_TRACKING_USERNAME=username
# export MLFLOW_TRACKING_PASSWORD=password

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)
    

def create_experiment():
    client = mlflow.MlflowClient(
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

    



if __name__ == "__main__":

    # --- 이제부터 dev 에 띄어져있는 mlflow를 쓰기 때문에 환경변수를 dev로 임의 설정해준다

    # running_in_github = os.getenv("GITHUB_ACTIONS") == "true"

    # if running_in_github:

    #     get_best_result()



    # else:
    #     path = 'data_temp_storage/final_data.csv'
    #     repo = 'https://github.com/joel-axb/MLOps_test.git'
    #     version = 'sp-data-001' #git commit tag

    #     data_url = dvc.api.get_url(
    #         path = path,
    #         repo = repo,
    #         rev = version
    #         )




    #     data = pd.read_csv(data_url, sep=",")

    create_experiment()

    # Use the fluent API to set the tracking uri and the active experiment
    # mlflow.set_tracking_uri("http://15.164.97.14:5000")

    # Sets the current active experiment to the "Apple_Models" experiment and returns the Experiment metadata
    mlflow.set_experiment("beauty_of_joseon_prophet_3")

    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    # run_name = "apples_rf_test"

    # Define an artifact path that the model will be saved to.
    # artifact_path = "s3://data-pipeline.dev.acrossb.net/tmp/"


    # ✅ Load the dataset
    data = pd.read_csv('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')
    data["order_created_at"] = pd.to_datetime(data["order_created_at"], unit="ms")

    # ✅ Filter for customer_id = 'beautyofjoseon'
    data_beauty = data[data["customer_id"] == "beautyofjoseon"]

    # ✅ Aggregate order count per day
    data_daily = data_beauty.groupby(data_beauty["order_created_at"].dt.date).size().reset_index(name="order_count")

    X = data_daily.drop(columns=["order_created_at"])
    y = data_daily['order_count']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



    # ✅ Start MLflow run
    mlflow.start_run()

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)

    # dataset_url = dvc.api.get_url('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')

    print(mlflow.get_artifact_uri())
    # ✅ Log Parameters, Metrics, and Model to MLflow
    mlflow.log_param('model_type', 'prophet')
    mlflow.log_metric("mape", mse)

    
    
    original_value = os.getenv("AWS_PROFILE")
    os.environ["AWS_PROFILE"] = "axb-dev-general"

    # Save the Model in MLflow

    # Use a temporary file to save and log the model
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pkl") as tmp_file:
        model_path = tmp_file.name  # Get the temporary file path

    # Save the model to the temporary file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


    with open("/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv.dvc", "r") as file:
    # Load DVC lock file (YAML)
        dvc_data = yaml.safe_load(file)  # ✅ Use yaml.safe_load() instead of json.load()

    # Extract dataset info
    dataset_info = dvc_data["outs"][0]  # Get the first dataset entry
    dataset_md5 = dataset_info["md5"]
    dataset_path = dataset_info["path"]

    print(f"Dataset Path: {dataset_path}")
    print(f"MD5 Hash: {dataset_md5}")


    folder_name = dataset_md5[:2]
    file_name = dataset_md5[2:]
    # Log the temporary file as an artifact
    mlflow.pyfunc.log_model("prophet_model", python_model=CustomModelWrapper(model))


    dataset_url = 's3://data-pipeline.prod.acrossb.net/tmp/mlops_test/dvc/files/md5/' + folder_name + '/' + file_name 
    dataset = PandasDataset(data, source=dataset_url)

    dataset = mlflow.data.from_pandas(
    data, source=dataset_url
)
    mlflow.log_input(dataset, context="training")
    mlflow.log_param("dataset_md5", "b1227b9cd58931300e31bbc3c640f0")

    # mlflow.log_input(mlflow.data.Dataset(source=dataset_url))

    mlflow.log_artifact(model_path)
    # Log script as an artifact
    mlflow.log_artifact(sys.argv[0])


    # 환경 변수 원래 값으로 복구
    if original_value is None:
        del os.environ["AWS_PROFILE"]  # 원래 없던 값이면 삭제
    else:
        os.environ["AWS_PROFILE"] = original_value  # 원래 값으로 복구


    print(os.environ["AWS_PROFILE"])


    # ✅ End MLflow Run
    mlflow.end_run()

    # ✅ Print Evaluation Results
    print(f"✅ MAPE for Last 5 Days: {mse:.2f}%")
