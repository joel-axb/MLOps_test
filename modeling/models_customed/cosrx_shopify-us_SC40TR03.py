# models_common/linear_regression.py

import mlflow
import pickle
import tempfile
import yaml
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow.pyfunc
from mlflow.data.pandas_dataset import PandasDataset

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from commons.common_functions import CustomModelWrapper

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

    def run(self):
        mlflow.set_experiment(self.experiment_name)

        mlflow.start_run()

        train_start_dt = self.X_train['forecast_dt'].min()
        train_end_dt = self.X_train['forecast_dt'].max()
        test_start_dt = self.X_val['forecast_dt'].min()
        test_end_dt = self.X_val['forecast_dt'].max()

        # -- little data-preprocessing --
        X_train = self.X_train.drop(columns=['forecast_dt'])
        X_val = self.X_val.drop(columns=['forecast_dt'])
        # -------------------------------

        model = LinearRegression()
        model.fit(X_train, self.y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(self.y_val, y_pred)

        print(mlflow.get_artifact_uri())
        mlflow.log_param('model_type', 'customed_linear_regression')
        mlflow.log_metric("mape", mse)

        original_value = os.getenv("AWS_PROFILE")
        os.environ["AWS_PROFILE"] = "axb-dev-general"


        with open(self.dataset_dvc_path, "r") as file:
            dvc_data = yaml.safe_load(file)
            dataset_info = dvc_data["outs"][0]
            dataset_md5 = dataset_info["md5"]
            folder_name = dataset_md5[:2]
            file_name = dataset_md5[2:]

        dataset_url = f's3://data-pipeline.prod.acrossb.net/tmp/mlops_test/dvc/files/md5/{folder_name}/{file_name}'
        dataset = mlflow.data.from_pandas(self.data, source=dataset_url)

        mlflow.pyfunc.log_model("linear_model", python_model=CustomModelWrapper(model))
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
        print(f"✅ MAPE for Last 5 Days: {mse:.2f}%")


        return y_pred