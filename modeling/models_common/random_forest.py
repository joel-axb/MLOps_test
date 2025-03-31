# models_common/linear_regression.py

import mlflow
import yaml
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow.pyfunc

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from commons.common_functions import CustomModelWrapper

class Model:

    def __init__(self, X_train, X_val, y_train, y_val, data, exp_name, PREPROCESSING_PATH):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.data = data
        self.dataset_dvc_path = '/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv.dvc'
        self.experiment_name = exp_name
        self.PREPROCESSING_PATH = PREPROCESSING_PATH

    def run(self):
        mlflow.set_experiment(self.experiment_name)

        mlflow.start_run()

        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_val)

        mse = mean_squared_error(self.y_val, y_pred)

        # dataset_url = dvc.api.get_url('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')

        print(mlflow.get_artifact_uri())
        # ✅ Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param('model_type', 'prophet')
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

        mlflow.pyfunc.log_model("random_forest_model", python_model=CustomModelWrapper(model))
        mlflow.pyfunc.log_model("random_forest_model", code_paths=[self.PREPROCESSING_PATH])
        mlflow.log_input(dataset, context="training")
        mlflow.log_param("dataset_md5", dataset_md5)
        # mlflow.log_artifact(model_path)

        if original_value is None:
            del os.environ["AWS_PROFILE"]
        else:
            os.environ["AWS_PROFILE"] = original_value

        mlflow.end_run()
        print(f"✅ MAPE for Last 5 Days: {mse:.2f}%")
