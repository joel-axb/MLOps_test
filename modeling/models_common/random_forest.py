# root 경로 설정용 setup
import subprocess, sys, os
# 현재 파일 위치 기준 절대 경로 추출
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # setup.py 기준
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", project_root])

import mlflow
import yaml
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow.pyfunc
import sys, os

import pre_processing_1.pre_processing as pre_processing



class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        
        return self.model.predict(model_input)



class PreprocessingWrapper(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input):
        return pre_processing.prepare_dataset(model_input)


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



    def pre_processing_2(self):

        # -- little data-preprocessing --
        X_train = self.X_train.drop(columns=['forecast_dt'])
        X_val = self.X_val.drop(columns=['forecast_dt'])
        # -------------------------------



        return X_train, X_val


    def run(self):
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()

        X_train, X_val = self.pre_processing_2()

        train_start_dt = self.X_train['forecast_dt'].min()
        train_end_dt = self.X_train['forecast_dt'].max()
        test_start_dt = self.X_val['forecast_dt'].min()
        test_end_dt = self.X_val['forecast_dt'].max()

        model = RandomForestRegressor(max_depth=200)
        model.fit(X_train, self.y_train)

        y_pred = model.predict(X_val)


        mse = mean_squared_error(self.y_val, y_pred)

        # dataset_url = dvc.api.get_url('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')

        print(mlflow.get_artifact_uri())
        # ✅ Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param('model_type', 'random_forest')
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

        # from commons.common_functions import CustomModelWrapper

        mlflow.pyfunc.log_model(
            artifact_path="random_forest_model",
            python_model=CustomModelWrapper(model),
            code_paths=[
                os.path.join(project_root, ".")
            ]
        )

        mlflow.pyfunc.log_model(
            artifact_path="preprocessing_model",
            python_model=PreprocessingWrapper(),
            code_paths=[
                os.path.join(project_root, ".")
            ]
        )

        
        mlflow.log_input(dataset, context="training")
        mlflow.log_param("dataset_md5", dataset_md5)
        mlflow.log_param("sku", self.sku)
        mlflow.log_param("customer_id", self.customer_id)

        mlflow.log_param("train_start_dt", train_start_dt)
        mlflow.log_param("train_end_dt", train_end_dt)
        mlflow.log_param("test_start_dt", test_start_dt)
        mlflow.log_param("test_end_dt", test_end_dt)

        mlflow.log_param("store_id", self.store_id)
        mlflow.log_artifact(os.path.join(project_root, "pre_processing_1/pre_processing.py"))
        mlflow.log_artifact(__file__)
        mlflow.log_artifact(os.path.join(project_root, "pre_processing_1/get_data_from_athena.py"))

        folder_to_log = os.path.join(project_root, "pre_processing_1/queries")

        for filename in os.listdir(folder_to_log):
            full_path = os.path.join(folder_to_log, filename)
            if os.path.isfile(full_path):
                mlflow.log_artifact(full_path, artifact_path="queries")


        if original_value is None:
            del os.environ["AWS_PROFILE"]
        else:
            os.environ["AWS_PROFILE"] = original_value

        mlflow.end_run()
        print(f"✅ MAPE: {mse:.2f}%")




        return y_pred