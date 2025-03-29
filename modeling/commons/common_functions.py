import mlflow
import mlflow.artifacts
import pandas as pd
import dvc.api


class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def get_best_artifacts():

    run_id = "dbaa981974b1475bb5ca71595353f735"
    artifact_uri = f"/Users/joel/Documents/github/MLOps_test/mlruns/0/{run_id}/artifacts/linear_regression.py"
    mlflow.artifacts.download_artifacts(artifact_uri, dst_path="./modeling")


def get_best_result():
    # Get experiment details

    run_id = "dbaa981974b1475bb5ca71595353f735"
    run = mlflow.get_run(run_id)
    
    if run is None:
        print(f"Run '{run_id}' not found.")
        return None

    metrics = run.data.metrics
    print(f"Metrics for run {run_id}: {metrics}")
    return metrics
    

def read_final_dataset(config):

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

    return data
