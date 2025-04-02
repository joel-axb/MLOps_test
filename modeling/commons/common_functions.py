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
            repo="None",         # current repo
            rev=None          # ← None = workspace (latest even if uncommitted)
        ) as fd:
            data = pd.read_csv(fd)

    return data



def get_best_result_for_each_sku():

    # 전체 실험(run) 정보 가져오기
    experiment_name = "joel_20250402-3"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # 예시: rmse 기준으로 가장 성능 좋은 모델 선택
    best_runs = (
        runs_df
        .sort_values("metrics.mape", ascending=True)
        .groupby("params.sku")  # SKU 기준으로 그룹화
        .first()  # 각 SKU별로 가장 좋은 run 하나
        .reset_index()
    )
    print('!!')
    print(best_runs[["params.sku", "params.model_type", "metrics.mape", "run_id"]])