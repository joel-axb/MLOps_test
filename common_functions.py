import mlflow
import mlflow.artifacts
import pandas as pd


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
