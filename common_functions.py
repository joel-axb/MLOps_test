import mlflow
import mlflow.artifacts

run_id = "dbaa981974b1475bb5ca71595353f735"

artifact_uri = f"/Users/joel/Documents/github/MLOps_test/mlruns/0/{run_id}/artifacts/linear_regression.py"
mlflow.artifacts.download_artifacts(artifact_uri, dst_path="./modeling")