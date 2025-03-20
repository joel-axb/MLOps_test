import mlflow
import os

# ✅ Load MLflow Credentials from Environment Variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ✅ Read Run ID from GitHub Actions ENV
run_id = os.getenv("RUN_ID")

if not run_id:
    raise ValueError("🚨 RUN_ID is not set!")

# ✅ Set MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Fetch run details by run_id
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

if not run:
    raise ValueError(f"🚨 No run found for run_id: {run_id}")

# ✅ Fetch the metric value (modify as needed)
metric_name = "mape"  # Change this to the metric you need
metric_value = run.data.metrics.get(metric_name, "N/A")

# ✅ Save metric to a file for GitHub Actions
with open("metric_value.txt", "w") as f:
    f.write(str(metric_value))

print(f"✅ Fetched metric: {metric_name} = {metric_value} for run_id: {run_id}")
