import mlflow
import json
import os

# ✅ MLflow Tracking URI (Modify if necessary)
MLFLOW_TRACKING_URI = "http://15.164.97.14:5000"

# ✅ Read experiment ID from GitHub Actions ENV
experiment_id = os.getenv("EXPERIMENT_ID")

if not experiment_id:
    raise ValueError("🚨 EXPERIMENT_ID is not set!")

# ✅ Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Get latest run in the experiment
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)

if not runs:
    raise ValueError(f"🚨 No runs found for experiment_id: {experiment_id}")

latest_run = runs[0]
run_id = latest_run.info.run_id

# ✅ Fetch a specific metric (modify as needed)
metric_name = "mape"  # Change this to the metric you need
metric_value = latest_run.data.metrics.get(metric_name, "N/A")

# ✅ Save metric to a file for GitHub Actions
with open("metric_value.txt", "w") as f:
    f.write(str(metric_value))

print(f"✅ Fetched metric: {metric_name} = {metric_value} for run_id: {run_id}")
