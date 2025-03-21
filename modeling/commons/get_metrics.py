import mlflow
import os
from dotenv import load_dotenv
load_dotenv()

# ✅ Load MLflow Credentials from Environment Variables
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
# MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# if not all([MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD]):
#     raise ValueError("🚨 MLflow credentials are not set properly!")

# ✅ Set MLflow Tracking URI and Pass Credentials
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ✅ Read Run ID from GitHub Actions ENV
run_id = os.getenv("RUN_ID")

# if not run_id:
#     raise ValueError("🚨 RUN_ID is not set!")

# ✅ Set MLflow Tracking URI
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_tracking_uri("http://15.164.97.14:5000")
    
# original_value = os.getenv("AWS_PROFILE")
# os.environ["AWS_PROFILE"] = "axb-dev-general"

# ✅ Fetch run details by run_id
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id='16fae3fb65254664bd44bd83eb7f950c')

if not run:
    raise ValueError(f"🚨 No run found for run_id: {run_id}")

# ✅ Fetch the metric value (modify as needed)
metric_name = "mape"  # Change this to the metric you need
new_metric_value = run.data.metrics.get(metric_name, "N/A")

# 환경 변수 원래 값으로 복구
# if original_value is None:
#     del os.environ["AWS_PROFILE"]  # 원래 없던 값이면 삭제
# else:
#     os.environ["AWS_PROFILE"] = original_value  # 원래 값으로 복구






# ✅ Save metric to a file for GitHub Actions
# with open("metric_value.txt", "w") as f:
#     f.write(str(metric_value))

print(f"✅ Fetched metric: {metric_name} = {new_metric_value} for run_id: {run_id}")



# Construct a description-based filter
customer_id = "'beautyofjoseon'"
sku = "'sku_001'"

filter_str = f"description LIKE '%CUSTOMER_ID={customer_id}%' AND description LIKE '%SKU_ID={sku}%'"
filter_str = f"name='test_1' and tag.customer_id={customer_id} and tag.sku={sku}"
results = client.search_registered_models(filter_string=filter_str)

if results:
    model = results[0]
    print(f"✅ Found model: {model.name}")
    latest_version = model.latest_versions[0]
    print(f"Latest version: {latest_version.version}, Run ID: {latest_version.run_id}")
else:
    print("❌ No matching model found in the registry.")




run = client.get_run(run_id=latest_version.run_id)

if not run:
    raise ValueError(f"🚨 No run found for run_id: {run_id}")

# ✅ Fetch the metric value (modify as needed)
metric_name = "mape"  # Change this to the metric you need
current_metric_value = run.data.metrics.get(metric_name, "N/A")

# 환경 변수 원래 값으로 복구
# if original_value is None:
#     del os.environ["AWS_PROFILE"]  # 원래 없던 값이면 삭제
# else:
#     os.environ["AWS_PROFILE"] = original_value  # 원래 값으로 복구




# ✅ Save metric to a file for GitHub Actions
# with open("metric_value.txt", "w") as f:
#     f.write(str(metric_value))

print(f"✅ Fetched metric: {metric_name} = {current_metric_value} for run_id: {run_id}")



new_metric = 0.87
old_metric = 0.90

# Print in KEY=VALUE format for GitHub Actions to parse
print(f"NEW_METRIC={new_metric}")
print(f"OLD_METRIC={old_metric}")