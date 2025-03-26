import mlflow
import os
from dotenv import load_dotenv


load_dotenv()

# Read Run ID from GitHub Actions ENV
run_id = os.getenv("RUN_ID")
customer_id = os.getenv("CUSTOMER_ID")
sku = os.getenv("SKU")

if not run_id:
    raise ValueError("🚨 RUN_ID is not set!")


# Fetch run details by run_id
client = mlflow.tracking.MlflowClient()


run = client.get_run(run_id=run_id)

if not run:
    raise ValueError(f"🚨 No run found for run_id: {run_id}")

# Fetch the metric value (modify as needed)
new_metric_name = "mape"  # Change this to the metric you need
new_metric_value = run.data.metrics.get(new_metric_name, "N/A")


print(f"✅ Fetched metric: {new_metric_name} = {new_metric_value} for run_id: {run_id}")


# Construct a description-based filter
customer_id = f"'{customer_id}'"
sku = f"'{sku}'"

# filter_str = f"description LIKE '%CUSTOMER_ID={customer_id}%' AND description LIKE '%SKU_ID={sku}%'"
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
current_metric_name = "mape"  # Change this to the metric you need
current_metric_value = run.data.metrics.get(current_metric_name, "N/A")

print(f"✅ Fetched metric: {current_metric_name} = {current_metric_value} for run_id: {run_id}")


# Print in KEY=VALUE format for GitHub Actions to parse
print(f"NEW_METRIC={new_metric_value}")
print(f"OLD_METRIC={current_metric_value}")