import mlflow
import os
from dotenv import load_dotenv
from common_functions import get_best_result_for_each_sku


# load_dotenv()

# # Read Run ID from GitHub Actions ENV
# run_id = os.getenv("RUN_ID")
# customer_id = os.getenv("CUSTOMER_ID")
# sku = os.getenv("SKU")

bests = get_best_result_for_each_sku()

# if not run_id:
#     raise ValueError("🚨 RUN_ID is not set!")

for one_tuple in bests:

    customer_id = one_tuple[0]
    store_id = one_tuple[1]
    sku = one_tuple[2]
    run_id = one_tuple[3]

    # Fetch run details by run_id
    client = mlflow.tracking.MlflowClient()


    run = client.get_run(run_id=run_id)
    # experiment_id = run.info.experiment_id



    if not run:
        raise ValueError(f"🚨 No run found for run_id: {run_id}")

    # Fetch the metric value (modify as needed)
    new_metric_name = "mape"  # Change this to the metric you need
    new_metric_value = run.data.metrics.get(new_metric_name, "N/A")


    print(f"✅ Fetched metric: {new_metric_name} = {new_metric_value} for run_id: {run_id}")


    # Construct a description-based filter
    customer_id = f"{customer_id}"
    store_id = f"{store_id}"
    sku = f"{sku}"

    # filter_str = f"description LIKE '%CUSTOMER_ID={customer_id}%' AND description LIKE '%SKU_ID={sku}%'"
    filter_str = f"name='{customer_id}_{store_id}_{sku}' and tag.customer_id='{customer_id}' and tag.store_id = '{store_id}' and tag.sku='{sku}'"
    results = client.search_registered_models(filter_string=filter_str)

    if results:
        model = results[0]
        print(f"✅ Found model: {model.name}")
        latest_version = model.latest_versions[0]
        print(f"Latest version: {latest_version.version}, Run ID: {latest_version.run_id}")

        is_first_model = False
    else:
        print("❌ No matching model found in the registry.")
        print("📦 No model found — registering new model.")

        model_name = f"{customer_id}_{store_id}_{sku}"

        # Register the current model (assumes artifact path is "model")
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        # Add tags to the registered model
        client.set_registered_model_tag(name=model_name, key="customer_id", value=customer_id)
        client.set_registered_model_tag(name=model_name, key="store_id", value=store_id)
        client.set_registered_model_tag(name=model_name, key="sku", value=sku)

        # Output flags and metrics for GitHub Actions
        print("IS_FIRST_MODEL=true")
        print(f"NEW_METRIC={new_metric_value}")
        print(f"OLD_METRIC=NA")

        # Exit early — no model to compare against
        is_first_model = True

        exit(0)




    run = client.get_run(run_id=latest_version.run_id)

    if not run:
        raise ValueError(f"🚨 No run found for run_id: {run_id}")

    # ✅ Fetch the metric value (modify as needed)
    current_metric_name = "mape"  # Change this to the metric you need
    current_metric_value = run.data.metrics.get(current_metric_name, "N/A")

    print(f"✅ Fetched metric: {current_metric_name} = {current_metric_value} for run_id: {run_id}")


    # Print in KEY=VALUE format for GitHub Actions to parse
    # print(f"NEW_METRIC={new_metric_value}")
    # print(f"OLD_METRIC={current_metric_value}")



    print(f"{customer_id},{store_id},{sku},NEW_METRIC={new_metric_value},OLD_METRIC={current_metric_value},IS_FIRST_MODEL={is_first_model}")
