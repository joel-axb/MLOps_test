import mlflow
import os
from dotenv import load_dotenv
from common_functions import get_best_result_for_each_sku

# # Load env variables
# load_dotenv()

client = mlflow.tracking.MlflowClient()

# Run from GitHub Actions ENV
# run_id_env = os.getenv("RUN_ID")
# customer_id_env = os.getenv("CUSTOMER_ID")
# sku_env = os.getenv("SKU")

# Get list of best runs
bests = get_best_result_for_each_sku()

if not bests:
    raise ValueError("🚨 No best runs found!")

for one_tuple in bests:
    customer_id, store_id, sku, run_id = one_tuple

    run = client.get_run(run_id=run_id)
    if not run:
        print(f"🚨 No run found for run_id: {run_id}")
        continue

    new_metric_name = "mape"
    new_metric_value = run.data.metrics.get(new_metric_name, "N/A")

    model_name = f"{customer_id}_{store_id}_{sku}"
    filter_str = (
        f"name='{model_name}' and "
        f"tag.customer_id='{customer_id}' and "
        f"tag.store_id='{store_id}' and "
        f"tag.sku='{sku}'"
    )

    results = client.search_registered_models(filter_string=filter_str)

    # 등록된 모델이 없는 경우
    if not results:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        client.set_registered_model_tag(name=model_name, key="customer_id", value=customer_id)
        client.set_registered_model_tag(name=model_name, key="store_id", value=store_id)
        client.set_registered_model_tag(name=model_name, key="sku", value=sku)

        is_first_model = True

        print(
            f"{customer_id},{store_id},{sku},"
            f"NEW_METRIC={new_metric_value},OLD_METRIC=NA,IS_FIRST_MODEL={is_first_model}"
        )
        continue

    # 등록된 모델이 있을 경우
    model = results[0]
    latest_version = model.latest_versions[0]
    existing_run = client.get_run(run_id=latest_version.run_id)

    if not existing_run:
        print(f"🚨 No run found for existing model version: {latest_version.run_id}")
        continue

    current_metric_value = existing_run.data.metrics.get(new_metric_name, "N/A")
    is_first_model = False

    print(
        f"{customer_id},{store_id},{sku},"
        f"NEW_METRIC={new_metric_value},OLD_METRIC={current_metric_value},IS_FIRST_MODEL={is_first_model}"
    )
