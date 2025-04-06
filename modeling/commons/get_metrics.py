import mlflow
from common_functions import get_best_result_for_each_sku

client = mlflow.tracking.MlflowClient()
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

    # ✅ 1. 등록된 모델이 없는 경우
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

    # ✅ 2. 등록된 모델이 있는 경우
    model = results[0]
    latest_version = model.latest_versions[0]
    existing_run = client.get_run(run_id=latest_version.run_id)

    if not existing_run:
        print(f"🚨 No run found for existing model version: {latest_version.run_id}")
        continue

    current_metric_value = existing_run.data.metrics.get(new_metric_name, "N/A")
    is_first_model = False

    try:
        if float(new_metric_value) < float(current_metric_value):
            print(f"✅ New model is better for {model_name}! Registering new version.")

            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri=model_uri, name=model_name)

            client.set_registered_model_tag(name=model_name, key="customer_id", value=customer_id)
            client.set_registered_model_tag(name=model_name, key="store_id", value=store_id)
            client.set_registered_model_tag(name=model_name, key="sku", value=sku)
        else:
            print(f"⚠️ New model is worse or equal for {model_name}. Not registering.")
    except Exception as e:
        print(f"⚠️ Metric comparison failed: {e}")

    # ✅ 최종 출력
    print(
        f"{customer_id},{store_id},{sku},"
        f"NEW_METRIC={new_metric_value},OLD_METRIC={current_metric_value},IS_FIRST_MODEL={is_first_model}"
    )
