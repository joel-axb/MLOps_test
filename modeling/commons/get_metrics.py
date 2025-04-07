import mlflow
import pandas as pd
from common_functions import get_best_result_for_each_sku
import argparse

client = mlflow.tracking.MlflowClient()

parser = argparse.ArgumentParser(description="Run experiment with a given name")
parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
args = parser.parse_args()
exp_name = args.exp_name


bests = get_best_result_for_each_sku(exp_name)

if not bests:
    raise ValueError("🚨 No best runs found!")

results_list = []

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

    # 기본값
    current_metric_value = "NA"
    is_first_model = False
    status = ""

    if not results:
        # 등록
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        client.set_registered_model_tag(name=model_name, key="customer_id", value=customer_id)
        client.set_registered_model_tag(name=model_name, key="store_id", value=store_id)
        client.set_registered_model_tag(name=model_name, key="sku", value=sku)

        is_first_model = True
        status = "first_model_registered"

    else:
        model = results[0]
        latest_version = model.latest_versions[0]
        existing_run = client.get_run(run_id=latest_version.run_id)

        if not existing_run:
            print(f"🚨 No run found for existing model version: {latest_version.run_id}")
            continue

        current_metric_value = existing_run.data.metrics.get(new_metric_name, "N/A")

        try:
            if float(new_metric_value) < float(current_metric_value):
                model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri=model_uri, name=model_name)

                client.set_registered_model_tag(name=model_name, key="customer_id", value=customer_id)
                client.set_registered_model_tag(name=model_name, key="store_id", value=store_id)
                client.set_registered_model_tag(name=model_name, key="sku", value=sku)

                status = "improved_and_registered"
            else:
                status = "not_registered_worse_or_equal"
        except Exception as e:
            status = f"comparison_failed: {e}"

    # 결과 저장
    results_list.append({
        "customer_id": customer_id,
        "store_id": store_id,
        "sku": sku,
        "new_metric": new_metric_value,
        "old_metric": current_metric_value,
        "is_first_model": is_first_model,
        "status": status
    })

# ✅ DataFrame 생성
results_df = pd.DataFrame(results_list)

# ✅ 저장 (옵션)
results_df.to_csv("model_comparison_result.csv", index=False)
print("📄 model_comparison_result.csv saved!")

# ✅ 출력 (옵션)
print(results_df)
