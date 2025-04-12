import mlflow
import pandas as pd
import dvc.api
import matplotlib.pyplot as plt
import os



class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        
        return self.model.predict(model_input)

    

def read_final_dataset(config):
    try:
        data = pd.read_csv('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')
    except:
        print("You already deleted the data in local!")
        print("I will get the most recent data you pushed via DVC")

        with dvc.api.open(
            path="data_temp_storage/final_data.csv",  # remove "../" because repo="." is already the root
            repo=".",         
            rev="HEAD"         # ← fetch latest committed DVC-tracked version
        ) as fd:
            data = pd.read_csv(fd)

    return data



def get_best_result_for_each_sku(exp_name):

    # 전체 실험(run) 정보 가져오기
    experiment_name = exp_name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # 예시: rmse 기준으로 가장 성능 좋은 모델 선택
    best_runs = (
        runs_df
        .sort_values("metrics.mape", ascending=True)
        .groupby(["params.customer_id", "params.store_id", "params.sku", "params.test_end_dt", "params.test_start_dt"])  # SKU 기준으로 그룹화
        .first()  # 각 SKU별로 가장 좋은 run 하나
        .reset_index()
    )


    print(best_runs[["params.test_start_dt", "params.test_end_dt", "params.customer_id", "params.store_id", "params.sku", "params.model_type", "metrics.mape", "run_id"]])
    
    best_runs.columns = [col.split(".")[-1] for col in best_runs.columns]

    tuples = [
    (row.customer_id, row.store_id, row.sku, row.run_id)
    for row in best_runs.itertuples(index=False)]

    return best_runs, tuples



def get_visualized_result(preds_df, targets_array, title="Time Series Prediction vs Actual"):

    # 복사본 만들고 실제값 컬럼 추가
    result_df = preds_df.copy()
    result_df["actual"] = targets_array

    # forecast_dt를 datetime으로 변환
    result_df["forecast_dt"] = pd.to_datetime(result_df["forecast_dt"])
    result_df.set_index("forecast_dt", inplace=True)

    # 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(result_df.index, result_df["sellout_raw"], label="Predicted", marker='o')
    plt.plot(result_df.index, result_df["actual"], label="Actual", marker='x')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sellout")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    filename = "time_series_result.png"

    base_filename, file_extension = os.path.splitext(filename)
    counter = 1

    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}{file_extension}"
        counter += 1

    plt.savefig(filename)  


def load_query_template(customer_id, store_id):
    with open("/Users/joel/Documents/github/MLOps_test/pre_processing/queries/get_master_items.sql", "r") as f:
        sql = f.read()
    return sql.format(customer_id=customer_id, store_id=store_id)