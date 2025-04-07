import pandas as pd
from get_data_from_athena import fetch_athena_query_as_dataframe
import matplotlib.pyplot as plt
import os
import yaml

def prepare_dataset(save_path):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
    TOTAL_CONFIG_PATH = os.path.join(BASE_DIR, "../modeling/commons", "test_lists.yaml")
    # load config.yaml
    with open(TOTAL_CONFIG_PATH, "r") as f:
        total_config = yaml.safe_load(f)

    customer_list = []
    store_id_list = []



    for one_config in total_config:

        # exp_name = total_config['exp_name']
        customer_list.append(one_config["customer_id"])
        store_id_list.append(one_config["store_id"])

    customer_list = tuple(customer_list)
    store_id_list = tuple(store_id_list)
    
    # get tables from Athena
    # sellout_raw = fetch_athena_query_as_dataframe('get_sellout_timeseries')
    sellout_raw = fetch_athena_query_as_dataframe(
        query_name="get_sellout_timeseries",
        customer_id=customer_list,
        store_id=store_id_list
    )

    print()
    # sellout_derived = fetch_athena_query_as_dataframe('get_sellout_derived')
    sellout_derived = fetch_athena_query_as_dataframe(
        query_name="get_sellout_derived",
        customer_id=customer_list,
        store_id=store_id_list
    )

    # sellout_timeseries = fetch_athena_query_as_dataframe('get_sellout_timeseries')
    sellout_timeseries = fetch_athena_query_as_dataframe(
        query_name="get_sellout_timeseries",
        customer_id=customer_list,
        store_id=store_id_list
    )



    # data pre-processing ------
    
    # 첫 번째 병합 (겹치는 컬럼엔 _dup1 붙이기)
    merged_df_1 = pd.merge(
        sellout_raw,
        sellout_derived,
        on=["sku", "store_id", "customer_id", "forecast_dt"],
        how="left",
        suffixes=("", "_dup1")
    )

    # 두 번째 병합 (겹치는 컬럼엔 _dup2 붙이기)
    merged_df_2 = pd.merge(
        merged_df_1,
        sellout_timeseries,
        on=["sku", "store_id", "customer_id", "forecast_dt"],
        how="left",
        suffixes=("", "_dup2")
    )


    merged_df_2["sellout"] = merged_df_2["sellout"].astype(float)
    merged_df_2["sellout_raw"] = merged_df_2["sellout_raw"].astype(float)
    merged_df_2["forecast_dt"] = pd.to_datetime(merged_df_2["forecast_dt"])

    merged_df_2 = merged_df_2.sort_values("forecast_dt")

    merged_df_2 = merged_df_2[merged_df_2['forecast_dt']<'2025-02-01']


    merged_df_2["sellout_raw"] = merged_df_2["sellout_raw"].interpolate(method="linear")
    merged_df_2["sellout"] = merged_df_2["sellout"].interpolate(method="linear")


    # 중복된 컬럼 제거
    merged_df_2 = merged_df_2[[col for col in merged_df_2.columns if not col.endswith("_dup1") and not col.endswith("_dup2")]]

    # data pre-processing ------

    # Extract components
    merged_df_2['year'] = merged_df_2['forecast_dt'].dt.year
    merged_df_2['month'] = merged_df_2['forecast_dt'].dt.month
    merged_df_2['week'] = merged_df_2['forecast_dt'].dt.isocalendar().week
    merged_df_2['day'] = merged_df_2['forecast_dt'].dt.day
    merged_df_2['weekday'] = merged_df_2['forecast_dt'].dt.weekday  # e.g., Monday

    # merged_df_2 = merged_df_2.drop(columns=['forecast_dt'])
    # --------------------------

    # save final dataset
    print("Data pre-processing done")
    print(f"shape: {merged_df_2.shape}")
    print(f"columns: {merged_df_2.columns}")

    if save_path is None:
        merged_df_2.to_csv(save_path, index=False)

    else:
        merged_df_2.to_csv(save_path, index=False)

    return merged_df_2


if __name__ == "__main__": 


    prepare_dataset("/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv")
