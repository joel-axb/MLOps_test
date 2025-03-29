import pandas as pd
from get_data_from_athena import fetch_athena_query_as_dataframe


if __name__ == "__main__": 
    
    # get tables from Athena
    orders = fetch_athena_query_as_dataframe('get_orders')
    sellout_derived = fetch_athena_query_as_dataframe('get_sellout_derived')
    sellout_timeseries = fetch_athena_query_as_dataframe('get_sellout_timeseries')


    # data pre-processing ------
    
    df = orders





    # --------------------------

    # save final dataset
    print("Data pre-processing done")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns}")
    df.to_csv("/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv", index=False)
