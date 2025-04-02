import pandas as pd
from get_data_from_athena import fetch_athena_query_as_dataframe


if __name__ == "__main__": 
    
    # get tables from Athena
    orders = fetch_athena_query_as_dataframe('get_orders')
    sellout_derived = fetch_athena_query_as_dataframe('get_sellout_derived')
    sellout_timeseries = fetch_athena_query_as_dataframe('get_sellout_timeseries')


    # data pre-processing ------
    
    df = orders

    df['order_date'] = pd.to_datetime(df['order_date'])

    # Extract components
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['week'] = df['order_date'].dt.isocalendar().week
    df['day'] = df['order_date'].dt.day
    df['weekday'] = df['order_date'].dt.weekday  # e.g., Monday

    # df = df.drop(columns=['order_date'])
    # --------------------------

    # save final dataset
    print("Data pre-processing done")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns}")
    df.to_csv("/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv", index=False)
