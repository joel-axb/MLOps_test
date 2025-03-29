import pandas as pd
import dvc.api


def read_final_dataset(config):

    try:
        data = pd.read_csv('/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv')
            
    except:
        print("you alrady deleted the data in local!")
        print("I will get the most recent data you did dvc push")
        with dvc.api.open(
            path="../data_temp_storage/final_data.csv",
            repo=".",         # current repo
            rev=None          # ← None = workspace (latest even if uncommitted)
        ) as fd:
            data = pd.read_csv(fd)

    return data



