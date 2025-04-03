import mlflow
from mlflow.tracking import MlflowClient
import os
import joblib
import mlflow.pyfunc

def retreive_artifacts():

    original_value = os.getenv("AWS_PROFILE")
    os.environ["AWS_PROFILE"] = "axb-dev-general"


    client = MlflowClient()

    MODEL_NAME = "cosrx_shopify_SC40AS01"
    STAGE = "None"

    # Get latest model version
    model_version = client.get_latest_versions(MODEL_NAME, stages=[STAGE])[0]


    # Download artifacts selectively
    def download_specific_artifacts(run_id, artifacts, dst_path="./"):
        os.makedirs(dst_path, exist_ok=True)
        downloaded_files = {}
        for artifact in artifacts:
            local_path = client.download_artifacts(run_id, artifact, dst_path)
            downloaded_files[artifact] = local_path
            print(f"Downloaded {artifact} to {local_path}")
        return downloaded_files



    artifacts_to_download = ["queries", "random_forest_model/python_model.pkl", "pre_processing.py", "random_forest.py", "get_data_from_athena.py"]
    artifact_paths = download_specific_artifacts(model_version.run_id, artifacts_to_download)


    if original_value is None:
        del os.environ["AWS_PROFILE"]
    else:
        os.environ["AWS_PROFILE"] = original_value

    return model_version.run_id

def pre_processing():

    prepare_dataset("./final_dataset.csv")
 


if __name__ == "__main__":

   

    run_id = retreive_artifacts()

    from get_data_from_athena import fetch_athena_query_as_dataframe
    from pre_processing import prepare_dataset
    from random_forest import Model

    final_df = prepare_dataset(None)

    exp_name = "None"
    sku = "None"
    PREPROCESSING_PATH = "None"

    train_mask = final_df["forecast_dt"] < "2025-01-01"
    val_mask = final_df["forecast_dt"] >= "2025-01-01"

    X = final_df.drop(columns=["sellout_raw", "sku", "customer_id", "store_id"])
    y = final_df[['forecast_dt', 'sellout_raw']]

    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y['sellout_raw'][train_mask]
    y_val = y['sellout_raw'][val_mask]


    model = Model(X_train, X_val, y_train, y_val, final_df, exp_name, sku, PREPROCESSING_PATH)
    print(model.pre_processing_2())

    model_uri = f"runs:/{run_id}/random_forest_model"

    loaded_model = mlflow.pyfunc.load_model(model_uri)  # Adjust path if needed

    # Step 3: Predict
    preds = loaded_model.predict(X_val)

    # Step 4: View or save predictions
    print("✅ Predictions:", preds)