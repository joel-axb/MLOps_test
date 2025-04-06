import yaml
from common_functions import read_final_dataset, get_visualized_result
from model_factory import get_model_runner as build_model
from sklearn.model_selection import train_test_split
import importlib.util, os
import pandas as pd
import argparse


exp_name = 'testtest_6'
# parser = argparse.ArgumentParser()
# parser.add_argument("--customer", required=True)
# parser.add_argument("--sku", required=True)
# args = parser.parse_args()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
TOTAL_CONFIG_PATH = os.path.join(BASE_DIR, "test_lists.yaml")
# load config.yaml
with open(TOTAL_CONFIG_PATH, "r") as f:
    total_config = yaml.safe_load(f)

customer_list = []
store_id_list = []
sku_lists = []
all_list = []


for one_config in total_config:

    # exp_name = total_config['exp_name']
    customer_list.append(one_config["customer_id"])
    store_id_list.append(one_config["store_id"])
    sku_lists.append(one_config["sku_list"]) #list
    all_list.append(one_config["all"])

# get config info
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
CONFIG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "configs"))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

MODELS_COMMON_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models_common"))
MODELS_COMMON_PATH = os.path.join(MODELS_COMMON_DIR, )
file_names = [f[:-3] for f in os.listdir(MODELS_COMMON_PATH) if os.path.isfile(os.path.join(MODELS_COMMON_PATH, f))]

for customer_id, store_id, sku_list, is_all in zip(customer_list, store_id_list, sku_lists, all_list):

    # load config.yaml
    with open(CONFIG_PATH, "r") as f:
        configs = yaml.safe_load(f)

    if is_all != True:
        filtered_configs = [config for config in configs 
                                if config["sku"] in sku_list and
                                   config["customer_id"] == customer_id and
                                   config["store_id"] == store_id]
    else:
        filtered_configs = [config for config in configs if config["customer_id"] == customer_id and
                                   config["store_id"] == store_id]

    for config in filtered_configs:

        customer = config["customer_id"]
        store_id = config["store_id"]
        sku = config["sku"]
        # customed = config["customed"]
        model_type = config["model_type"] #list
        data_preprocessing = config["dat_preprocessing"]

        # Build preprocessing path]
        PREPROCESSING_DIR = os.path.normpath(os.path.join(BASE_DIR, "../..", "pre_processing"))
        PREPROCESSING_PATH = os.path.join(PREPROCESSING_DIR, data_preprocessing)


        # set experiment_name
        # exp_name = f'{customer}_{store_id}_{sku}_{model_type}'
        print(f"exp_name is: {exp_name}")

        # get newly pre-processed data or dvc pushed data
        data = read_final_dataset(config)
        data = data[data['sku']==sku]
        
        # # get train and test set  
        # X = data.drop(columns=["sellout_raw", "sku", "customer_id", "store_id"])
        # y = data['sellout_raw']
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 날짜 기준으로 train/val 나누기
        train_mask = data["forecast_dt"] < "2024-11-01"
        val_mask = data["forecast_dt"] >= "2024-11-01"

        X = data.drop(columns=["sellout_raw", "sku", "customer_id", "store_id"])
        y = data[['forecast_dt', 'sellout_raw']]

        X_train = X[train_mask]
        X_val = X[val_mask]
        y_train = y['sellout_raw'][train_mask]
        y_val = y['sellout_raw'][val_mask]



        for one_model_type in model_type:

            # when the model is in models_common
            if one_model_type in file_names : 
                model_runner = build_model(one_model_type)
                model = model_runner(X_train, X_val, y_train, y_val, data, 
                                    exp_name, customer, store_id, sku, PREPROCESSING_PATH)
                y_pred = model.run()

                get_visualized_result(y[val_mask], y_pred)


            # when the model is a customed one
            else: 
                module_name = f'{customer}_{store_id}_{sku}'
                model_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "models_customed"))
                model_path = os.path.join(model_dir, f"{module_name}.py")

                spec = importlib.util.spec_from_file_location(module_name, model_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                ModelClass = getattr(module, 'Model')
                model_instance = ModelClass(X_train, X_val, y_train, y_val, data, 
                                            exp_name, customer, store_id, sku, PREPROCESSING_PATH)
                y_pred = model_instance.run()

                get_visualized_result(y[val_mask], y_pred.values)