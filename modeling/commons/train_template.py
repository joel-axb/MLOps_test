import yaml
from common_functions import read_final_dataset
from model_factory import get_model_runner as build_model
from sklearn.model_selection import train_test_split
import importlib.util, os
import pandas as pd
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--customer", required=True)
# parser.add_argument("--sku", required=True)
# args = parser.parse_args()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
TOTAL_CONFIG_PATH = os.path.join(BASE_DIR, "test_lists.yaml")
# load config.yaml
with open(TOTAL_CONFIG_PATH, "r") as f:
    total_config = yaml.safe_load(f)


exp_name = total_config['exp_name']
customer = total_config["customer_id"]
store_id = total_config["store_id"]
sku_list = total_config["sku_list"] #list

# get config info
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
CONFIG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "configs", customer, store_id))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

MODELS_COMMON_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models_common"))
MODELS_COMMON_PATH = os.path.join(MODELS_COMMON_DIR, )
file_names = [f[:-3] for f in os.listdir(MODELS_COMMON_PATH) if os.path.isfile(os.path.join(MODELS_COMMON_PATH, f))]


# load config.yaml
with open(CONFIG_PATH, "r") as f:
    configs = yaml.safe_load(f)


filtered_configs = [config for config in configs if config["sku"] in sku_list]


for config in filtered_configs:

    customer = config["customer_id"]
    store_id = config["store_id"]
    sku = config["sku"]
    customed = config["customed"]
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
    
    # get train and test set
    X = data.drop(columns=["count", "sku"])
    y = data['count']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for one_model_type in model_type:

        # when the model is in models_common
        if one_model_type in file_names : 
            model_runner = build_model(one_model_type)
            model = model_runner(X_train, X_val, y_train, y_val, data, 
                                exp_name, sku, PREPROCESSING_PATH)
            model.run()



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
                                        exp_name, sku, PREPROCESSING_PATH)
            model_instance.run()