import yaml
from common_functions import read_final_dataset
from model_factory import get_model_runner as build_model
from sklearn.model_selection import train_test_split
import importlib.util, os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--customer", required=True)
parser.add_argument("--sku", required=True)
args = parser.parse_args()

# get config info
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
CONFIG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "configs", args.customer, args.sku))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

# load config.yaml
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

customer = config["customer_id"]
store_id = config["store_id"]
sku = config["sku"]
customed = config["customed"]
model_type = config["model_type"]
data_preprocessing = config['data_preprocessing']

PREPROCESSING_DIR = os.path.normpath(os.path.join(BASE_DIR, "../..", "pre_processing"))
PREPROCESSING_PATH = os.path.join(PREPROCESSING_DIR, "pre_processing.py")

# set experiment_name
exp_name = f'{customer}_{store_id}_{sku}_{model_type}'
print(f"exp_name is: {exp_name}")

# get newly pre-processed data or dvc pushed data
data = read_final_dataset(config)

# get train and test set
X = data.drop(columns=["count"])
y = data['count']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# when the model is in models_common
if customed == False : 
    model_runner = build_model(model_type)
    model = model_runner(X_train, X_val, y_train, y_val, data, exp_name, PREPROCESSING_PATH)
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
    model_instance = ModelClass(X_train, X_val, y_train, y_val, data, exp_name, PREPROCESSING_PATH)
    model_instance.run()