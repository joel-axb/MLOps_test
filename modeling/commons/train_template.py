# root 경로 설정용 setup
import subprocess, sys, os
# 현재 파일 위치 기준 절대 경로 추출
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # setup.py 기준
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", project_root])


# 필요한 packages 로드
import yaml
import importlib.util
import argparse


from modeling.commons.common_functions import read_final_dataset, get_visualized_result
from modeling.commons.model_factory import get_model_runner as build_model



parser = argparse.ArgumentParser(description="Run experiment with a given name")
parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
args = parser.parse_args()
exp_name = args.exp_name


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
TOTAL_CONFIG_PATH = os.path.join(BASE_DIR, "test_lists.yaml")
# load config.yaml
with open(TOTAL_CONFIG_PATH, "r") as f:
    total_config = yaml.safe_load(f)

customer_list = []
store_id_list = []
sku_lists = []
all_list = []
validation_windows = []

for one_config in total_config:

    # exp_name = total_config['exp_name']
    customer_list.append(one_config["customer_id"])
    store_id_list.append(one_config["store_id"])
    sku_lists.append(one_config["sku_list"]) #list
    all_list.append(one_config["all"])
    validation_windows.append(one_config["validation_windows"])

# get config info
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
CONFIG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "configs"))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

MODELS_COMMON_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models_common"))
MODELS_COMMON_PATH = os.path.join(MODELS_COMMON_DIR, )
file_names = [f[:-3] for f in os.listdir(MODELS_COMMON_PATH) if os.path.isfile(os.path.join(MODELS_COMMON_PATH, f))]

for customer_id, store_id, sku_list, is_all, validation_window in zip(customer_list, store_id_list, sku_lists, all_list, validation_windows):

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
        for i in validation_window:

            # train_mask = data["forecast_dt"] < "2024-11-01"
            # val_mask = data["forecast_dt"] >= "2024-11-01"

            train_mask = data["forecast_dt"] < i[0]
            val_mask = (data["forecast_dt"] >= i[0]) & (data["forecast_dt"] < i[1])

            X = data.drop(columns=["sellout_raw", "sku", "customer_id", "store_id"])
            y = data[['forecast_dt', 'sellout_raw']]

            X_train = X[train_mask]
            X_val = X[val_mask]
            y_train = y['sellout_raw'][train_mask]
            y_val = y['sellout_raw'][val_mask]


            for one_model_type in model_type:
                print(one_model_type)
                try:
                    # when the model is in models_common
                    if one_model_type in file_names:
                        model_runner = build_model(one_model_type)
                        model = model_runner(X_train, X_val, y_train, y_val, data, 
                                            exp_name, customer, store_id, sku, PREPROCESSING_PATH)
                        
                        y_pred = model.run()
                        # get_visualized_result(y[val_mask], y_pred)

                    else:  # custom model
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

                        # get_visualized_result(y[val_mask], y_pred.values)

                except Exception as e:
                    print(f"❌ [SKIPPED] Failed to run model: {one_model_type} | {customer}-{store_id}-{sku}")
                    print(f"↪️  Reason: {e}")
                    continue  # 다음 모델로 넘어감