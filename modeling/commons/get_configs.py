import argparse
import sys
import os
import json
import yaml

# 상위 디렉토리 기준 상대경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modeling.commons.common_functions import load_query_template
from pre_processing.get_data_from_athena import fetch_athena_query_as_dataframe

def save_dict_list_as_python_file(dict_list, output_path):
    for d in dict_list:
        d["dat_preprocessing"] = "pre_processing.py"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict_list, f, indent=2, ensure_ascii=False)

    print(f"✅ 저장 완료: {output_path}")



def get_configs():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
    TOTAL_CONFIG_PATH = os.path.join(BASE_DIR, "test_lists.yaml")
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




    store_items = fetch_athena_query_as_dataframe(
        query_name="get_master_items",
        customer_id=customer_list,
        store_id=store_id_list
    )

    print(store_items)
    
    target_df = store_items[['customer_id', 'store_id', 'ssku']]  # 웨 customer_id, store_id에 따라 raw_data.product_master_item 에서 가져온다.

    config_list = []


    for row in target_df.itertuples(index=False):
        sku_template = {
            "customer_id": row.customer_id,
            "store_id": row.store_id,
            "sku": row.ssku,
            "model_type": ["random_forest"]
        }
        config_list.append(sku_template)

    working_dir = os.getcwd()
    print(working_dir)
    
    save_dict_list_as_python_file(config_list, "../configs/config.yaml")




if __name__ == "__main__":


    

    get_configs()
