import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from models_common.prophet_test import Model as Prophet
from models_common.linear_regression import Model as LinearRegression
from models_common.random_forest import Model as RandomForest

 
def get_model_runner(model_type):
    if model_type == "linear_regression":
        return LinearRegression
    elif model_type == "random_forest":
        return RandomForest
    elif model_type == "prophet_test":
        return Prophet  # 특화된 함수 반환
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
