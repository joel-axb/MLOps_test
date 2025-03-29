# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from models_common.prophet_test import run_prophet
from models_common.linear_regression import LinearRegressionModel
 
def get_model_runner(model_type):
    if model_type == "linear_regression":
        return LinearRegressionModel
    elif model_type == "random_forest":
        return RandomForestRegressor
    elif model_type == "prophet":
        return run_prophet  # 특화된 함수 반환
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
