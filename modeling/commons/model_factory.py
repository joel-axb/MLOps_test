# root 경로 설정용 setup
import subprocess, sys, os
# 현재 파일 위치 기준 절대 경로 추출
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # setup.py 기준
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", project_root])

from modeling.models_common.prophet_test import Model as Prophet
from modeling.models_common.linear_regression import Model as LinearRegression
from modeling.models_common.random_forest import Model as RandomForest

 
def get_model_runner(model_type):
    if model_type == "linear_regression":
        return LinearRegression
    elif model_type == "random_forest":
        return RandomForest
    elif model_type == "prophet_test":
        return Prophet  # 특화된 함수 반환
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
