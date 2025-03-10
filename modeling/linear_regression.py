import mlflow
import mlflow.artifacts
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import dvc.api
import pandas as pd

import os

running_in_github = os.getenv("GITHUB_ACTIONS") == "true"

if running_in_github:
    data = pd.read_csv('dvc_storage_S3/files/md5/73/2b44631f0cf1242a668a04d542700a')

else:
    path = 'pre_processed_1_dvcs/merged_data.csv'
    repo = 'https://github.com/joel-axb/MLOps_test.git'
    version = 'v2' #git commit tag

    data_url = dvc.api.get_url(
        path = path,
        repo = repo,
        rev = version
        )



    data = pd.read_csv(data_url, sep=",")

test_period = ['2025-07', '2025-07']

# 데이터셋 로드
X = data.drop(columns = ['Datetime', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'])
y = data[['PowerConsumption_Zone3']]

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# run마다의 메타데이터 저장 위치 설정
mlflow.set_tracking_uri("/Users/joel/Documents/github/MLOps_test/mlruns")

# MLflow 실행 시작
mlflow.start_run()

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 메트릭 계산
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# 매개변수, 메트릭, 모델 기록
mlflow.log_param('model_type', 'linear_regression')
mlflow.log_metric(f"{test_period[0]}_{test_period[1]}", mse)

# 모델 저장
mlflow.sklearn.log_model(model, "model")
mlflow.log_artifact(__file__)

# MLflow 실행 종료
mlflow.end_run()

