import pandas as pd

# 파일 경로 설정
powers_path = "/Users/joel/Documents/github/MLOps_test/raw_data(Athena)/powers.csv"
weathers_path = "/Users/joel/Documents/github/MLOps_test/raw_data(Athena)/weathers.csv"

# 데이터 불러오기
powers_df = pd.read_csv(powers_path)
weathers_df = pd.read_csv(weathers_path)

# Datetime 컬럼을 datetime 형식으로 변환
powers_df["Datetime"] = pd.to_datetime(powers_df["Datetime"])
weathers_df["Datetime"] = pd.to_datetime(weathers_df["Datetime"])

# 시간 기준으로 병합 (inner join)
merged_df = pd.merge(powers_df, weathers_df, on="Datetime", how="inner")

# 병합된 데이터 확인
print(merged_df.head())

# CSV 파일로 저장 (선택사항)
merged_df.to_csv("/Users/joel/Documents/github/MLOps_test/pre_processed_1(S3)/merged_data.csv", index=False)
