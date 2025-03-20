import boto3
import os
import time
import pandas as pd
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# AWS_PROFILE 설정 (환경 변수에서 가져오기)
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")  # 기본값 "default"

# AWS 세션 설정 (Assume Role 자동 처리됨)
session = boto3.Session(profile_name=AWS_PROFILE)

# Athena & S3 클라이언트 생성
athena_client = session.client("athena")
s3_client = session.client("s3")

# S3 결과 저장 경로 (필수)

S3_BUCKET = "data-pipeline.prod.acrossb.net"  # 버킷 이름만 설정
S3_PREFIX = "tmp/mlops_test/athena-results/"  # Athena 결과 저장 디렉토리
S3_OUTPUT = f"s3://{S3_BUCKET}/{S3_PREFIX}"  # 최종 S3 저장 경로


# 실행할 Athena 쿼리 (쿼리에서 데이터베이스 직접 지정)
QUERY = """ SELECT *
            FROM analysis_data.integrated_orders 
            LIMIT 5;"""

def run_athena_query(query, s3_output):
    """Athena에서 쿼리를 실행하고, 실행 ID를 반환"""
    response = athena_client.start_query_execution(
        QueryString=query,
        ResultConfiguration={"OutputLocation": s3_output}
    )
    return response["QueryExecutionId"]

def get_query_status(query_execution_id):
    """쿼리 실행 상태 확인"""
    while True:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response["QueryExecution"]["Status"]["State"]
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            return status
        time.sleep(2)




def get_s3_result_file(query_execution_id):
    """쿼리 결과 파일의 S3 경로 반환"""
    return f"athena-results/{query_execution_id}.csv"



def get_query_results(query_execution_id):
    """쿼리 결과를 가져와 Pandas DataFrame으로 변환 (모든 페이지 가져오기)"""
    columns = []
    rows = []
    next_token = None

    while True:
        # 요청에 따라 첫 번째 호출 또는 페이지네이션 토큰을 사용한 호출
        if next_token:
            response = athena_client.get_query_results(QueryExecutionId=query_execution_id, NextToken=next_token)
        else:
            response = athena_client.get_query_results(QueryExecutionId=query_execution_id)

        # 컬럼명 추출 (첫 번째 호출에서만 설정)
        if not columns:
            columns = [col["Name"] for col in response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]

        # 행 데이터 추출
        for row in response["ResultSet"]["Rows"][1:]:  # 첫 번째 행은 헤더이므로 제외
            rows.append([col.get("VarCharValue", None) for col in row["Data"]])

        # 다음 페이지 확인
        next_token = response.get("NextToken")

        if not next_token:
            break  # 다음 페이지가 없으면 종료

    # DataFrame 변환
    return pd.DataFrame(rows, columns=columns)




def delete_s3_result(query_execution_id):
    """Athena 결과 파일을 S3에서 삭제"""
    object_key = get_s3_result_file(query_execution_id)
    s3_client.delete_object(Bucket=S3_BUCKET, Key=S3_PREFIX+object_key.split('/')[1])
    s3_client.delete_object(Bucket=S3_BUCKET, Key=S3_PREFIX+object_key.split('/')[1]+'.metadata')
    print(f"✅ S3 파일 삭제 완료: {object_key}")



if __name__ == "__main__": 

    # Athena에서 쿼리 실행
    query_execution_id = run_athena_query(QUERY, S3_OUTPUT)

    # 쿼리 상태 확인
    status = get_query_status(query_execution_id)
    if status != "SUCCEEDED":
        print(f"쿼리 실행 실패: {status}")
        exit()

    # 쿼리 결과 가져오기
    df = get_query_results(query_execution_id)

    # S3에서 즉시 결과 파일 삭제
    delete_s3_result(query_execution_id)

    # 데이터 확인
    print("Athena 데이터 가져오기 완료")
    print(df.head())
