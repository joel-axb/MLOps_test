import boto3
import os
import time
import pandas as pd
from dotenv import load_dotenv

# load .env
load_dotenv()

session = boto3.Session()
athena_client = session.client("athena")
s3_client = session.client("s3")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 commons/
S3_BUCKET = "data-pipeline.prod.acrossb.net"  # 버킷 이름만 설정
S3_PREFIX = "tmp/mlops_test/athena-results/"  # Athena 결과 저장 디렉토리
S3_OUTPUT = f"s3://{S3_BUCKET}/{S3_PREFIX}"  # 최종 S3 저장 경로


def read_query_file(QUERY_FILE_PATH):

    with open(QUERY_FILE_PATH, "r") as f:
        query = f.read()
    return query


def run_athena_query(query, s3_output):

    response = athena_client.start_query_execution(
        QueryString=query,
        ResultConfiguration={"OutputLocation": s3_output}
    )
    return response["QueryExecutionId"]


def get_query_status(query_execution_id):

    while True:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response["QueryExecution"]["Status"]["State"]
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            return status
        time.sleep(2)


def get_s3_result_file(query_execution_id):
    return f"athena-results/{query_execution_id}.csv"


def get_query_results(query_execution_id):

    columns = []
    rows = []
    next_token = None

    while True:

        if next_token:
            response = athena_client.get_query_results(QueryExecutionId=query_execution_id, NextToken=next_token)
        else:
            response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        if not columns:
            columns = [col["Name"] for col in response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
        for row in response["ResultSet"]["Rows"][1:]: 
            rows.append([col.get("VarCharValue", None) for col in row["Data"]])
        next_token = response.get("NextToken")
        if not next_token:
            break 

    return pd.DataFrame(rows, columns=columns)



def delete_s3_result(query_execution_id):
    """Athena 결과 파일을 S3에서 삭제"""
    object_key = get_s3_result_file(query_execution_id)
    s3_client.delete_object(Bucket=S3_BUCKET, Key=S3_PREFIX+object_key.split('/')[1])
    s3_client.delete_object(Bucket=S3_BUCKET, Key=S3_PREFIX+object_key.split('/')[1]+'.metadata')

    print(f"✅ S3 파일 삭제 완료: {object_key}")



def fetch_athena_query_as_dataframe(query_name: str) -> pd.DataFrame:

    query_dir = os.path.normpath(os.path.join(BASE_DIR, "queries"))
    query_path = os.path.join(query_dir, f"{query_name}.sql")


    # query_path = f'pre_processing/queries/{query_name}.sql'
    query = read_query_file(query_path)
    execution_id = run_athena_query(query, S3_OUTPUT)
    status = get_query_status(execution_id)

    if status != "SUCCEEDED":
        raise RuntimeError(f"쿼리 실행 실패: {status}")

    df = get_query_results(execution_id)
    delete_s3_result(execution_id)

    print(f"✅ [{query_name}.sql] Athena 데이터 로딩 완료")
    return df

