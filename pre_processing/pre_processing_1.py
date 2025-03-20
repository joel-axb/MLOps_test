import pandas as pd
from athena_related import *


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

    # 전처리를 한다
    df = df.iloc[:-8]

    # CSV 파일로 저장
    # csv 파일로 저장이 되는 실파일은 dvc add -> push 를 통해 dvc 파일이 생성 저장?
    # 이 후 로컬에 저장된 실파일과 cache 파일은 삭제한다.
    # git push 한다.
    df.to_csv("/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv", index=False)
