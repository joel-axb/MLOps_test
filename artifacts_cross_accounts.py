import asyncio
from prefect.client import get_client
import mlflow
import os


async def trigger_env(deployment_name: str, parameters):

    async with get_client() as client:

        deployments = await client.read_deployments()
        deployment = next((d for d in deployments if d.name == deployment_name), None)

        flow_run = await client.create_flow_run_from_deployment(
            deployment_id=deployment.id,

            parameters = parameters
        )

        print(f"✅ Triggered flow run for DEV: AWS Cross Account Data Pipeline Publisher")


# 예: dev 환경 실행
if __name__ == "__main__":
    

    # step_1: model_registry (prd)에 저장된 모델들 각각의 최신 버젼 run_id 들을 가져온다. 

    # step_2: 각각의 run에 대한 artifact와 사용되었던 찾아서 dev 환경에서 저장한다 (prod->dev)
        # artifact 만으로는 어느 customer_id, store, sku 정보가 없기 때문에 해당 정보를 바탕으로 s3 directory를 구성한다
        # Ex) copied_artifacts - cosrx - shopify - sku_001 - artifact

    # 하나의 run 복사
    parameters = {
                    "detail": {
                                "from_env": "dev",
                                "to_env": "dev",
                                "type": "s3-to-s3",
                                "args": {
                                    "s3_copy_descriptors":[{
                                    "s3_from_bucket": "data-pipeline.dev.acrossb.net",
                                    "s3_to_bucket": "data-pipeline.dev.acrossb.net",
                                    "copy_details": [

                                        {
                                            "source_s3_prefix": "tmp/100",
                                            "target_s3_prefix": "tmp/100_copied_2"
                                        }
                                    ]
                                    }],
                                    "config": {
                                    "overwrite_if_exists": False
                                    }
                                }
                            }
                        }

    # asyncio.run(trigger_env("DEV: AWS Cross Account Data Pipeline Publisher", parameters=parameters))


    # step_4: 각 model 마다 적용된 pre_processing.py를 이용해서 아테나 테이블의 데이터를 전처리하고 model에 넣어서 예측하고 다시 아테나에 적재한다.
    original_value = os.getenv("AWS_PROFILE")
    os.environ["AWS_PROFILE"] = "axb-dev-general"
    
    model_uri = "s3://data-pipeline.dev.acrossb.net/tmp/117/f9cd73066ffa470283289263713467c8/artifacts/random_forest_model/"
    preprocessing_uri = "s3://data-pipeline.dev.acrossb.net/tmp/117/f9cd73066ffa470283289263713467c8/artifacts/preprocessing_model/"

    
    # preprocessing_model = mlflow.pyfunc.load_model(preprocessing_uri)
    # output = preprocessing_model.predict("/Users/joel/Documents/github/MLOps_test/data_temp_storage/final_data.csv")


    model = mlflow.pyfunc.load_model(preprocessing_uri) 







