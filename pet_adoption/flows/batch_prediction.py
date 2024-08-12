import os
from datetime import datetime

from dotenv import load_dotenv
from prefect import flow

from pet_adoption.flows.tasks import (
    apply_model,
    get_production_model_run_id,
    load_model,
    preprocess_data,
    read_data_from_s3,
    simulate_batch_predictions,
)

load_dotenv()


@flow
def batch_prediction_flow() -> None:
    """
    The batch prediction flow orchestrated with Prefect.

    The batch prediction flow performs the following steps:
    - Read the test data from an S3 bucket.
    - Preprocess the test data.
    - Load the trained model form MLflow.
    - Make predictions using the loaded model.
    - Read the training data from S3 and apply the loaded models to get predictions.
    - Create monitoring metrics for the test set based on the train set.
    """
    datetime_now = datetime.now()

    # read test data from S3
    df_test = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_test.csv")

    # preprocess the data
    df_test = preprocess_data(df=df_test)

    # get the MLflow run id of the production model
    RUN_ID = get_production_model_run_id(bucket_name=os.getenv("S3_BUCKET_MLFLOW"), file_key="production_model.json")
    print(f"RUN_ID: {RUN_ID}")

    # load the model
    model = load_model(run_id=RUN_ID)

    # read the training data from S3 and apply the model to get predictions for monitoring
    df_train = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_train.csv")
    df_train = apply_model(df=df_train, model=model)

    simulate_batch_predictions(df_train=df_train, df_test=df_test, date_time=datetime_now, model=model)
    return None


if __name__ == "__main__":
    batch_prediction_flow()
