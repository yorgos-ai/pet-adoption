import os
from datetime import date

from dotenv import load_dotenv
from prefect import flow

from pet_adoption.flows.tasks import (
    apply_model,
    extract_report_data,
    get_production_model_run_id,
    load_model,
    monitor_model_performance,
    preprocess_data,
    read_data_from_s3,
    save_dict_in_s3,
    store_data_in_s3,
)

load_dotenv()


@flow
def batch_prediction_flow() -> None:
    """
    The batch prediction flow orchestrated with Prefect.

    The batch prediction flow performs the following steps:
    1. Read the test data from an S3 bucket.
    2. Preprocess the test data.
    3. Load the trained model form MLflow.
    4. Make predictions using the loaded model.
    5. Read the training data from S3 and apply the loaded models to get predictions.
    6. Create monitoring metrics for the test set based on the train set.
    """
    today = date.today()
    batch_date = today.strftime("%Y-%m-%d")

    # read test data from S3
    df_test = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_test.csv")

    # preprocess the data
    df_test = preprocess_data(df_test)

    # get the MLflow run id of the production model
    RUN_ID = get_production_model_run_id(bucket_name=os.getenv("S3_BUCKET_MLFLOW"), file_key="production_model.json")
    print(f"RUN_ID: {RUN_ID}")

    # load the model
    model = load_model(run_id=RUN_ID)

    # make batch predictions
    df_test = apply_model(df_test, model)

    # store the test data in S3
    store_data_in_s3(
        df=df_test,
        bucket_name=os.getenv("S3_BUCKET"),
        file_key=f"predictions/{batch_date.replace('-', '_')}_df_test.csv",
    )

    # read the training data from S3 for monitoring
    df_train = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_train.csv")
    df_train = apply_model(df_train, model)

    # Evidently batch prediction monitoring metrics
    metrics_dict = monitor_model_performance(reference_data=df_train, current_data=df_test)
    save_dict_in_s3(metrics_dict, os.getenv("S3_BUCKET"), "data/monitoring_metrics_prediction.json")
    extract_report_data(batch_date=batch_date, metrics_dict=metrics_dict, db_name="predict_monitoring")

    return None


if __name__ == "__main__":
    batch_prediction_flow()
