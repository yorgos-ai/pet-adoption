import os

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
)

TARGET = "AdoptionLikelihood"
NUM_FEATURES = [
    "AgeMonths",
    "WeightKg",
    "Vaccinated",
    "HealthCondition",
    "TimeInShelterDays",
    "AdoptionFee",
    "PreviousOwner",
]
CAT_FEATURES = ["PetType", "Breed", "Color", "Size"]
RANDOM_STATE = 42

load_dotenv()


RUN_ID = "9d7c094197a34b93bee0d77a1b042075"


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
    # read test data from S3
    df = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_test.csv")

    # preprocess the data
    df = preprocess_data(df)

    # get the MLflow run id of the production model
    RUN_ID = get_production_model_run_id(bucket_name=os.getenv("S3_BUCKET_MLFLOW"), file_key="production_model.json")
    print(f"RUN_ID: {RUN_ID}")

    # load the model
    model = load_model(run_id=RUN_ID)

    # make batch predictions
    df = apply_model(df, model)

    # read the training data from S3 for monitoring
    df_train = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_train.csv")
    df_train = apply_model(df_train, model)

    # Evidently batch prediction monitoring metrics
    metrics_dict = monitor_model_performance(reference_data=df_train, current_data=df)
    save_dict_in_s3(metrics_dict, os.getenv("S3_BUCKET"), "data/monitoring_metrics_prediction.json")
    extract_report_data(batch_date="2024-08-03", metrics_dict=metrics_dict, db_name="predict_monitoring")
    return df


if __name__ == "__main__":
    batch_prediction_flow()
