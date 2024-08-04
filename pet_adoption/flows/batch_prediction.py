import os
from io import StringIO

import boto3
import mlflow
import pandas as pd
from catboost import Pool
from dotenv import load_dotenv
from mlflow.pyfunc import PyFuncModel
from prefect import flow, task

from pet_adoption.flows.model_training import (
    extract_report_data,
    monitor_model_performance,
    preprocess_data,
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


@task(name="Read Data from S3")
def read_data_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    """
    Read a CSV file from an S3 bucket and return it as a Pandas DataFrame.

    :param bucket_name: the name of the S3 bucket
    :param file_key: the full path to the CSV file
    :return: a Pandas DataFrame
    """
    # Create an S3 resource
    s3 = boto3.resource("s3")
    # Read the CSV file from the S3 bucket
    obj = s3.Object(bucket_name, file_key)
    csv_string = obj.get()["Body"].read().decode("utf-8")
    # Convert the CSV string to a DataFrame
    df = pd.read_csv(StringIO(csv_string))
    return df


@task(name="Load Model from MLflow")
def load_model(run_id: str) -> PyFuncModel:
    """
    Load the trained CatBoost model from MLflow.

    :param run_id: run ID of the MLflow run
    :return: the trained CatBoost model
    """
    # load the model from S3
    logged_model = f"s3://{os.getenv('S3_BUCKET_MLFLOW')}/{run_id}/artifacts/{os.getenv('MLFLOW_MODEL_NAME')}"
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


@task(name="Batch Prediction")
def make_batch_predictions(df: pd.DataFrame, model: PyFuncModel) -> pd.DataFrame:
    """
    Make batch predictions using the trained CatBoost model.

    :param df: the data on which to make predictions
    :param model: the trained CatBoost model
    :return: a DataFrame with the predictions
    """
    test_pool = Pool(data=df[CAT_FEATURES + NUM_FEATURES], cat_features=CAT_FEATURES)
    predictions = model.predict(test_pool)
    df["prediction"] = predictions
    return df


@flow
def batch_prediction_flow() -> None:
    """
    The batch prediction flow orchestrated with Prefect.

    The batch prediction flow performs the following steps:
    1. Read the test data from an S3 bucket.
    2. Preprocess the test data.
    3. Make batch predictions using the trained CatBoost model.
    """
    # read test data from S3
    df = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_test.csv")

    # preprocess the data
    df = preprocess_data(df)

    # load the model
    model = load_model(run_id="866d44d4cd71460e8e90cafb6213b3c1")
    print(type(model))

    # make batch predictions
    df = make_batch_predictions(df, model)

    # read the training data from S3 for monitoring
    df_train = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_train.csv")
    df_train = make_batch_predictions(df_train, model)

    # monitor model performance
    metrics_dict = monitor_model_performance(reference_data=df_train, current_data=df)
    save_dict_in_s3(metrics_dict, os.getenv("S3_BUCKET"), "data/monitoring_metrics_prediction.json")
    extract_report_data(batch_date="2024-08-03", metrics_dict=metrics_dict, db_name="predict_monitoring")
    return df


if __name__ == "__main__":
    batch_prediction_flow()
