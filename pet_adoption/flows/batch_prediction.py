import os
from io import StringIO

import boto3
import mlflow
import pandas as pd
from catboost import Pool
from dotenv import load_dotenv
from prefect import flow, task

from pet_adoption.flows.model_training import preprocess_data

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


@task(name="Batch Prediction")
def make_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make batch predictions using the trained CatBoost model.

    :param df: the data on which to make predictions
    :param model: the trained CatBoost model
    :return: a DataFrame with the predictions
    """
    # Make predictions on the data
    RUN_ID = "a96b5c2d9cdb4f3293e84aa49a1bde66"

    # load the model from S3
    logged_model = f"s3://{os.getenv('S3_BUCKET_MLFLOW')}/{RUN_ID}/artifacts/os.getenv('MLFLOW_MODEL_NAME')"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    test_pool = Pool(data=df[CAT_FEATURES + NUM_FEATURES], cat_features=CAT_FEATURES)
    predictions = loaded_model.predict(test_pool)
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
    df = read_data_from_s3(bucket_name=os.getenv("S3_BUCKET"), file_key="data/df_test.csv")
    df = preprocess_data(df)
    df = make_batch_predictions(df)
    return df


if __name__ == "__main__":
    batch_prediction_flow()
