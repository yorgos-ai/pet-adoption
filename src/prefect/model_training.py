# from src.data_loader import read_csv_from_s3
import os
from typing import List, Tuple

import boto3
import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.model_selection import train_test_split

from src.evaluate import classification_metrics
from src.feature_engineering import numerical_cols_as_float, object_cols_as_category
from src.utils import get_project_root, log_classification_report_to_mlflow, log_confusion_matrix_to_mlflow

os.environ["AWS_PROFILE"] = (
    "mlops-zoomcamp"  # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials
)
TRACKING_SERVER_HOST = (
    # "ec2-3-249-16-206.eu-west-1.compute.amazonaws.com"  # fill in with the public DNS of the EC2 instance
    "sqlite:///mlflow.db"  # use this for SQLite tracking server
)
MLFLOW_EXPERIMENT = "CatBoost model"  # fill in with the name of your MLflow experiment
S3_BUCKET_MLFLOW = "mlflow-artifacts-pet-adoption"  # the s3 bucket to store MLflow artifacts
S3_BUCKET = "pet-adoption-mlops"  # the s3 bucket to store the data
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


@task(name="MLflow Setup")
def setup_mlflow():
    """
    Set up the MLflow tracking server and create an experiment.

    :return: None
    """
    # mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)

    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    else:
        # Create a new experiment with the given name and S3 path as the artifact folder
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=f"s3://{S3_BUCKET_MLFLOW}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)


@task(name="Data Ingestion")
def read_data() -> pd.DataFrame:
    """
    Read the pet adoption data from the data directory.

    :return: a Pandas DataFrame
    """
    project_root = get_project_root()
    data_path = project_root / "data" / "pet_adoption_data.csv"
    df = pd.read_csv(data_path)
    return df


@task(name="Data Preprocessing")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data.

    :param df: the raw data
    :return: a preprocessed DataFrame
    """
    df = object_cols_as_category(df)
    df = numerical_cols_as_float(df)
    return df


@task(name="Data Splitting")
def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET,
    random_state: int = RANDOM_STATE,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    frac_test: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train, validation, and test sets, stratified on the target column.

    :param df: initial dataframe
    :param target_col: the name fo the target column, defaults to TARGET
    :param frac_train: the fraction of data in the train set, defaults to 0.6
    :param frac_val: the fraction of data in the validation set, defaults to 0.2
    :param frac_test: the fraction of data in the test set, defaults to 0.2
    :param random_state: the random seed, defaults to RANDOM_STATE
    :return: a tuple of DataFrames (train, validation, test)
    """
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError("Fractions must sum to 1")

    # Perform initial split into train/temp and validate/test
    df_train, df_temp = train_test_split(
        df, stratify=df[target_col], test_size=(1.0 - frac_train), random_state=random_state
    )

    # Further split temp into val/test
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test = train_test_split(
        df_temp, stratify=df_temp[target_col], test_size=relative_frac_test, random_state=random_state
    )

    return df_train, df_val, df_test


@task(name="Store data in S3")
def store_data_in_s3(df: pd.DataFrame, bucket_name: str, file_key: str) -> None:
    """
    Write a Pandas DataFrame to an S3 bucket as a CSV file.

    :param df: the DataFrame to be written
    :param bucket_name: the name of the S3 bucket
    :param file_key: the full path to the csv file
    """
    # Create an S3 resource
    s3 = boto3.resource("s3")

    # Convert the DataFrame to a CSV string
    csv_string = df.to_csv(index=False)

    # Write the CSV string to the S3 bucket
    s3.Object(bucket_name, file_key).put(Body=csv_string)


@task(name="Model Training", log_prints=True)
def train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    target: str = TARGET,
    num_features: List[str] = NUM_FEATURES,
    cat_features: List[str] = CAT_FEATURES,
    random_state: int = RANDOM_STATE,
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier on the training data and evaluate the model
    performance on the validation data.

    :param df_train: the training dataset
    :param df_val: the validation dataset
    :param target: the name of the target column, defaults to TARGET
    :param num_features: the list of numerical features, defaults to NUM_FEATUERS
    :param cat_features: the list of categorical features, defaults to CAT_FEATURES
    :param random_state: the random seed, defaults to RANDOM_STATE
    :return: a fitted CatBoost classifier
    """
    # training set
    X_train = df_train[num_features + cat_features]
    y_train = df_train[target]

    # validation set
    X_val = df_val[num_features + cat_features]
    y_val = df_val[target]

    with mlflow.start_run():
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            cat_features=cat_features,
            verbose=200,
            random_state=random_state,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        # classification metrics on train set
        y_pred_train = model.predict(X_train)
        metrics_train = classification_metrics(y_true=y_train, y_pred=y_pred_train, mode="train")

        # classification metrics on validation set
        y_pred_val = model.predict(X_val)
        metrics_val = classification_metrics(y_true=y_val, y_pred=y_pred_val, mode="val")

        # log train and validation metrics
        metrics = {**metrics_train, **metrics_val}
        mlflow.log_metrics(metrics)

        # log the model
        signature = infer_signature(X_val, y_pred_val)
        mlflow.catboost.log_model(
            model,
            "catboost-model",
            registered_model_name="catboost-model",
            await_registration_for=None,
            signature=signature,
        )

        # log the model params
        model_params = model.get_all_params()
        mlflow.log_params(model_params)

        # log the confusion matrix
        log_confusion_matrix_to_mlflow(y_true=y_val, y_pred=y_pred_val)

        # log the classification report
        log_classification_report_to_mlflow(y_true=y_val, y_pred=y_pred_val)
    return model


@flow
def training_flow() -> None:
    """
    The main training flow orchestrated using Prefect.

    The training flow performs the following steps:
        1. Sets up the MLflow tracking server and creates an experiment.
        2. Reads the pet adoption data from the local data directory.
        3. Preprocesses the data.
        4. Splits the data into training, validation, and test sets.
        5. Stores the training, validation, and test sets in S3.
        6. Trains a CatBoost classifier on the training data and evaluates the model performance on the validation set.\
            It also logs the model and evaluation metrics to MLflow.
    """
    setup_mlflow()
    df = read_data()
    df = preprocess_data(df)
    df_train, df_val, df_test = stratified_split(df)
    store_data_in_s3(df_train, S3_BUCKET, "data/df_train.csv")
    store_data_in_s3(df_val, S3_BUCKET, "data/df_val.csv")
    store_data_in_s3(df_test, S3_BUCKET, "data/df_test.csv")
    train_model(df_train, df_val)


if __name__ == "__main__":
    training_flow()
