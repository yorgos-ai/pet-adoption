import json
import os
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, Tuple

import boto3
import mlflow
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from catboost import CatBoostClassifier, Pool
from evidently import ColumnMapping
from evidently.metrics import (
    ClassificationQualityMetric,
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from prefect import task
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from pet_adoption.evaluate import classification_metrics
from pet_adoption.feature_engineering import numerical_cols_as_float
from pet_adoption.utils import (
    get_project_root,
    log_classification_report_to_mlflow,
    log_confusion_matrix_to_mlflow,
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


@task(name="MLflow Setup")
def setup_mlflow() -> None:
    """
    Set up the MLflow tracking server and create an experiment.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT"))
    if experiment:
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT"))
    else:
        # Create a new experiment with the given name and S3 path as the artifact folder
        mlflow.create_experiment(
            os.getenv("MLFLOW_EXPERIMENT"), artifact_location=f"s3://{os.getenv('S3_BUCKET_MLFLOW')}"
        )
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT"))


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


@task(name="Data Preprocessing")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data.

    :param df: the raw data
    :return: a preprocessed DataFrame
    """
    df = numerical_cols_as_float(df)
    return df


@task(name="Data Splitting")
def stratified_split(
    df: pd.DataFrame,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    frac_test: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train, validation, and test sets, stratified on the target column.

    :param df: initial dataframe
    :param frac_train: the fraction of data in the train set, defaults to 0.6
    :param frac_val: the fraction of data in the validation set, defaults to 0.2
    :param frac_test: the fraction of data in the test set, defaults to 0.2
    :return: a tuple of DataFrames (train, validation, test)
    """
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError("Fractions must sum to 1")

    # Perform initial split into train/temp and validate/test
    df_train, df_temp = train_test_split(
        df, stratify=df[TARGET], test_size=(1.0 - frac_train), random_state=RANDOM_STATE
    )

    # Further split temp into val/test
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test = train_test_split(
        df_temp, stratify=df_temp[TARGET], test_size=relative_frac_test, random_state=RANDOM_STATE
    )

    return df_train, df_val, df_test


@task(name="Store Data in S3")
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
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> Tuple[CatBoostClassifier, pd.DataFrame, pd.DataFrame, str]:
    """
    Train a CatBoost classifier on the training data and evaluate the model
    performance on the validation data.

    :param df_train: the training dataset
    :param df_val: the validation dataset
    :return: the fitted model, the train and validation sets including the model prediction and the MLflow run ID
    """
    # training set
    X_train = df_train[NUM_FEATURES + CAT_FEATURES]
    y_train = df_train[TARGET]

    # validation set
    X_val = df_val[NUM_FEATURES + CAT_FEATURES]
    y_val = df_val[TARGET]

    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run_id: {run_id}")

        train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
        val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            verbose=200,
            random_state=RANDOM_STATE,
        )

        model.fit(train_pool, eval_set=val_pool, plot=True, use_best_model=True)

        # train metrics
        y_pred_train = model.predict(X_train)
        df_train["prediction"] = y_pred_train
        metrics_train = classification_metrics(y_true=y_train, y_pred=y_pred_train, mode="train")

        # validation metrics
        y_pred_val = model.predict(X_val)
        df_val["prediction"] = y_pred_val
        metrics_val = classification_metrics(y_true=y_val, y_pred=y_pred_val, mode="val")

        # log train and validation metrics
        metrics = {**metrics_train, **metrics_val}
        mlflow.log_metrics(metrics)

        # log train and validation sets
        train_data = mlflow.data.from_pandas(df=df_train, name="train_data")
        mlflow.log_input(train_data, "training")
        val_data = mlflow.data.from_pandas(df=df_val, name="val_data")
        mlflow.log_input(val_data, "validation")

        # log the confusion matrix
        log_confusion_matrix_to_mlflow(y_true=y_val, y_pred=y_pred_val)

        # log the classification report
        log_classification_report_to_mlflow(y_true=y_val, y_pred=y_pred_val)

        # log the model
        mlflow.catboost.log_model(
            model,
            os.getenv("MLFLOW_MODEL_NAME"),
            await_registration_for=None,
            signature=None,  # MLflow data types and CatBoost categorical features do not work well together
        )

        # log the model params
        model_params = model.get_all_params()
        mlflow.log_params(model_params)

        # register the model if the validation recall is above 0.9
        print(f"The challenger model has a validation recall of {metrics_val['val_recall']}.")
        if metrics_val["val_recall"] > 0.9:
            mlflow.register_model(f"runs:/{run_id}/catboost-model", "catboost-model")
            print(f"Model registered in MLflow with run_id: {run_id}")

            # promote to production
            client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))
            client.transition_model_version_stage(
                name=os.getenv("MLFLOW_MODEL_NAME"),
                version=client.get_latest_versions(os.getenv("MLFLOW_MODEL_NAME"), stages=["None"])[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            print("Model promoted to Production.")
        else:
            print("The challenger model did not meet the recall threshold of 0.9 on the validation set.")

    return model, df_train, df_val, run_id


@task(name="Store MLflow Run ID")
def store_json_in_s3(dict_obj: Dict, bucket_name: str, file_key: str) -> None:
    """
    Store a dictionary as a JSON file in an S3 bucket.

    :param dict_obj: the dictionary to store
    :param bucket_name: the name of the S3 bucket
    :param file_key: the full path to the file
    """
    # Create an S3 resource
    s3 = boto3.resource("s3")
    # Convert the dictionary to a JSON string
    json_string = json.dumps(dict_obj)
    # Write the JSON string to the S3 bucket
    s3.Object(bucket_name, file_key).put(Body=json_string)


@task(name="Training Monitoring Metrics")
def training_monitoring(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    """
    Create the training flow monitoring report using Evidently.

    :param reference_data: the reference data
    :param val_data: the current data
    :return: a dictionary containing the monitoring metrics
    """
    col_mapping = ColumnMapping(
        target=TARGET, prediction="prediction", numerical_features=NUM_FEATURES, categorical_features=CAT_FEATURES
    )

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name=TARGET),
            ClassificationQualityMetric(),
        ]
    )

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=col_mapping)
    metrics_dict = report.as_dict()["metrics"]
    return metrics_dict


@task(name="Prediction Monitoring Metrics")
def prediction_monitoring(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    """
    Create monitoring report using Evidently.

    :param reference_data: the reference data
    :param val_data: the current data
    :return: a dictionary containing the monitoring metrics
    """
    col_mapping = ColumnMapping(
        target=None, prediction="prediction", numerical_features=NUM_FEATURES, categorical_features=CAT_FEATURES
    )

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=col_mapping)
    metrics_dict = report.as_dict()["metrics"]
    return metrics_dict


@task(name="Store Monitoring Metrics")
def save_monitoring_metrics_in_s3(metrics_dict: dict, bucket_name: str, file_key: str) -> None:
    """
    Store a dictionary as a JSON file in an S3 bucket.

    :param metrics_dict: the dictionary to be stored
    :param bucket_name: the name of the S3 bucket
    :param file_key: the full path to the JSON file
    """
    save_dict_in_s3(data=metrics_dict, bucket=bucket_name, file_path=file_key)


@task(name="Send Training Metrics to Postgres")
def train_metrics_to_db(batch_date, metrics_dict: dict, db_name: str) -> None:
    """
    Extract the training monitoring metrics and store them in a PostgreSQL database.

    :param batch_date: the date of the batch
    :param metrics_dict: the dictionary containing the monitoring metrics
    :param db_name: the name of the Postgres db
    """
    drift_target = {
        "batch_date": batch_date,
        "drift_stat_test": metrics_dict[3]["result"]["stattest_name"],
        "drift_stat_threshold": metrics_dict[3]["result"]["stattest_threshold"],
        "drift_score": metrics_dict[3]["result"]["drift_score"],
        "drift_detected": metrics_dict[3]["result"]["drift_detected"],
    }

    drift_prediction = {
        "batch_date": batch_date,
        "drift_stat_test": metrics_dict[0]["result"]["stattest_name"],
        "drift_stat_threshold": metrics_dict[0]["result"]["stattest_threshold"],
        "drift_score": metrics_dict[0]["result"]["drift_score"],
        "drift_detected": metrics_dict[0]["result"]["drift_detected"],
    }

    drift_dataset = {
        "batch_date": batch_date,
        "drift_share": metrics_dict[1]["result"]["drift_share"],
        "number_of_columns": metrics_dict[1]["result"]["number_of_columns"],
        "number_of_drifted_columns": metrics_dict[1]["result"]["number_of_drifted_columns"],
        "share_of_drifted_columns": metrics_dict[1]["result"]["share_of_drifted_columns"],
        "dataset_drift": metrics_dict[1]["result"]["dataset_drift"],
    }

    classification_metrics = {
        "batch_date": batch_date,
        "train_accuracy": metrics_dict[4]["result"]["current"]["accuracy"],
        "train_precision": metrics_dict[4]["result"]["current"]["precision"],
        "train_recall": metrics_dict[4]["result"]["current"]["recall"],
        "train_f1": metrics_dict[4]["result"]["current"]["f1"],
        "val_accuracy": metrics_dict[4]["result"]["reference"]["accuracy"],
        "val_precision": metrics_dict[4]["result"]["reference"]["precision"],
        "val_recall": metrics_dict[4]["result"]["reference"]["recall"],
        "val_f1": metrics_dict[4]["result"]["reference"]["f1"],
    }

    params = {
        "user": os.getenv("POSTGRES_USER"),
        "pass": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "database": db_name,
    }
    engine = create_engine("postgresql://%(user)s:%(pass)s@%(host)s:%(port)s/%(database)s" % params)

    # insert metrics
    pd.DataFrame(drift_target, index=[0]).to_sql("drift_target", engine, if_exists="replace", index=False)
    pd.DataFrame(drift_prediction, index=[0]).to_sql("drift_prediction", engine, if_exists="replace", index=False)
    pd.DataFrame(drift_dataset, index=[0]).to_sql("drift_dataset", engine, if_exists="replace", index=False)
    pd.DataFrame(classification_metrics, index=[0]).to_sql(
        "classification_metrics", engine, if_exists="replace", index=False
    )


@task(name="Send Prediciton Metrics to Postgres")
def prediction_metrics_to_db(batch_date, metrics_dict: dict, db_name: str) -> None:
    """
    Extract the batch prediction monitoring metrics and store them in a PostgreSQL database.

    :param batch_date: the date of the batch
    :param metrics_dict: the dictionary containing the monitoring metrics
    :param db_name: the name of the Postgres db
    """
    drift_prediction = {
        "batch_date": batch_date,
        "drift_stat_test": metrics_dict[0]["result"]["stattest_name"],
        "drift_stat_threshold": metrics_dict[0]["result"]["stattest_threshold"],
        "drift_score": metrics_dict[0]["result"]["drift_score"],
        "drift_detected": metrics_dict[0]["result"]["drift_detected"],
    }

    drift_dataset = {
        "batch_date": batch_date,
        "drift_share": metrics_dict[1]["result"]["drift_share"],
        "number_of_columns": metrics_dict[1]["result"]["number_of_columns"],
        "number_of_drifted_columns": metrics_dict[1]["result"]["number_of_drifted_columns"],
        "share_of_drifted_columns": metrics_dict[1]["result"]["share_of_drifted_columns"],
        "dataset_drift": metrics_dict[1]["result"]["dataset_drift"],
    }

    params = {
        "user": os.getenv("POSTGRES_USER"),
        "pass": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "database": db_name,
    }
    engine = create_engine("postgresql://%(user)s:%(pass)s@%(host)s:%(port)s/%(database)s" % params)

    # insert metrics
    pd.DataFrame(drift_prediction, index=[0]).to_sql("drift_prediction", engine, if_exists="append", index=False)
    pd.DataFrame(drift_dataset, index=[0]).to_sql("drift_dataset", engine, if_exists="append", index=False)


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


@task(name="Apply Model")
def apply_model(df: pd.DataFrame, model: PyFuncModel) -> pd.DataFrame:
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


@task(name="Get Production Model Run ID from S3")
def get_production_model_run_id(bucket_name: str, file_key: str = "production_model.json") -> str:
    """
    Read the latest production model run ID from the JSON file in S3.

    :param bucket_name: the name of the S3 bucket
    :param file_key: the full path to the JSON file, defaults to "production_model.json"
    :return: the latest production model run ID
    """
    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        # Read the JSON file from the S3 bucket
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        json_string = response["Body"].read().decode("utf-8")

        # Parse the JSON string
        data = json.loads(json_string)

        # Get the latest production model run ID
        latest_run_id = data["run_id"]

        return latest_run_id
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"The file '{file_key}' does not exist in the bucket '{bucket_name}'.")
            return None
        else:
            raise e


@task(name="Make Batch Predictions")
def simulate_batch_predictions(
    df_train: pd.DataFrame, df_test: pd.DataFrame, date_time: datetime, model: PyFuncModel
) -> None:
    """
    Simulate hourly batch predictions by splitting the test set in 12 chunks.

    :param df_train: the training data
    :param df_test: the test data
    :param date_time: the date used to create batches
    :param model: the loaded model
    """
    # simulate hourly batch predictions by splitting the test set in 12 chunks
    batch_size = 12
    batch_dfs = np.array_split(df_test, batch_size)

    for idx, batch_df in enumerate(batch_dfs):
        batch_date = date_time + timedelta(hours=idx + 1)
        batch_date_str = batch_date.strftime("%Y-%m-%d %H:%M:%S")
        batch_df["batch_date"] = batch_date

        # apply the model to the batch
        batch_df = apply_model(df=batch_df, model=model)

        # store the batch dataframe in S3
        store_data_in_s3(
            df=batch_df,
            bucket_name=os.getenv("S3_BUCKET"),
            file_key=f"predictions/df_test-{batch_date_str.replace('-', '_')}.csv",
        )

        # create monitoring metrics for the batch
        metrics_dict = prediction_monitoring(reference_data=df_train, current_data=batch_df)
        save_dict_in_s3(
            data=metrics_dict, bucket=os.getenv("S3_BUCKET"), file_path=f"prediction_metrics/{batch_date_str}.json"
        )

        # extract the monitoring metrics and store them in a PostgreSQL database
        prediction_metrics_to_db(batch_date=batch_date_str, metrics_dict=metrics_dict, db_name="predict_monitoring")
