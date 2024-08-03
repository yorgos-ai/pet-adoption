import json
import os
from typing import List, Tuple

import boto3
import mlflow
import pandas as pd
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.report import Report
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from pet_adoption.evaluate import classification_metrics
from pet_adoption.feature_engineering import numerical_cols_as_float
from pet_adoption.utils import get_project_root, log_classification_report_to_mlflow, log_confusion_matrix_to_mlflow

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


@task(name="MLflow Setup")
def setup_mlflow() -> None:
    """
    Set up the MLflow tracking server and create an experiment.
    """
    # mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
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
) -> Tuple[CatBoostClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Train a CatBoost classifier on the training data and evaluate the model
    performance on the validation data.

    :param df_train: the training dataset
    :param df_val: the validation dataset
    :param target: the name of the target column, defaults to TARGET
    :param num_features: the list of numerical features, defaults to NUM_FEATUERS
    :param cat_features: the list of categorical features, defaults to CAT_FEATURES
    :param random_state: the random seed, defaults to RANDOM_STATE
    :return: the fitted CatBoost classifier and the train and validation sets including the model prediction
    """
    # training set
    X_train = df_train[num_features + cat_features]
    y_train = df_train[target]

    # validation set
    X_val = df_val[num_features + cat_features]
    y_val = df_val[target]

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
            random_state=random_state,
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

    return model, df_train, df_val


@task(name="Monitoring metrics")
def monitor_model_performance(train_data: pd.DataFrame, val_data: pd.DataFrame) -> dict:
    """
    Create monitoring report using Evidently.

    :param train_data: the training set
    :param val_data: the validation set
    :return: a dictionary containing the monitoring metrics
    """
    col_mapping = ColumnMapping(
        target=None, prediction="prediction", numerical_features=NUM_FEATURES, categorical_features=CAT_FEATURES
    )

    report = Report(
        metrics=[
            # ClassificationPRCurve(
            #     col_mapping=col_mapping,
            #     prediction_column="prediction",
            #     target_column=TARGET,
            #     pos_label=1
            # ),
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    report.run(reference_data=train_data, current_data=val_data, column_mapping=col_mapping)
    metric_sdict = report.as_dict()["metrics"]
    return metric_sdict


@task(name="Store Monitoring Metrics")
def save_dict_in_s3(metrics_dict: dict, bucket_name: str, file_key: str) -> None:
    """
    Store a dictionary as a JSON file in an S3 bucket.

    :param metrics_dict: the dictionary to be stored
    :param bucket_name: the name of the S3 bucket
    :param file_key: the full path to the JSON file
    """
    # Create an S3 resource
    s3 = boto3.resource("s3")
    # Convert the dictionary to a JSON string
    json_string = json.dumps(metrics_dict)
    # Write the JSON string to the S3 bucket
    s3.Object(bucket_name, file_key).put(Body=json_string)


@task(name="extract_batch_report_data")
def extract_report_data(batch_date, metrics_dict: dict) -> None:
    # report.run(reference_data=training_data, current_data=batch_data, column_mapping=column_mapping)
    # drift_report = report.as_dict()["metrics"]

    drift_prediction = {
        "batch_date": batch_date,
        "drift_stat_test": metrics_dict[0]["result"]["stattest_name"],
        "drift_stat_threshold": metrics_dict[0]["result"]["stattest_threshold"],
        "drift_score": metrics_dict[0]["result"]["drift_score"],
        "drift_detected": metrics_dict[0]["result"]["drift_detected"],
    }

    drift_dataset = {
        "batch_date": batch_date,
        "drift_dataset": metrics_dict[1]["result"]["drift_share"],
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
        "database": "metrics_training",
    }
    engine = create_engine("postgresql://%(user)s:%(pass)s@%(host)s:%(port)s/%(database)s" % params)

    # insert metrics
    pd.DataFrame(drift_prediction, index=[0]).to_sql("drift_prediction", engine, if_exists="append", index=False)
    pd.DataFrame(drift_dataset, index=[0]).to_sql("drift_dataset", engine, if_exists="append", index=False)


@flow
def training_flow() -> None:
    """
    The main training flow orchestrated with Prefect.

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
    store_data_in_s3(df_train, os.getenv("S3_BUCKET"), "data/df_train.csv")
    store_data_in_s3(df_val, os.getenv("S3_BUCKET"), "data/df_val.csv")
    store_data_in_s3(df_test, os.getenv("S3_BUCKET"), "data/df_test.csv")
    model, train_df, val_df = train_model(df_train, df_val)
    metrics_dict = monitor_model_performance(train_data=train_df, val_data=val_df)
    save_dict_in_s3(metrics_dict, os.getenv("S3_BUCKET"), "data/monitoring_metrics.json")
    extract_report_data(batch_date="2024-08-03", metrics_dict=metrics_dict)
    print(metrics_dict)


if __name__ == "__main__":
    training_flow()
