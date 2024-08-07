import json
from datetime import date
from pathlib import Path

import boto3
import mlflow
import pandas as pd

from pet_adoption.evaluate import plot_classification_report, plot_confusion_matrix


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


def get_artifact_path() -> Path:
    """Returns the path to the artifact folder."""
    project_root = get_project_root()
    return project_root / "artifacts"


def date_today_str(date_format: str = "%Y-%m-%d") -> str:
    """
    Returns the current date as a string.

    :param date_format: the date format, defaults to "%Y-%m-%d"
    :return: today's date as a string
    """
    return date.today().strftime(date_format)


def log_confusion_matrix_to_mlflow(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Log the confusion matrix to MLflow.

    :param y_true: the actual labels
    :param y_pred: the model predictions
    """
    artifact_path = get_artifact_path()
    cfn_matrix_file_name = "confusion_matrix.png"
    cfn_matrix_path = Path.joinpath(artifact_path, cfn_matrix_file_name)
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, save_path=cfn_matrix_path)
    mlflow.log_artifact(cfn_matrix_path)


def log_classification_report_to_mlflow(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Log the classification report to MLflow.

    :param y_true: the actual labels
    :param y_pred: the model predictions
    """
    artifact_path = get_artifact_path()
    cls_report_name = "classification_report.json"
    cls_report_path = Path.joinpath(artifact_path, cls_report_name)
    plot_classification_report(y_true=y_true, y_pred=y_pred, save_path=cls_report_path)
    mlflow.log_artifact(cls_report_path)


def save_dict_in_s3(data: dict, bucket: str, file_path: str) -> None:
    """
    Save a dictionary as a JSON file in an S3 bucket.

    :param data: the dictionary to save
    :param bucket: the S3 bucket name
    :param file_path: the full path to save the data
    """
    # Create an S3 resource
    s3 = boto3.resource("s3")
    # Convert the dictionary to a JSON string
    json_str = json.dumps(data)
    # Write the JSON string to the S3 bucket
    s3.Object(bucket, file_path).put(Body=json_str)
