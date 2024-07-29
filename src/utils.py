from pathlib import Path

import mlflow
import pandas as pd

from src.evaluate import plot_classification_report, plot_confusion_matrix


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


def get_artifact_path() -> Path:
    """Returns the path to the artifact folder."""
    project_root = get_project_root()
    return project_root / "artifacts"


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
