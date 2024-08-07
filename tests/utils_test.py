from pathlib import Path

from pet_adoption.utils import (
    date_today_str,
    get_artifact_path,
    get_project_root,
)


def test_get_project_root():
    project_root = get_project_root()
    assert project_root == Path("/home/ubuntu/pet-adoption")


def test_get_artifact_path():
    artifact_path = get_artifact_path()
    assert artifact_path == Path("/home/ubuntu/pet-adoption/artifacts")


def test_date_today_str():
    today_str = date_today_str()
    assert isinstance(today_str, str)


# def test_log_confusion_matrix_to_mlflow():
#     # Assuming y_true and y_pred are pandas Series
#     y_true = pd.Series([0, 1, 0, 1])
#     y_pred = pd.Series([0, 0, 1, 1])
#     log_confusion_matrix_to_mlflow(y_true, y_pred)
#     # Add assertions to check if the confusion matrix is logged to MLflow

# def test_log_classification_report_to_mlflow():
#     # Assuming y_true and y_pred are pandas Series
#     y_true = pd.Series([0, 1, 0, 1])
#     y_pred = pd.Series([0, 0, 1, 1])
#     log_classification_report_to_mlflow(y_true, y_pred)
#     # Add assertions to check if the classification report is logged to MLflow
