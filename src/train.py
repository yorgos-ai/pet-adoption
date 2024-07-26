import pandas as pd
import mlflow
import os
from catboost import CatBoostClassifier
from typing import List, Tuple
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from src.evaluate import classification_metrics, plot_classification_report, plot_confusion_matrix
from src.utils import get_artifact_path
from pathlib import Path

os.environ["AWS_PROFILE"] = (
    "mlops-zoomcamp"  # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials
)
TRACKING_SERVER_HOST = (
    "ec2-34-254-164-188.eu-west-1.compute.amazonaws.com"  # fill in with the public DNS of the EC2 instance
)
MLFLOW_EXPERIMENT = "pet-adoption-catboost"  # fill in with the name of your MLflow experiment
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


def object_type_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast object columns as category type.

    :param df: initial dataframe
    :return: dataframe with object columns casted to category type
    """
    object_columns = df.select_dtypes(include=["object"]).columns
    df[object_columns] = df[object_columns].astype("category")
    return df


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
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

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
        metrics_val = classification_metrics(y_true=y_train, y_pred=y_pred_train, mode="val")

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
