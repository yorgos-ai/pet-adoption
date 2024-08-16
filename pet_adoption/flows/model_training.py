import os

from dotenv import load_dotenv
from prefect import flow

from pet_adoption.flows.tasks import (
    preprocess_data,
    read_data,
    save_monitoring_metrics_in_s3,
    setup_mlflow,
    store_data_in_s3,
    store_json_in_s3,
    stratified_split,
    train_metrics_to_db,
    train_model,
    training_monitoring,
)
from pet_adoption.utils import date_today_str

load_dotenv()


@flow
def training_flow() -> None:
    """
    The main training flow orchestrated with Prefect.

    The training flow performs the following steps:
    - Sets up the MLflow tracking server and creates an experiment.
    -  Reads the pet adoption data from the local data directory.
    -  Preprocesses the data.
    -  Splits the data into training, validation, and test sets.
    -  Stores the training, validation, and test sets in S3.
    -  Trains a CatBoost classifier on the training data and evaluates the model performance on the validation set.\
        It also logs the model and evaluation metrics to MLflow.
    """
    batch_date = date_today_str()
    setup_mlflow()

    df = read_data()
    df = preprocess_data(df=df)
    df_train, df_val, df_test = stratified_split(df=df)

    # store the data in S3
    store_data_in_s3(df_train, os.getenv("S3_BUCKET"), "data/df_train.csv")
    store_data_in_s3(df_val, os.getenv("S3_BUCKET"), "data/df_val.csv")
    store_data_in_s3(df_test, os.getenv("S3_BUCKET"), "data/df_test.csv")

    # train the model
    _, train_df, val_df, run_id = train_model(df_train=df_train, df_val=df_val)

    # save the MLflow run ID of the production model in S3
    run_id_dict = {"run_id": run_id}
    store_json_in_s3(dict_obj=run_id_dict, bucket_name=os.getenv("S3_BUCKET_MLFLOW"), file_key="production_model.json")

    # monitoring metrics
    metrics_dict = training_monitoring(reference_data=train_df, current_data=val_df)
    save_monitoring_metrics_in_s3(
        metrics_dict=metrics_dict, bucket_name=os.getenv("S3_BUCKET"), file_key="data/monitoring_metrics_training.json"
    )
    train_metrics_to_db(batch_date=batch_date, metrics_dict=metrics_dict, db_name="training_monitoring")


if __name__ == "__main__":
    training_flow()
