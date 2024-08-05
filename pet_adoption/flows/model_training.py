import os

from dotenv import load_dotenv
from prefect import flow

from pet_adoption.flows.tasks import (
    extract_report_data,
    monitor_model_performance,
    preprocess_data,
    read_data,
    save_dict_in_s3,
    setup_mlflow,
    store_data_in_s3,
    store_json_in_s3,
    stratified_split,
    train_model,
)

load_dotenv()


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

    # store the data in S3
    store_data_in_s3(df_train, os.getenv("S3_BUCKET"), "data/df_train.csv")
    store_data_in_s3(df_val, os.getenv("S3_BUCKET"), "data/df_val.csv")
    store_data_in_s3(df_test, os.getenv("S3_BUCKET"), "data/df_test.csv")

    # train the model
    _, train_df, val_df, run_id = train_model(df_train, df_val)

    # save the MLflow run ID of the production model in S3
    run_id_dict = {"run_id": run_id}
    store_json_in_s3(dict_obj=run_id_dict, bucket_name=os.getenv("S3_BUCKET_MLFLOW"), file_key="production_model.json")

    # monitoring metrics
    metrics_dict = monitor_model_performance(reference_data=train_df, current_data=val_df)
    save_dict_in_s3(metrics_dict, os.getenv("S3_BUCKET"), "data/monitoring_metrics_training.json")
    extract_report_data(batch_date="2024-08-03", metrics_dict=metrics_dict, db_name="training_monitoring")


if __name__ == "__main__":
    training_flow()
