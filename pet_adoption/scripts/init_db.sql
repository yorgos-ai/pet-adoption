CREATE DATABASE training_monitoring;
GRANT ALL PRIVILEGES ON DATABASE training_monitoring TO admin;

\c training_monitoring admin

CREATE TABLE IF NOT EXISTS drift_target (
    batch_date timestamp NOT NULL,
    drift_stat_test character varying(255),
    drift_stat_threshold decimal,
    drift_score decimal,
    drift_detected boolean
);

CREATE TABLE IF NOT EXISTS drift_prediction (
    batch_date timestamp NOT NULL,
    drift_stat_test character varying(255),
    drift_stat_threshold decimal,
    drift_score decimal,
    drift_detected boolean
);

CREATE TABLE IF NOT EXISTS drift_dataset (
    batch_date timestamp NOT NULL,
    drift_share decimal,
    number_of_columns smallint,
    number_of_drifted_columns smallint,
    share_of_drifted_columns decimal,
    dataset_drift boolean
);

CREATE TABLE IF NOT EXISTS classification_metrics (
    batch_date timestamp NOT NULL,
    train_accuracy decimal,
    train_precision decimal,
    train_recall decimal,
    train_f1 decimal,
    val_accuracy decimal,
    val_precision decimal,
    val_recall decimal,
    val_f1 decimal
);


CREATE DATABASE predict_monitoring;
GRANT ALL PRIVILEGES ON DATABASE predict_monitoring TO admin;

\c predict_monitoring admin

CREATE TABLE IF NOT EXISTS drift_prediction (
    batch_date timestamp NOT NULL,
    drift_stat_test character varying(255),
    drift_stat_threshold decimal,
    drift_score decimal,
    drift_detected boolean
);

CREATE TABLE IF NOT EXISTS drift_dataset (
    batch_date timestamp NOT NULL,
    drift_share decimal,
    number_of_columns smallint,
    number_of_drifted_columns smallint,
    share_of_drifted_columns decimal,
    dataset_drift boolean
);

CREATE DATABASE prefect;
GRANT ALL PRIVILEGES ON DATABASE prefect TO admin;

-- CREATE DATABASE mlflow;
-- GRANT ALL PRIVILEGES ON DATABASE mlflow TO admin;
