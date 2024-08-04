GRANT ALL PRIVILEGES ON DATABASE monitoring TO admin;

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