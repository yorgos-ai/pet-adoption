# Pet Adoption Classifier
This repository contains an implementation of an end to end pet adoption classifier.

It is the final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course from [DataTalks.Club](https://datatalks.club/).

## Author
- Name: Yorgos Papageorgiou
- [LinkedIn profile](https://www.linkedin.com/in/yorgos-papageorgiou-137312107/)

## About the project

The primary aim of this project is to develop an end-to-end machine learning system to predict the likelihood of a pet being adopted from a shelter.

> [!NOTE]
> The focus of this project is towards the implementation of MLOps practices rather than development of the ML model.

## About the data

Data source: [Predict Pet Adoption Status Dataset (Kaggle)](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset/data)

The Pet Adoption Dataset provides a comprehensive look into various factors that can influence the likelihood of a pet being adopted from a shelter. This dataset includes detailed information about pets available for adoption, covering various characteristics and attributes. The data set consists of 2008 records. The complete csv file is located under the [data](data) directory of the repository.

## Project solution architecture

There are two main flows in this project. The training flow uses the training and validation data to train a CatBoost classifier that learns to predict the probability of a pet being adopted. The batch prediction flow retrieves the test data and uses the trained model to provide predictions.

### Training flow
![plot](images/training_flow.png)

This section provides an overview of the training flow. First, it reads the csv file from the [data](data) directory of the repository. The preprocessing step casts all numerical columns as float type. Afterwards, the data is split in three subsets (train, validation and test) using statified splitting, to ensure that relative class frequencies is approximately preserved in each subset. All three subsets are stored in S3. A CatBoost classifier with default hyperparameters is trained on the training set. The validation set is used to evaluate the performance of the model and select the optimal number of trees. Experiment tracking is implemented using MLflow with a SQLite database as the backend. All model hyperparameters along with training and validation performance metrics are logged in Mlflow. Furthermore, the classification report and confusion matrix images from the validation set are tracked by MLflow. The MLflow Model Registry is used to promote the model to production stage if the recall metric on validation set exceeds the 0.9 threshold. The MLflow RUN ID of the promoted model is stored in S3. Model monitoring is implemented with Evidently AI. The monitoring report is stored in S3 and a selection of monitoring metrics are stored in a PostgreSQL database. Finally, Grafana ingests the monitoring metrics to visualize the monitoring dashboard of the training flow.

### Batch prediction flow
![plot](images/prediction_flow.png)

This section provides an overview of the batch prediction flow. The test subset is retrieved from S3 and the data get preprocessed. The MLflow RUN ID of the production model is retrieved from S3, which is used to load the trained model from MLflow. The test subset is split in 12 chunks and a timestamp incremented by 1 hour is appended to each chuck in order to simulate a batch prediction scenario. The model is applied to each chunk to predict the adoption probability. Evidently calculates a monitoring report for each chunk which is stored in S3. A selection of monitoring metrics is stored in the PostgreSQL database. Finally, Grafana loads the batch prediction metrics from the Postgres database to populate the batch prediction monitoring plots.

> [!NOTE]
> Due to the limited number of data records available (approx. 2000), the test set is preserved to simulate hourly batch predictions.

## Pre-requisites

Before you build the project to start up the services required to execute the flows, there are some necessary steps that you need to perform. These pre-requisites are described in detail in the [PRE-REQUISITES](./PRE-REQUISITES.md) file.

## Build project

> [!IMPORTANT]
> Make sure to follow the steps described in the previous section, before you execute any of the commands below

To start up the application stack run the following command:
```
make build
```
This command initiates all the necessary applications that are used in this project. You can find the complete list of all services as well as their url in the table below.

|   service   | port  | Interface  |          url           |       description       |
|------------ |------ |----------- |----------------------- |------------------------ |
| Prefect     | 4200  | 127.0.0.1  | http://127.0.0.1:4200  | Prefect UI              |
| MLflow      | 5000  | 127.0.0.1  | http://127.0.0.1:5000  | MLflow UI               |
| Grafana     | 3000  | 127.0.0.1  | http://127.0.0.1:3000  | Grafana UI              |
| Adminer     | 8080  | 127.0.0.1  | http://127.0.0.1:8080  | Adminer UI              |


> [!IMPORTANT]
> You might have to manually configure the port-forwarding from VScode in your machine in order to be able to access the URL links above.


## Workflow orchestration with Prefect
This project uses Prefect to orchestration and deploy the workflows.

There are two main flows in this project, the `training_flow` and the `batch_prediction_flow`. The Python scripts for both flows can be accessed at [pet_adoption/flows/](pet_adoption/flows/) directory.

Once you have build the project services you can run the training and prediction flows in a couple of ways:

#### 1. Execute the flows from CLI

To run both training and prediction flows, type the following command in your CLI:
```
make run_flows
```
This make command will first execute the training flow and then the prediction flow. You can access the Prefect UI to see the execution of both flows.


#### 2. Deploy the Prefect flows and manually run them from Prefect UI

To deploy the Prefect flows, run the following command:
```
make prefect_deploy
```
This command will deploy the workflows, create a Prefect worker and finally start the worker. Once this step finishes, you can access Prefect UI and run the flows manually.

## Experiment tracking with MLflow
MLflow Tracking Server is used to track the experiments and log training artifacts and metrics. Once the training pipeline executes, you can access MLflow UI to view the experiment run and all the relevant metrics and artifacts.

## Monitoring with Evidently

Both flows are monitored using Evidently. The metrics are collected during the flow runs and are stored in a Postgres DB. Grafana connects to tehe Postgres DB to get the metrics and build monirting dashboards. To view the dashboard, go to http://127.0.0.1:3000 and navigate to the `Dahsboards` section on the left panel.

## Unit tests
Run the unit tests and generate the coverage report:
```
make tests
```

## Linting and formatting
This project uses Ruff for both linting and formatting. Details about the configuration can be found in the [pyproject.toml](pyproject.toml) file.

## Pre-commit hooks
Pre-commit hooks are used for code linting/formatting and other checks before every commit. You can find the complete configuration in the [.pre-commit-config.yaml](.pre-commit-config.yaml) file.

## Backlog for future improvements
1. Add integration tests (Prefect flows).
2. IaC with Terraform to manage the infratructure.
3. AWS EventBridge and SQS to trigger the prediction pipeline automatically when a new test set is uploaded in S3.
