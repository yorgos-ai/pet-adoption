# Pet Adoption Classifier
This repository contains an implementation of an end to end pet adoption classifier.

It is the final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course from [DataTalks.Club](https://datatalks.club/).

## Author
- Name: Yorgos Papageorgiou
- [LinkedIn profile](https://www.linkedin.com/in/yorgos-papageorgiou-137312107/)

## About the project

The primary aim of this project is to develop an end-to-end machine learning system to predict the likelihood of a pet being adopted from a shelter. A considerable amount of attention is directed towards the implementation of MLOps practices, rather than the development of the machine learning model.

## About the data

Data source: [Predict Pet Adoption Status Dataset (Kaggle)](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset/data)

The Pet Adoption Dataset provides a comprehensive look into various factors that can influence the likelihood of a pet being adopted from a shelter. This dataset includes detailed information about pets available for adoption, covering various characteristics and attributes. The data set consists of 2008 records. The complete csv file is located under the `data` directory of the repository.

## Pre-requisites

### 1. Create an IAM user
Create AWS user with administrator access. Note the `AWS_SECRET_ACCESS_KEY` and `AWS_ACCESS_KEY_ID`.

https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html

### 2. Create an EC2 instance
Create an EC2 instance. A remote machine with 8 GB of RAM should be more than enough for this project.

Log in in the AWS concole using the IAM role you created in the previous step. Navigate to EC2 service and click on `Instances` and then `Launch an instance`.
1. Add a name for your EC2 instance.
2. Go to the Ubuntu OS images and from the drop down menu select the `Ubuntu Server 22.04` image.
3. Select `t2.xlarge` as the instance type.
4. On the `Key pair (login)` panel click on `Create a new key pair`. Enter a name for your key-pair .pem file and then click on `Create key pair`. Save the .pem file on your local machine. YOu will need it later to connect from your local machine to the remote EC2 instance using SSH.
5. On the `Configure storage` panel, edit the amount of disk storage to 30 GiB.
6. Click on `Launch instance`. The EC2 instance will be launched and started automatically.

### 3. Configure SSH from local machine to EC2 instance
In your local mahcine, navigate to the .ssh folder.
1. Copy the .pem file you downloaded earlier in the .ssh directory.
2. Create a file named `config`.
3. Add the following settings in the config file:
```
Host <the name of the EC2 instance you created>
    HostName <the Public IPv4 address of the EC2 instance>
    User ubuntu
    IdentityFile <the path to the .pem file on your local machine>
    StrictHostKeyChecking no
```
4. Open VS code in your local machine and install the `SSH remote` extension.
5. Once the extension is installed, click on `Remote` and then `Connect to Host` and select the name of your EC2 instance.

### 4. Clone the project repository using HTTPS
```
$ git clone https://github.com/yorgos-ai/pet-adoption.git
$ cd pet-adoption/
$ code .
```

### 5. Run the init_setup.sh script
```
$ sudo apt update
$ sudo apt install make
$ make set_up
```

### 6. Configure AWS credentials
Set up your AWS profile by adding your `AWS_SECRET_ACCESS_KEY` and `AWS_ACCESS_KEY_ID` form step 1 to your AWS profile. These credentials will be automatically used to authenticate your IAM role and be able access AWS services from your EC2 instance.

```
aws configure --profile mlops-zoomcamp
```

### 7. Create S3 buckets
This project uses two S3 buckets. The `pet-adoption-mlops` is a general purpose S3 bucket for storing artifacts and the second, `mlflow-artifacts-pet-adoption` is the artifact folder for the MLflow tracking server.

## Project solution architecture

There are two main flows in this project. The training flow uses the training and validation data to train a CatBoost classifier that learns to predict the probability of a pet being adopted. The batch prediction flow retrieves the test data and uses the trained model to provide predictions.

### Training flow
![plot](images/training_flow.drawio.png)

This section provides an overview of the training flow. First, it reads the csv file from the [data](data) directory of the repository. The preprocessing step casts all numerical columns as float type. Afterwards, the data is split in three subsets (train, validation and test) using statified splitting, to ensure that relative class frequencies is approximately preserved in each subset. All three subsets are stored in S3. A CatBoost classifier with default hyperparameters is trained on the training set. The validation set is used to evaluate the performance of the model and select the optimal number of trees. Experiment tracking is implemented using MLflow with a SQLite database as the backend. All model hyperparameters along with training and validation performance metrics are logged in Mlflow. Furthermore, the classification report and confusion matrix images from the validation set are tracked by MLflow. The MLflow Model Registry is used to promote the model to production stage if the recall metric on validation set exceeds the 0.9 threshold. The MLflow RUN ID of the promoted model is stored in S3. Model monitoring is implemented with Evidently AI. The monitoring report is stored in S3 and a selection of monitoring metrics are stored in a PostgreSQL database. Finally, Grafana ingests the monitoring metrics to visualize the monitoring dashboard of the training flow.


### Batch prediction flow
![plot](images/prediction_flow.drawio.png)

This section provides an overview of the batch prediction flow. The test subset is retrieved from S3 and the data get preprocessed. The MLflow RUN ID of the production model is retrieved from S3, which is used to load the trained model from MLflow. The test subset is split in 12 chunks and a timestamp incremented by 1 hour is appended to each chuck in order to simulate a batch prediction scenario. The model is applied to each chunk to predict the adoption probability. Evidently calculates a monitoring report for each chunk which is stored in S3. A selection of monitoring metrics is stored in the PostgreSQL database. Finally, Grafana loads the batch prediction metrics from the Postgres database to populate the batch prediction monitoring plots.

> [!NOTE]
> Due to the limited number of data records available (approx. 2000), the test set is preserved to simulate hourly batch predictions.

## Build project
To start upthe application stack run the following command:
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
<!-- | PostgreSQL  | 5432  | 127.0.0.1  | http://127.0.0.1:5432  | Postgres database       | -->


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
This make command will first execute the training flow. Once the first flows finishes successfully, it will execute the prediction flow. You can access the Prefect UI to see the execution of both flows.


#### 2. Deploy the Prefect flows and manually run them from Prefect UI

To deploy the Prefect workflows, run the following command:
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
1. Integration tests
2. IaC with Terraform
3. AWS EventBridge and SQS to trigger the prediction pipeline automatically when a new test set is uploaded in S3.
