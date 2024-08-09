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

The Pet Adoption Dataset provides a comprehensive look into various factors that can influence the likelihood of a pet being adopted from a shelter. This dataset includes detailed information about pets available for adoption, covering various characteristics and attributes.

The data set consists of 2008 records. The complete csv file is located under the `data` directory of the repository. Due to the limited number of data records available, the data pre-processing is adjusted to accommodate for that. During the training flow, the dataset is splitted in three separate sets (train, validation and test). All the subsets as well as the complete initial dataset are stored in S3. The training pipeline uses the train and validation sets to train a CatBoost classifier and evaluate the model performance on the validation set. The test set is used in the prediction pipeline to simulate hourly batch predictions.

## Pre-requisites

### 1. Create an IAM user
Create AWS user with administrator access. Note the `AWS_SECRET_ACCESS_KEY` and `AWS_ACCESS_KEY_ID`.

https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html

### 2. Create an EC2 instance
Create an EC2 instance. A remote machine with 8 GB of RAM should be more than enough for this project.

Since I am using WSL in Windows 10 with Ubuntu 22.04 as my local machine, I chose to create an Ubuntu 22.04 EC2 instance, to make sure that both my local and remote machines are using the same OS.

#### Install aws-cli

```
$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
$ sudo apt install unzip
$ unzip awscliv2.zip
$ sudo ./aws/install
```

#### Configure AWS credentials
Set up your AWS profile by adding your `AWS_SECRET_ACCESS_KEY` and `AWS_ACCESS_KEY_ID` form step 1 to your AWS profile. These credentials will be automatically used to authenticate your IAM role and be able access AWS services from your EC2 instance.

#### Install Docker

To install Doker on Ubuntu 22.04, run the following commands:

```
$ sudo apt update
$ sudo apt install docker.io
$ sudo groupadd docker
$ sudo usermod -aG docker ${USER}
```

#### Install docker-compose
```
$ wget https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64
```

#### Install Pyenv
Pyenv allows us to install and handle multiple Python versions on a single machine. To install Pyenv on Ubuntu 22.04, you can follow the instructions in this excellent [Medium guide](https://medium.com/@aashari/easy-to-follow-guide-of-how-to-install-pyenv-on-ubuntu-a3730af8d7f0).

```
$ sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libpq-dev
$ curl https://pyenv.run | bash
$ echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
```

#### Install Poetry

This project uses Poetry to manage Python dependecies.
```
$ curl -sSL https://install.python-poetry.org | python3 -
```
Once the installation is finished, Poetry will prompt you to add its bin directory to your PATH in order to be able to use poetry from your CLI. You can do that by running:
```
$ echo 'export PATH="/home/{user}/.local/bin:$PATH"' >> ~/.bashrc
```
Replace `{user}` with your user name.

#### Configure Poetry with Pyenv
This project uses Python 3.10.12. You can install this specific version with Pyenv and then use Poetry to install the dependencies.
```
$ pyenv install 3.10.12
$ pyenv local 3.10.12
$ pyenv shell
$ poetry env use 3.10.12
$ poetry shell
$ poetry install
$ pre-commit install
```

### 3. Create S3 buckets
This project uses two S3 buckets. The `pet-adoption-mlops` is a general purpose S3 bucket for storing artifacts and the second, `mlflow-artifacts-pet-adoption` is the artifact folder for the MLflow tracking server.

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
