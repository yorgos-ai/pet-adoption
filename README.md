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
To start all the required services, you can run the following command:
```
make start_services
```
This command will execute the docker-compose.yaml file, which initates all the Docker containers used in this project.

## Experiment tracking with MLflow
Execute the following command to start the MLflow server:
```
$ make mlflow
```
The MLflow server can be accessed at http://127.0.0.1:5000.

## Workflow orchestration with Prefect
This project uses Prefect for workflow orchestration.

The Prefect server can be accessed at http://127.0.0.1:4200.

To deploy the Prefect workflows, run the following command:
```
make prefect_deploy
```
This command will deploy the workflows and start the Prefect Agent.

<!-- ```
$ prefect work-pool create --type process process-pool
$ prefect worker start --pool process-pool
$ prefect deployment run 'training-flow/pet-adoption-local'
``` -->
