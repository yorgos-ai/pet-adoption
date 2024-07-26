# Pet Adoption Classifier
This repository contains an implementation of an end to end pet adoption classifier.

It is the final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course from [DataTalks.Club](https://datatalks.club/).

## Author
- Name: Yorgos Papageorgiou
- LinkedIn profile: [Yorgos Papageorgiou](https://www.linkedin.com/in/yorgos-papageorgiou-137312107/)

## About the project

The primary aim of this project is to develop an end-to-end machine learning system to predict the likelihood of a pet being adopted from a shelter. A considerable amount of attention is directed towards the implementation of MLOps practices, rather than the development of the machine learning model.

## About the data

Data source: [Predict Pet Adoption Status Dataset (Kaggle)](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset/data)

The Pet Adoption Dataset provides a comprehensive look into various factors that can influence the likelihood of a pet being adopted from a shelter. This dataset includes detailed information about pets available for adoption, covering various characteristics and attributes.

## Pre-requisites

### 1. Create an IAM user
Create AWS user with administrator access. Note the AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID.

https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html

### 2. Create an EC2 instance
Create an EC2 instance. A remote machine with 8 GB of RAM should be more than enough for this project.

Since I am using WSL in Windows 10 with Ubuntu 22.04 as my local machine, I chose to create an Ubuntu 22.04 EC2 instance, to make sure that both my local and remote machines are operating with the same OS.

#### Install aws-cli

```
$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
$ sudo apt install unzip
$ unzip awscliv2.zip
$ sudo ./aws/install
```

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
$
```


### 3. MLflow on AWS EC2 instance.
This project uses an EC2 instance to set up a remote tracking server for MLflow. To set up a remote tracking server, follow the instructions in this [guide](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md).

Make sure that you create an AWS Linux machine for the MLflow server.
