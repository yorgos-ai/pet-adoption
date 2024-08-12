.PHONY: tests mlflow start_services run_flows e2e_flow build
include .env

# Complete setup

init_setup:
	chmod +x pet_adoption/scripts/set_up.sh
	bash pet_adoption/scripts/set_up.sh

pyenv:
	pyenv install 3.10.12
	pyenv local 3.10.12

poetry:
	poetry env use 3.10.12
	poetry install

pre-commit:
	poetry shell
	pre-commit install

env_setup: pyenv poetry pre-commit

# Applications

mlflow:
	poetry shell
	mlflow server --backend-store-uri '${MLFLOW_TRACKING_URI}' --default-artifact-root 's3://${S3_BUCKET_MLFLOW}'

start_services:
	poetry shell
	dotenv
	sudo docker-compose up --build -d

build: start_services mlflow

kill_services:
	sudo docker-compose down --remove-orphans --volumes

# Workflows orchestration

run_flows:
	poetry run python pet_adoption/flows/model_training.py
	poetry run python pet_adoption/flows/batch_prediction.py

prefect_deploy:
	poetry shell
	python pet_adoption/flows/prefect_deploy.py
	prefect work-pool create --type process main_pool
	prefect worker start --pool main_pool

e2e_flow: build run_flows

# Tests

tests:
	poetry run pytest -v --cov-report term --cov=pet_adoption tests/
