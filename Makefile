.PHONY: tests mlflow start_services run_flows e2e_flow build
include .env

# Complete setup

init_setup:
	chmod +x pet_adoption/scripts/set_up.sh
	bash pet_adoption/scripts/set_up.sh

pyenv_setup:
	cd pet-adoption
	pyenv install 3.10.12
	pyenv local 3.10.12

poetry_setup:
	pyenv shell 3.10.12
	poetry env use 3.10.12

env_setup:
	poetry shell
	poetry install
	pre-commit install

set_up: init_setup pyenv_setup poetry_setup env_setup

# Applications

mlflow:
	poetry shell
	mlflow server --backend-store-uri '${MLFLOW_TRACKING_URI}' --default-artifact-root 's3://${S3_BUCKET_MLFLOW}'

start_services:
	poetry shell
	dotenv
	docker-compose up --build -d

build: start_services mlflow

kill_services:
	docker-compose down --remove-orphans --volumes

# Workflows orchestration

run_flows:
	poetry shell
	python pet_adoption/flows/model_training.py
	python pet_adoption/flows/batch_prediction.py

prefect_deploy:
	poetry shell
	python pet_adoption/flows/prefect_deploy.py
	prefect work-pool create --type process main_pool
	prefect worker start --pool main_pool

e2e_flow: build run_flows

# Tests

tests:
	poetry shell
	pytest -v --cov-report term --cov=pet_adoption tests/
