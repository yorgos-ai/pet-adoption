.PHONY: tests mlflow start_services run_flows build
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
	poetry run mlflow server --backend-store-uri '${MLFLOW_TRACKING_URI}' --default-artifact-root 's3://${S3_BUCKET_MLFLOW}'

start_services:
	poetry run dotenv
	sudo docker-compose up --build -d
	poetry run prefect server start &
	poetry run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

build: start_services mlflow

kill_services:
	sudo docker-compose down --remove-orphans --volumes
	sudo docker system prune

# Workflows orchestration

run_flows:
	poetry run python pet_adoption/flows/model_training.py
	poetry run python pet_adoption/flows/batch_prediction.py

prefect_deploy:
	poetry run python pet_adoption/flows/prefect_deploy.py
	poetry run prefect worker start --pool main_pool

# Tests

tests:
	poetry run pytest -v --cov-report term --cov=pet_adoption tests/
