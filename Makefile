.PHONY: tests mllfow start_services run_flows e2e_flow build


set_up:
	chmod +x pet_adoption/scripts/set_up.sh
	bash pet_adoption/scripts/set_up.sh

mlflow:
	poetry shell
	mlflow server --backend-store-uri 'sqlite:///mlflow.db' --default-artifact-root 's3://mlflow-artifacts-pet-adoption'

tests:
	poetry shell
	pytest -v --cov-report term --cov=pet_adoption tests/

start_services:
	poetry shell
	dotenv
	docker-compose up --build -d

build: start_services mlflow

kill_services:
	docker-compose down --remove-orphans --volumes

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
