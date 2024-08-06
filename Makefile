mlflow:
	poetry shell
	mlflow server --backend-store-uri 'sqlite:///mlflow.db' --default-artifact-root 's3://mlflow-artifacts-pet-adoption'

e2e_flow:
	poetry shell
	dotenv
	docker-compose up --build -d
	python pet_adoption/flows/model_training.py
	python pet_adoption/flows/batch_prediction.py

destroy_e2e_flow:
	docker-compose down --remove-orphans --volumes
