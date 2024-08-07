from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments import Deployment

from pet_adoption.flows.batch_prediction import batch_prediction_flow
from pet_adoption.flows.model_training import training_flow

deployment_train = Deployment.build_from_flow(
    flow=training_flow,
    name="model_training",
    schedule=CronSchedule(cron="0 0 * * 1"),
    work_queue_name="main",
)

deployment_predict = Deployment.build_from_flow(
    flow=batch_prediction_flow,
    name="batch_prediction",
    schedule=CronSchedule(cron="0 0 * * 4"),
    work_queue_name="main",
)

if __name__ == "__main__":
    deployment_train.apply()
    deployment_predict.apply()
