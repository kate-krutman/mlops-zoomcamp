from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    name="scheduled-deployment-2",
    flow_location="/Users/katush/Homeworks/mlops-zoomcamp/03-orchestration/homework.py",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["hw_deploy"]
)
