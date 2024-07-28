import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/ashishshukla888/mlops-mini-proj.mlflow')
dagshub.init(repo_owner='ashishshukla888', repo_name='mlops-mini-proj', mlflow=True)


with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
