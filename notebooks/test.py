import dagshub
dagshub.init(repo_owner='aamirbstele', repo_name='mlops_dvc_ci_video16', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)