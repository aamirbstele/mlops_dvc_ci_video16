import json
from mlflow.tracking import MlflowClient
import mlflow
import os
##import dagshub
##dagshub.init(repo_owner='aamirbstele', repo_name='mlops_dvc_ci_video16', mlflow=True)

# Set the experiment name in MLflow

##mlflow.set_experiment("Final_Model")

# Set the tracking URI for MLflow to log the experiment in DagsHub
##mlflow.set_tracking_uri("https://dagshub.com/aamirbstele/mlops_dvc_ci_video16.mlflow")

##dagshub_token = os.getenv("DAGSHUB_TOKEN")
##if not dagshub_token:
##    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

##os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
##os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow_url = "http://mlflow:5000"
repo_owner = "aamirbstele"
repo_name = "mlops_dvc_ci_video16"
mlflow.set_tracking_uri(f"{mlflow_url}")
mlflow.set_experiment("Final_Model")



# Load the run ID and model name from the saved JSON file
reports_path = "reports/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id'] # Fetch run id from the JSON file
model_name = run_info['model_name']  # Fetch model name from the JSON file

# Create an MLflow client
client = MlflowClient()

# Create the model URI
model_uri = f"runs:/{run_id}/artifacts/{model_name}"

# Register the model
reg = mlflow.register_model(model_uri, model_name)

# Get the model version
model_version = reg.version  # Get the registered model version

# Transition the model version to Staging
new_stage = "Staging"

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")