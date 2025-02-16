import mlflow
from mlflow.tracking import MlflowClient
import os

# Load Dagshub token from environment variables for secure access
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# Set Mlflow tracking environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "aamirbstele"
repo_name = "mlops_dvc_ci_video16"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
###mlflow.set_experiment("Final_Model")

model_name = "Best Model" # Specify your registered model name

try: 

    def promote_model_to_production():
        """Promote the latest model in Staging to Production and archive the current Production model"""
        client = MlflowClient()

        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Promoting model: {model_name}")

        # Get the latest model in the Staging stage
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            print("No model found in the 'Staging' stage.")
            return
        latest_staging_version = staging_versions[0]
        staging_version_number = latest_staging_version.version
        print(f"Found model version {staging_version_number} in 'Staging'.")

        # Get the current Production model, if any
        production_versions = client.get_latest_versions(model_name, stages=["Production"])

        if production_versions:
            current_production_version = production_versions[0]
            production_version_number = current_production_version.version

            # Transition the current Production model to Archived
            client.transition_model_version_stage(
                name=model_name,
                version=production_version_number,
                stage="Archived",
                archive_existing_versions=False,
            )
            print(f"Archived model version {production_version_number} in 'Production'.")
        else:
            print("No model currently in 'Production'.")

        # Transition the latest Staging model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version_number,
            stage="Production",
            archive_existing_versions=False,
        )
        print(f"Promoted model version {staging_version_number} to 'Production'.")
except Exception as e:
    print(f"Error: {e}")

if __name__ == "__main__":
    promote_model_to_production()

