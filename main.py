import mlflow
from mlflow.projects import docker

def run_step(entrypoint, parameters=None):
    print("----------\nLAUNCHING STEP: entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def workflow():
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run():
        submitted_load_data = run_step('load_raw_data')
        artifact_location = submitted_load_data.info.artifact_uri

        submitted_preprocess = run_step('preprocess_data', parameters={'data_location': artifact_location})
        processed_data_location = submitted_preprocess.info.artifact_uri
        
        run_step('train_classifier', parameters={'data_location': processed_data_location})
        print('Workflow finished.')
        

if __name__ == "__main__":
    workflow()
