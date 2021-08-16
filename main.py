import mlflow
import os

def run_step(entrypoint, parameters=None):
    print("----------\nLAUNCHING STEP: entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def workflow():
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run():
        mlflow.set_tag('step-name', 'pipeline run')

        submitted_load_data = run_step('load_raw_data')
        artifact_location = submitted_load_data.info.artifact_uri

        submitted_preprocess = run_step('preprocess_data', parameters={'data_location': artifact_location})
        processed_data_location = submitted_preprocess.info.artifact_uri
        
        run_step('train_classifier', parameters={'data_location': processed_data_location})
        print('Workflow finished.')
        
        print('Clearing temporary files from memory...')
        # here: delete features and target txt's from local and artifacts
        # I currently have the tmp files duplicated? should fix...
        os.remove('features.txt')
        os.remove('target.txt')
        # BUG: python does not find the files in artifacts even though they exist
        #os.remove(processed_data_location + '/features.txt')
        #os.remove(processed_data_location + '/target.txt')
        print('Clearing finished.')
        

if __name__ == "__main__":
    workflow()
