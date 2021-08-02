import mlflow

def run_step(entrypoint, parameters=None):
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return #mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

    # the return value is used in https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/main.py
    # but I guess it is not needed here?

def workflow():
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run():
        #add pipeline run here...
        run_step('load_raw_data')
        run_step('preprocess_data')
        run_step('train_classifier')
        print('Workflow finished.')
        


if __name__ == "__main__":
    workflow()
