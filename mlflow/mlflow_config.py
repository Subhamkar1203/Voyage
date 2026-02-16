"""
MLflow configuration and experiment tracking utilities.
"""
import os
import mlflow


MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')
EXPERIMENT_NAMES = {
    'regression': 'flight_price_regression',
    'classification': 'flight_class_classification',
    'recommendation': 'route_recommendation'
}


def setup_mlflow(experiment_key='regression'):
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = EXPERIMENT_NAMES.get(experiment_key, experiment_key)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def log_model_metrics(model_name, metrics_dict, params_dict=None):
    """Log metrics and parameters for a model run."""
    with mlflow.start_run(run_name=model_name, nested=True):
        if params_dict:
            mlflow.log_params(params_dict)
        mlflow.log_metrics(metrics_dict)


def log_artifact(file_path, artifact_path=None):
    """Log a file as an MLflow artifact."""
    mlflow.log_artifact(file_path, artifact_path)
