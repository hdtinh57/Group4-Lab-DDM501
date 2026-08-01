"""
Model Registry Stage for ML Pipeline.

This module handles:

"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import MLFLOW_EXPERIMENT_NAME
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    ascending: bool = True
) -> Dict[str, Any]:
    """
    Find the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to optimize (default: 'rmse')
        ascending: If True, lower is better (default: True for RMSE)
        
    Returns:
        Dictionary with best run information:
        {
            'run_id': str,
            'metrics': dict,
            'params': dict,
            'artifact_uri': str
        }
        
    Example:
        best = find_best_run(metric='rmse', ascending=True)
        print(f"Best RMSE: {best['metrics']['rmse']}")
    """
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "metrics": dict(best_run.data.metrics),
        "params": dict(best_run.data.params),
        "artifact_uri": best_run.info.artifact_uri,
    }
def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model"
) -> str:
    """
    Register a model from an MLflow run to the Model Registry.
    
    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        artifact_path: Path to the model artifact within the run
        
    Returns:
        Version number of the registered model (as string)
        
    Example:
        version = register_model(run_id, "movie-rating-model")
        print(f"Registered model version: {version}")
    """
    
    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Registering model from %s as '%s'", model_uri, model_name)
    result = mlflow.register_model(model_uri, model_name)
    return str(result.version)
def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Production"
) -> None:
    """
    Transition a model version to a new stage.
    
    Args:
        model_name: Name of the registered model
        version: Version number to transition
        stage: Target stage ('Staging', 'Production', 'Archived')
        
    Valid stages:
        
    Example:
        transition_model_stage("movie-rating-model", "1", "Production")
    """
    
    valid_stages = {"None", "Staging", "Production", "Archived"}
    if stage not in valid_stages:
        raise ValueError(f"Unsupported model stage: {stage}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("Model %s v%s moved to %s", model_name, version, stage)
def register_best_model(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    model_name: str = "movie-rating-model",
    metric: str = "rmse",
    stage: str = "Production"
) -> Dict[str, Any]:
    """
    Find the best model and register it to the Model Registry.
    
    Args:
        experiment_name: Name of the MLflow experiment
        model_name: Name for the registered model
        metric: Metric to optimize (default: 'rmse')
        stage: Stage to transition to (default: 'Production')
        
    Returns:
        Dictionary with registration info:
        {
            'run_id': str,
            'model_name': str,
            'version': str,
            'stage': str,
            'metrics': dict
        }
        
    Example:
        result = register_best_model()
        print(f"Registered {result['model_name']} v{result['version']}")
    """
    
    best_run = find_best_run(experiment_name, metric, ascending=True)
    version = register_model(best_run["run_id"], model_name)
    transition_model_stage(model_name, version, stage)
    return {
        "run_id": best_run["run_id"],
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "metrics": best_run["metrics"],
    }
def list_registered_models() -> List[Dict[str, Any]]:
    """
    List all registered models.
    
    Returns:
        List of model information dictionaries
    """
    client = MlflowClient()
    models = client.search_registered_models()
    
    return [
        {
            "name": model.name,
            "latest_versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }
                for v in model.latest_versions
            ]
        }
        for model in models
    ]


def get_production_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the current production version of a model.
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        Dictionary with model info or None if not found
    """
    client = MlflowClient()
    
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            v = versions[0]
            return {
                "name": model_name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
            }
    except Exception as e:
        logger.error(f"Error getting production model: {e}")
    
    return None


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top N runs from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        top_n: Number of runs to return
        
    Returns:
        List of run information sorted by metric
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        return []
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_n
    )
    
    return [
        {
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        for run in runs
    ]
if __name__ == "__main__":
    print("Testing Registry Module")
    print("=" * 50)
    print("\nRegistered models:", list_registered_models())
    
    print("\nRegistry module loaded successfully.")
