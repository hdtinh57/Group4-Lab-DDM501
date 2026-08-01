"""
Model Training Stage for ML Pipeline.

This module handles:

"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import mlflow
from surprise import SVD, NMF, KNNBasic

from pipeline.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_CONFIGS,
    MODELS_DIR,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "svd": SVD,
    "nmf": NMF,
    "knn": KNNBasic,
}


class SurpriseModelWrapper(mlflow.pyfunc.PythonModel):
    """Expose a Surprise rating model through MLflow's standard model format."""

    def load_context(self, context) -> None:
        with open(context.artifacts["model_file"], "rb") as model_file:
            self.model = pickle.load(model_file)

    def predict(self, context, model_input):
        required_columns = {"user_id", "movie_id"}
        if not required_columns.issubset(model_input.columns):
            missing = required_columns.difference(model_input.columns)
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return [
            self.model.predict(str(user_id), str(movie_id)).est
            for user_id, movie_id in zip(model_input["user_id"], model_input["movie_id"])
        ]


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
) -> None:
    """
    Setup MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured: URI={tracking_uri}, Experiment={experiment_name}")
def train_model(
    trainset: Any,
    model_type: str = "svd",
    run_name: Optional[str] = None,
    **model_params
) -> Tuple[Any, str]:
    """
    Train a recommendation model and log to MLflow.
    
    Args:
        trainset: Surprise trainset object
        model_type: Type of model ('svd', 'nmf', 'knn')
        run_name: Optional name for the MLflow run
        **model_params: Model hyperparameters
        
    Returns:
        Tuple of (trained_model, run_id)
        
    Example:
        model, run_id = train_model(
            trainset, 
            model_type='svd',
            n_factors=100,
            n_epochs=20
        )
    """
    
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {', '.join(sorted(MODEL_CLASSES))}"
        )

    params = model_params or get_default_params(model_type)
    logged_params = {"model_type": model_type, **params}

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(logged_params)
        model = MODEL_CLASSES[model_type](**params)
        model.fit(trainset)

        run_id = active_run.info.run_id
        model_path = MODELS_DIR / f"{model_type}_{run_id}.pkl"
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        mlflow.log_artifact(str(model_path), artifact_path="raw_model")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SurpriseModelWrapper(),
            artifacts={"model_file": str(model_path)},
        )

    logger.info("Training complete. Run ID: %s", run_id)
    return model, run_id
def train_with_config(trainset: Any, config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Train model using a configuration dictionary.
    
    Args:
        trainset: Surprise trainset object
        config: Configuration dictionary with model_type and hyperparameters
        
    Returns:
        Tuple of (trained_model, run_id)
        
    Example:
        config = {"model_type": "svd", "n_factors": 100, "n_epochs": 20}
        model, run_id = train_with_config(trainset, config)
    """
    
    config_copy = config.copy()
    model_type = config_copy.pop("model_type", "svd")
    return train_model(trainset, model_type=model_type, **config_copy)
def get_model_class(model_type: str):
    """
    Get the model class for a given model type.
    
    Args:
        model_type: Type of model ('svd', 'nmf', 'knn')
        
    Returns:
        Model class from Surprise library
        
    Raises:
        ValueError: If model_type is not supported
    """
    
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {', '.join(sorted(MODEL_CLASSES))}"
        )
    return MODEL_CLASSES[model_type]
def get_default_params(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary of default parameters
    """
    return MODEL_CONFIGS.get(model_type, {})


def list_available_models() -> list:
    """
    List all available model types.
    
    Returns:
        List of model type names
    """
    return list(MODEL_CLASSES.keys())
if __name__ == "__main__":
    from pipeline.data_ingestion import load_and_split
    
    print("Testing Training Module")
    print("=" * 50)
    setup_mlflow()
    trainset, testset, _ = load_and_split()
    
    print("\nAvailable models:", list_available_models())
    print("Default SVD params:", get_default_params("svd"))
