"""
Model Evaluation Stage for ML Pipeline.

This module handles:

"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import matplotlib.pyplot as plt
from surprise import accuracy

from pipeline.config import ARTIFACTS_DIR
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def evaluate_model(
    model: Any,
    testset: List,
    run_id: str,
    log_to_mlflow: bool = True
) -> Dict[str, float]:
    """
    Evaluate model and log metrics to MLflow.
    
    Args:
        model: Trained Surprise model
        testset: Test set as list of (user, item, rating) tuples
        run_id: MLflow run ID to log metrics to
        log_to_mlflow: Whether to log metrics to MLflow
        
    Returns:
        Dictionary with evaluation metrics {'rmse': float, 'mae': float}
        
    Example:
        metrics = evaluate_model(model, testset, run_id)
        print(f"RMSE: {metrics['rmse']:.4f}")
    """
    
    predictions = model.test(testset)
    rmse = float(accuracy.rmse(predictions, verbose=False))
    mae = float(accuracy.mae(predictions, verbose=False))
    metrics = {"rmse": rmse, "mae": mae, **calculate_additional_metrics(predictions)}

    if log_to_mlflow:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            figures = [
                (create_prediction_distribution_plot(predictions), "prediction_distribution.png"),
                (create_error_by_rating_plot(predictions), "error_by_rating.png"),
            ]
            for figure, artifact_name in figures:
                mlflow.log_figure(figure, artifact_name)
                plt.close(figure)

    logger.info("Evaluation complete. RMSE=%.4f, MAE=%.4f", rmse, mae)
    return metrics
def calculate_additional_metrics(predictions: List) -> Dict[str, float]:
    """
    Calculate additional evaluation metrics beyond RMSE and MAE.
    
    Args:
        predictions: List of Surprise Prediction objects
        
    Returns:
        Dictionary with additional metrics
    """
    
    if not predictions:
        return {"mse": 0.0, "mape": 0.0, "coverage": 0.0, "n_predictions": 0}

    actuals = np.asarray([prediction.r_ui for prediction in predictions], dtype=float)
    estimates = np.asarray([prediction.est for prediction in predictions], dtype=float)
    non_zero_actuals = actuals != 0
    mape = (
        float(np.mean(np.abs((actuals[non_zero_actuals] - estimates[non_zero_actuals]) /
                             actuals[non_zero_actuals])) * 100)
        if np.any(non_zero_actuals)
        else 0.0
    )
    predictable = sum(
        not prediction.details.get("was_impossible", False)
        for prediction in predictions
    )
    return {
        "mse": float(np.mean((actuals - estimates) ** 2)),
        "mape": mape,
        "coverage": predictable / len(predictions),
        "n_predictions": len(predictions),
    }
def create_prediction_distribution_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing prediction vs actual rating distribution.
    
    Args:
        predictions: List of Surprise Prediction objects
        
    Returns:
        Matplotlib figure
    """
    actuals = [pred.r_ui for pred in predictions]
    estimated = [pred.est for pred in predictions]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(actuals, estimated, alpha=0.1, s=1)
    axes[0].plot([1, 5], [1, 5], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual Rating')
    axes[0].set_ylabel('Predicted Rating')
    axes[0].set_title('Actual vs Predicted Ratings')
    axes[0].legend()
    axes[1].hist(actuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Actual Ratings')
    errors = np.array(estimated) - np.array(actuals)
    axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Prediction Errors')
    
    plt.tight_layout()
    return fig


def create_error_by_rating_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing error distribution by actual rating.
    
    Args:
        predictions: List of Surprise Prediction objects
        
    Returns:
        Matplotlib figure
    """
    rating_groups = {}
    for pred in predictions:
        rating = round(pred.r_ui)
        if rating not in rating_groups:
            rating_groups[rating] = []
        rating_groups[rating].append(pred.est - pred.r_ui)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ratings = sorted(rating_groups.keys())
    positions = range(len(ratings))
    
    bp = ax.boxplot(
        [rating_groups[r] for r in ratings],
        positions=positions,
        widths=0.6
    )
    
    ax.set_xticklabels([str(r) for r in ratings])
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error by Actual Rating')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    return fig


def save_evaluation_report(metrics: Dict, filepath: str) -> None:
    """
    Save evaluation metrics to a text file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the report
    """
    with open(filepath, 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{name}: {value}\n")
    
    logger.info(f"Evaluation report saved to {filepath}")
if __name__ == "__main__":
    print("Testing Evaluation Module")
    print("=" * 50)
    
    print("Evaluation module loaded successfully.")
