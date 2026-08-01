"""
Experiment Runner - Run multiple experiments for hyperparameter tuning.

This script runs multiple experiments with different configurations
and logs all results to MLflow for comparison.


Usage:
    python -m experiments.run_experiments
"""

import logging
from typing import Dict, Any, List
import json
from datetime import datetime
from pathlib import Path

import mlflow

from pipeline.config import EXPERIMENT_CONFIGS, MLFLOW_EXPERIMENT_NAME
from pipeline.data_ingestion import load_and_split
from pipeline.training import train_model, setup_mlflow
from pipeline.evaluation import evaluate_model
from pipeline.registry import compare_runs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def run_single_experiment(
    trainset: Any,
    testset: Any,
    config: Dict[str, Any],
    experiment_name: str = "hyperparameter-tuning"
) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    Args:
        trainset: Training data
        testset: Test data
        config: Configuration dictionary with model_type and hyperparameters
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Dictionary with experiment results:
        {
            'config': dict,
            'run_id': str,
            'metrics': dict
        }
    """
    
    setup_mlflow(experiment_name=experiment_name)
    config_copy = config.copy()
    model_type = config_copy.pop("model_type")
    model, run_id = train_model(
        trainset,
        model_type=model_type,
        run_name=f"{experiment_name}-{model_type}",
        **config_copy,
    )
    metrics = evaluate_model(model, testset, run_id)
    return {"config": config.copy(), "run_id": run_id, "metrics": metrics}
def run_all_experiments(
    configs: List[Dict[str, Any]] = EXPERIMENT_CONFIGS,
    experiment_name: str = "hyperparameter-tuning"
) -> List[Dict[str, Any]]:
    """
    Run all experiments defined in configs.
    
    Args:
        configs: List of configuration dictionaries
        experiment_name: Name of the MLflow experiment
        
    Returns:
        List of experiment results
    """
    
    trainset, testset, _ = load_and_split()
    results = []
    for config in configs:
        try:
            results.append(
                run_single_experiment(trainset, testset, config, experiment_name)
            )
        except Exception as error:
            logger.exception("Experiment failed for configuration %s", config)
            results.append({"config": config.copy(), "error": str(error)})
    return results
def generate_experiment_report(
    results: List[Dict[str, Any]],
    output_path: str = "experiment_report.md"
) -> str:
    """
    Generate a markdown report from experiment results.
    
    Args:
        results: List of experiment results
        output_path: Path to save the report
        
    Returns:
        Report content as string
    """
    
    successful = [result for result in results if "metrics" in result]
    ranked = sorted(successful, key=lambda result: result["metrics"]["rmse"])
    lines = [
        "# Experiment Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- Total experiments: {len(results)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(results) - len(successful)}",
        "",
        "## Results",
        "| Rank | Model | Parameters | RMSE | MAE | Run ID |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for rank, result in enumerate(ranked, start=1):
        config = result["config"]
        params = {key: value for key, value in config.items() if key != "model_type"}
        metrics = result["metrics"]
        lines.append(
            f"| {rank} | {config['model_type']} | `{json.dumps(params, sort_keys=True)}` "
            f"| {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {result['run_id']} |"
        )

    lines.extend(["", "## Best Model"])
    if ranked:
        best = ranked[0]
        lines.extend([
            f"- Configuration: `{json.dumps(best['config'], sort_keys=True)}`",
            f"- RMSE: {best['metrics']['rmse']:.4f}",
            f"- MAE: {best['metrics']['mae']:.4f}",
            "- Recommendation: register this run after validating its serving behavior.",
        ])
    else:
        lines.append("No successful experiments were available for recommendation.")

    report = "\n".join(lines) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report
def main():
    """Run all experiments and generate report."""
    
    logger.info("=" * 60)
    logger.info("Starting Experiment Runner")
    logger.info("=" * 60)
    setup_mlflow()
    results = run_all_experiments(
        configs=EXPERIMENT_CONFIGS,
        experiment_name="hyperparameter-tuning"
    )
    report = generate_experiment_report(results, "docs/experiment_report.md")
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Summary")
    logger.info("=" * 60)
    
    successful = [r for r in results if 'metrics' in r]
    if successful:
        best = min(successful, key=lambda x: x['metrics'].get('rmse', float('inf')))
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Best RMSE: {best['metrics']['rmse']:.4f}")
        logger.info(f"Best config: {best['config']}")
    logger.info("\nTop 5 runs:")
    top_runs = compare_runs(metric="rmse", top_n=5)
    for i, run in enumerate(top_runs, 1):
        logger.info(f"  {i}. RMSE={run['metrics'].get('rmse', 'N/A'):.4f} - {run['params']}")
    
    logger.info("\nReport saved to: docs/experiment_report.md")
    logger.info("View experiments in MLflow UI: http://localhost:5000")
    
    return results


if __name__ == "__main__":
    main()
