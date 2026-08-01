"""Behavioral tests for the MLflow-backed pipeline stages.

The tests isolate external MLflow infrastructure while checking the contracts
required by the lab: parameters/artifacts are logged, metrics are calculated,
the best run is registered, and experiment reports are reproducible.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _run_context(run_id: str = "run-123"):
    """Return an MLflow-like context manager with a deterministic run ID."""
    context = MagicMock()
    context.__enter__.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id=run_id)
    )
    return context


def test_train_model_logs_parameters_and_model_artifact(monkeypatch, tmp_path):
    """Training must make the run reproducible and store a model artifact."""
    from pipeline import training

    class FakeModel:
        def __init__(self, **params):
            self.params = params
            self.trainset = None

        def fit(self, trainset):
            self.trainset = trainset
            return self

    mlflow = MagicMock()
    mlflow.start_run.return_value = _run_context()
    monkeypatch.setattr(training, "mlflow", mlflow)
    monkeypatch.setattr(training, "MODEL_CLASSES", {"svd": FakeModel})
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(training.pickle, "dump", lambda *_args, **_kwargs: None)

    model, run_id = training.train_model(
        trainset="training-data", model_type="svd", n_factors=12, n_epochs=3
    )

    assert run_id == "run-123"
    assert model.params == {"n_factors": 12, "n_epochs": 3}
    assert model.trainset == "training-data"
    mlflow.log_params.assert_called_once_with(
        {"model_type": "svd", "n_factors": 12, "n_epochs": 3}
    )
    mlflow.log_artifact.assert_called_once()
    assert mlflow.log_artifact.call_args.kwargs["artifact_path"] == "raw_model"
    mlflow.pyfunc.log_model.assert_called_once()
    assert mlflow.pyfunc.log_model.call_args.kwargs["artifact_path"] == "model"


def test_train_model_rejects_unknown_model_type():
    """An unsupported model type must fail before opening an MLflow run."""
    from pipeline.training import train_model

    with pytest.raises(ValueError, match="Unsupported model type"):
        train_model("training-data", model_type="unknown")


def test_evaluate_model_logs_core_and_additional_metrics(monkeypatch):
    """Evaluation must calculate rating metrics and attach them to its run."""
    from pipeline import evaluation

    predictions = [
        SimpleNamespace(r_ui=4.0, est=3.0, details={"was_impossible": False}),
        SimpleNamespace(r_ui=2.0, est=2.5, details={"was_impossible": True}),
    ]
    model = MagicMock()
    model.test.return_value = predictions
    mlflow = MagicMock()
    mlflow.start_run.return_value = _run_context("eval-run")
    monkeypatch.setattr(evaluation, "mlflow", mlflow)
    monkeypatch.setattr(evaluation.accuracy, "rmse", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr(evaluation.accuracy, "mae", lambda *_args, **_kwargs: 0.6)
    monkeypatch.setattr(evaluation, "create_prediction_distribution_plot", lambda _p: MagicMock())
    monkeypatch.setattr(evaluation, "create_error_by_rating_plot", lambda _p: MagicMock())

    metrics = evaluation.evaluate_model(model, testset=[("u", "i", 4)], run_id="eval-run")

    assert metrics["rmse"] == 0.8
    assert metrics["mae"] == 0.6
    assert metrics["mse"] == pytest.approx(0.625)
    assert metrics["coverage"] == 0.5
    mlflow.start_run.assert_called_once_with(run_id="eval-run")
    mlflow.log_metrics.assert_called_once_with(metrics)
    assert mlflow.log_figure.call_count == 2


def test_registry_selects_registers_and_promotes_best_run(monkeypatch):
    """The registry path must choose the lowest RMSE and promote that version."""
    from pipeline import registry

    fake_run = SimpleNamespace(
        info=SimpleNamespace(run_id="best-run", artifact_uri="file:///artifacts"),
        data=SimpleNamespace(metrics={"rmse": 0.73}, params={"n_factors": "100"}),
    )
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="7")
    client.search_runs.return_value = [fake_run]
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)

    best = registry.find_best_run("movie-rating-prediction")

    assert best["run_id"] == "best-run"
    assert client.search_runs.call_args.kwargs["order_by"] == ["metrics.rmse ASC"]

    mlflow = MagicMock()
    mlflow.register_model.return_value = SimpleNamespace(version="4")
    monkeypatch.setattr(registry, "mlflow", mlflow)
    assert registry.register_model("best-run", "movie-rating-model") == "4"
    mlflow.register_model.assert_called_once_with(
        "runs:/best-run/model", "movie-rating-model"
    )

    result = registry.register_best_model("movie-rating-prediction")
    assert result["model_name"] == "movie-rating-model"
    assert result["version"] == "4"
    assert result["stage"] == "Production"
    client.transition_model_version_stage.assert_called_once_with(
        name="movie-rating-model",
        version="4",
        stage="Production",
        archive_existing_versions=True,
    )


def test_experiments_use_one_dataset_and_generate_report(monkeypatch, tmp_path):
    """Multiple configurations must share one split and yield a useful report."""
    from experiments import run_experiments

    trainset, testset = object(), object()
    monkeypatch.setattr(run_experiments, "setup_mlflow", lambda **_kwargs: None)
    monkeypatch.setattr(run_experiments, "load_and_split", lambda: (trainset, testset, {}))
    monkeypatch.setattr(
        run_experiments,
        "train_model",
        lambda _trainset, model_type, **_params: (model_type, f"{model_type}-run"),
    )
    monkeypatch.setattr(
        run_experiments,
        "evaluate_model",
        lambda _model, _testset, _run_id: {"rmse": 0.8, "mae": 0.6},
    )

    configs = [
        {"model_type": "svd", "n_factors": 50},
        {"model_type": "nmf", "n_factors": 40},
    ]
    results = run_experiments.run_all_experiments(configs, "lab-test")
    report = run_experiments.generate_experiment_report(
        results, str(tmp_path / "experiment_report.md")
    )

    assert len(results) == 2
    assert "Total experiments: 2" in report
    assert "Best Model" in report
    assert (tmp_path / "experiment_report.md").is_file()
