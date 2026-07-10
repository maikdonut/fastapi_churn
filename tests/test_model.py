import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from app.services.model import ChurnModelService
from app.schemas.churn import TrainingConfigChurn


@pytest.fixture
def model_service():
    """Свежий экземпляр ChurnModelService для каждого теста."""
    return ChurnModelService()


@pytest.fixture
def trained_model_service(sample_df):
    """ChurnModelService с обученной моделью на синтетических данных."""
    from app.services.dataset import dataset_service
    from app.services.preprocessing import ChurnPreprocessingService

    dataset_service._df = sample_df

    preprocessing = ChurnPreprocessingService()
    service = ChurnModelService()
    parameters = TrainingConfigChurn(
        model_type="logreg",
        hyperparameters={"class_weight": "balanced"},
    )

    # подменяем preprocessing_service внутри model
    import app.services.model as model_module
    original = model_module.preprocessing_service
    model_module.preprocessing_service = preprocessing

    service.train(parameters)

    model_module.preprocessing_service = original
    return service


def test_build_pipeline_structure(model_service):
    """Pipeline содержит шаги preprocessor и model."""
    from sklearn.linear_model import LogisticRegression
    pipeline = model_service._build_pipeline(LogisticRegression())

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_build_sklearn_model_logreg(model_service):
    """_build_sklearn_model возвращает LogisticRegression для logreg."""
    model_service._model_type = "logreg"
    model_service._hyperparameters = {}
    model = model_service._build_sklearn_model()
    assert isinstance(model, LogisticRegression)


def test_build_sklearn_model_random_forest(model_service):
    """_build_sklearn_model возвращает RandomForestClassifier для random_forest."""
    model_service._model_type = "random_forest"
    model_service._hyperparameters = {"n_estimators": 10}
    model = model_service._build_sklearn_model()
    assert isinstance(model, RandomForestClassifier)


def test_build_sklearn_model_unknown(model_service):
    """_build_sklearn_model кидает ValueError для неизвестного типа."""
    model_service._model_type = "unknown_model"
    model_service._hyperparameters = {}
    with pytest.raises(ValueError):
        model_service._build_sklearn_model()


def test_is_trained_false_by_default(model_service):
    """Новый экземпляр не считается обученным."""
    assert model_service.is_trained() is False


def test_is_trained_true_after_train(trained_model_service):
    """После обучения is_trained возвращает True."""
    assert trained_model_service.is_trained() is True


def test_pipeline_raises_if_not_trained(model_service):
    """Обращение к pipeline до обучения кидает RuntimeError."""
    with pytest.raises(RuntimeError):
        _ = model_service.pipeline


def test_train_returns_metrics(trained_model_service):
    """train() возвращает словарь с нужными метриками."""
    metrics = trained_model_service._metrics

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "train_size" in metrics
    assert "test_size" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_predict_returns_correct_structure(trained_model_service, sample_df):
    """predict() возвращает список словарей с нужными полями."""
    from app.services.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

    X = sample_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].head(5)
    results = trained_model_service.predict(X)

    assert len(results) == 5
    for result in results:
        assert "churn" in result
        assert "churn_probability" in result
        assert "churn_label" in result
        assert result["churn"] in [0, 1]
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["churn_label"] in ["churn", "stay"]


def test_status_not_trained(model_service):
    """status() возвращает is_trained=False если модель не обучена."""
    status = model_service.status()
    assert status["is_trained"] is False
    assert status["trained_at"] is None
    assert status["metrics"] is None


def test_status_trained(trained_model_service):
    """status() возвращает is_trained=True после обучения."""
    status = trained_model_service.status()
    assert status["is_trained"] is True
    assert status["trained_at"] is not None
    assert status["model_type"] == "logreg"