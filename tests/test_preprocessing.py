import pytest
import pandas as pd
from app.services.preprocessing import ChurnPreprocessingService, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET


def test_separate_features_and_target(sample_df):
    service = ChurnPreprocessingService()
    X, y = service._separate_features_and_target(sample_df)

    # X не содержит churn
    assert TARGET not in X.columns

    # X содержит все нужные колонки
    assert list(X.columns) == NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    # y содержит только churn
    assert y.name == TARGET

    # размеры совпадают
    assert len(X) == len(sample_df)
    assert len(y) == len(sample_df)


def test_split_sizes(sample_df):
    service = ChurnPreprocessingService()

    # подменяем df в dataset_service
    from app.services.dataset import dataset_service
    dataset_service._df = sample_df

    info = service.split_info()

    assert info["train_size"] + info["test_size"] == len(sample_df)
    assert info["train_size"] == 80  # 80% от 100
    assert info["test_size"] == 20   # 20% от 100


def test_split_stratification(sample_df):
    service = ChurnPreprocessingService()

    from app.services.dataset import dataset_service
    dataset_service._df = sample_df

    # принудительно сбрасываем кеш
    service._X_train = None
    service._X_test = None
    service._y_train = None
    service._y_test = None

    train_dist = service.y_train.value_counts(normalize=True)
    test_dist = service.y_test.value_counts(normalize=True)

    # доля churn=1 в train и test отличается не более чем на 5%
    assert abs(train_dist.get(1, 0) - test_dist.get(1, 0)) < 0.05