import pandas as pd
from sklearn.model_selection import train_test_split

from app.services.dataset import dataset_service

NUMERICAL_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]

TARGET = "churn"


class ChurnPreprocessingService:
    """Подготавливает churn-датасет к обучению модели."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

        self._X_train: pd.DataFrame | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._y_test: pd.Series | None = None

    def _separate_features_and_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Отделяет матрицу признаков X от целевой переменной y."""
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET]
        return X, y

    def prepare(self) -> None:
        """
        Основной метод:
        1. Берёт df из dataset_service
        2. Отделяет X от y
        3. Делает train/test split
        4. Кеширует результат
        """
        df = dataset_service.df
        X, y = self._separate_features_and_target(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

    @property
    def X_train(self) -> pd.DataFrame:
        if self._X_train is None:
            self.prepare()
        return self._X_train

    @property
    def X_test(self) -> pd.DataFrame:
        if self._X_test is None:
            self.prepare()
        return self._X_test

    @property
    def y_train(self) -> pd.Series:
        if self._y_train is None:
            self.prepare()
        return self._y_train

    @property
    def y_test(self) -> pd.Series:
        if self._y_test is None:
            self.prepare()
        return self._y_test

    def split_info(self) -> dict:
        """Возвращает размеры train/test и распределение churn в каждой части."""
        train_dist = self.y_train.value_counts().to_dict()
        test_dist = self.y_test.value_counts().to_dict()

        return {
            "train_size": len(self.y_train),
            "test_size": len(self.y_test),
            "train_churn_distribution": {
                "churn_0": train_dist.get(0, 0),
                "churn_1": train_dist.get(1, 0),
            },
            "test_churn_distribution": {
                "churn_0": test_dist.get(0, 0),
                "churn_1": test_dist.get(1, 0),
            },
        }


preprocessing_service = ChurnPreprocessingService()
