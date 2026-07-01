import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from preprocessing import preprocessing_service
from dataset import dataset_service


class ChurnModelService:
    """Обучает, хранит и применяет модель классификации churn."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """Собирает sklearn Pipeline: ColumnTransformer + LogisticRegression."""
        return Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num_preprocess", StandardScaler(), NUMERICAL_FEATURES),
                ("cat_preprocess", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ])),
            ("model", LogisticRegression(random_state=self.random_state)),
        ])

    def train(self) -> dict:
        """
        Берёт данные из preprocessing_service, обучает pipeline,
        вычисляет метрики на тестовой выборке, возвращает их.
        """
        try:
            df = dataset_service.df
        except FileNotFoundError as e:
            raise ValueError(f"Датасет не найден: {e}") from e

        if df.empty:
            raise ValueError("Датасет пустой.")

        X_train = preprocessing_service.X_train
        y_train = preprocessing_service.y_train
        X_test = preprocessing_service.X_test
        y_test = preprocessing_service.y_test

        self._pipeline = train_churn_model(X_train, y_train)

        y_pred = self._pipeline.predict(X_test)

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Принимает DataFrame с признаками одного или нескольких объектов,
        возвращает предсказание класса и вероятность churn.
        """
        y_pred = self.pipeline.predict(X)
        y_proba = self.pipeline.predict_proba(X)[:, 1]

        return {
            "churn": int(y_pred[0]),
            "churn_probability": round(float(y_proba[0]), 4),
        }

    @property
    def pipeline(self) -> Pipeline:
        """Возвращает обученный pipeline. Если не обучен — кидает понятную ошибку."""
        if self._pipeline is None:
            raise RuntimeError("Модель не обучена. Вызовите сначала POST /model/train.")
        return self._pipeline

    def is_trained(self) -> bool:
        """Возвращает True если модель уже обучена."""
        return self._pipeline is not None


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Строит и обучает Pipeline на переданных данных.
    Возвращает fitted pipeline.
    Используется внутри ChurnModelService.train().
    """
    service = ChurnModelService()
    pipeline = service._build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


model_service = ChurnModelService()