import logging
import time
import pandas as pd
import joblib
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.history import history_service
from app.core.paths import MODEL_PATH
from app.services.dataset import dataset_service
from app.services.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    preprocessing_service,
)

logger = logging.getLogger(__name__)


class ChurnModelService:
    """Обучает, хранит и применяет модель классификации churn."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._trained_at: datetime | None = None
        self._metrics: dict | None = None
        self._model_type: str | None = None
        self._hyperparameters: dict | None = None

    def _build_pipeline(self, model) -> Pipeline:
        return Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num_preprocess", StandardScaler(), NUMERICAL_FEATURES),
                ("cat_preprocess", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ])),
            ("model", model),
        ])

    def _build_sklearn_model(self):
        if self._model_type == "logreg":
            return LogisticRegression(**self._hyperparameters)
        elif self._model_type == "random_forest":
            return RandomForestClassifier(**self._hyperparameters)
        raise ValueError("Выбранная модель не доступна.")

    def train(self, parameters) -> dict:
        logger.info(f"Training started: model_type={parameters.model_type}, hyperparameters={parameters.hyperparameters}")

        try:
            df = dataset_service.df
        except FileNotFoundError as e:
            logger.error(f"Датасет не найден: {e}")
            raise ValueError(f"Датасет не найден: {e}") from e

        if df.empty:
            logger.error("Датасет пустой.")
            raise ValueError("Датасет пустой.")

        self._model_type = parameters.model_type
        self._hyperparameters = parameters.hyperparameters
        self._hyperparameters.setdefault("random_state", self.random_state)

        model = self._build_sklearn_model()

        X_train = preprocessing_service.X_train
        y_train = preprocessing_service.y_train
        X_test = preprocessing_service.X_test
        y_test = preprocessing_service.y_test

        start_time = time.time()
        self._pipeline = self._build_pipeline(model)
        self._pipeline.fit(X_train, y_train)
        elapsed = round(time.time() - start_time, 2)

        self._trained_at = datetime.now()
        y_pred = self._pipeline.predict(X_test)

        self._metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_pred)), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        logger.info(
            f"Обучение завершено за {elapsed}с: "
            f"accuracy={self._metrics['accuracy']}, "
            f"f1={self._metrics['f1']}, "
            f"roc_auc={self._metrics['roc_auc']}"
        )

        history_service.add({
            "timestamp": self._trained_at.isoformat(),
            "model_type": self._model_type,
            "hyperparameters": self._hyperparameters,
            "metrics": self._metrics,
        })

        self.save()
        return self._metrics

    def predict(self, X: pd.DataFrame) -> list[dict]:
        start_time = time.time()
        y_pred = self.pipeline.predict(X)
        y_proba = self.pipeline.predict_proba(X)[:, 1]
        elapsed = round(time.time() - start_time, 4)

        logger.info(f"Предскзание завершено: n_objects={len(X)}, время={elapsed}с")

        return [
            {
                "churn": int(pred),
                "churn_probability": round(float(proba), 4),
                "churn_label": "churn" if int(pred) == 1 else "stay",
            }
            for pred, proba in zip(y_pred, y_proba)
        ]

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, MODEL_PATH)
        logger.info(f"Модель сохранена {MODEL_PATH}")

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            logger.info(f"Файл модели не найден в {MODEL_PATH}")
            return False

        loaded: ChurnModelService = joblib.load(MODEL_PATH)
        self._pipeline = loaded._pipeline
        self._model_type = getattr(loaded, "_model_type", None)
        self._hyperparameters = getattr(loaded, "_hyperparameters", None)
        self._trained_at = getattr(loaded, "_trained_at", None)
        self._metrics = getattr(loaded, "_metrics", None)
        logger.info(f"Model loaded from {MODEL_PATH}: model_type={self._model_type}")
        return True

    def status(self) -> dict:
        return {
            "is_trained": self.is_trained(),
            "trained_at": self._trained_at.isoformat() if self._trained_at else None,
            "metrics": self._metrics,
            "model_type": self._model_type,
            "hyperparameters": self._hyperparameters,
        }

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("Модель не обучена. Вызовите сначала POST /model/train.")
        return self._pipeline

    def is_trained(self) -> bool:
        return self._pipeline is not None


model_service = ChurnModelService()