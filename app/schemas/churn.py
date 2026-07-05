from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Literal


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "monthly_fee": 19.99,
                "usage_hours": 42.5,
                "support_requests": 2,
                "account_age_months": 14,
                "failed_payments": 0,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 1,
            }
        }
    )

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "churn": 0,
                "churn_probability": 0.23,
                "churn_label": "stay",
            }
        }
    )

    churn: int = Field(..., ge=0, le=1, description="Предсказанный класс: 0 — остался, 1 — ушёл")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность оттока")
    churn_label: str = Field(..., description="Читаемый результат: 'churn' или 'stay'")

class TrainingConfigChurn(BaseModel):
    model_type: Literal["logreg", "random_forest"] = "logreg"
    hyperparameters: dict[str, Any] = {}
