import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.dataset import dataset_service
from app.services.model import model_service


@pytest.fixture
def client(sample_df, monkeypatch):
    from app.services.preprocessing import preprocessing_service

    dataset_service._df = sample_df

    # сбрасываем кеш preprocessing
    preprocessing_service._X_train = None
    preprocessing_service._X_test = None
    preprocessing_service._y_train = None
    preprocessing_service._y_test = None

    monkeypatch.setattr(model_service, "load", lambda: False)
    monkeypatch.setattr(model_service, "save", lambda: None)

    model_service._pipeline = None
    model_service._trained_at = None
    model_service._metrics = None
    model_service._model_type = None
    model_service._hyperparameters = None

    with TestClient(app) as c:
        yield c

@pytest.fixture
def trained_client(client):
    """TestClient с уже обученной моделью."""
    client.post("/model/train", json={
        "model_type": "logreg",
        "hyperparameters": {"class_weight": "balanced"},
    })
    return client


VALID_FEATURES = {
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


# --- health ---

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "ml churn service is running"


# --- dataset ---

def test_dataset_preview(client):
    response = client.get("/dataset/preview?n=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert "monthly_fee" in data[0]
    assert "churn" in data[0]


def test_dataset_info(client):
    response = client.get("/dataset/info")
    assert response.status_code == 200
    data = response.json()
    assert data["n_rows"] == 100
    assert "churn_distribution" in data
    assert "feature_names" in data


# --- model/train ---

def test_train_default_params(client):
    response = client.post("/model/train")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "f1" in data
    assert "roc_auc" in data


def test_train_logreg(client):
    response = client.post("/model/train", json={
        "model_type": "logreg",
        "hyperparameters": {"class_weight": "balanced"},
    })
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["accuracy"] <= 1.0
    assert 0.0 <= data["roc_auc"] <= 1.0


def test_train_random_forest(client):
    response = client.post("/model/train", json={
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 10},
    })
    assert response.status_code == 200


def test_train_invalid_model_type(client):
    response = client.post("/model/train", json={
        "model_type": "xgboost",
        "hyperparameters": {},
    })
    assert response.status_code == 422
    data = response.json()
    assert data["code"] == "VALIDATION_ERROR"


# --- model/status ---

def test_status_before_train(client):
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_trained"] is False
    assert data["trained_at"] is None


def test_status_after_train(trained_client):
    response = trained_client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_trained"] is True
    assert data["trained_at"] is not None
    assert data["model_type"] == "logreg"


# --- predict ---

def test_predict_without_model(client):
    response = client.post("/predict", json=[VALID_FEATURES])
    assert response.status_code == 503
    data = response.json()
    assert data["code"] == "MODEL_NOT_TRAINED"


def test_predict_with_model(trained_client):
    response = trained_client.post("/predict", json=[VALID_FEATURES])
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["churn"] in [0, 1]
    assert 0.0 <= data[0]["churn_probability"] <= 1.0
    assert data[0]["churn_label"] in ["churn", "stay"]


def test_predict_single_object(trained_client):
    response = trained_client.post("/predict", json=VALID_FEATURES)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["churn"] in [0, 1]
    assert 0.0 <= data[0]["churn_probability"] <= 1.0
    assert data[0]["churn_label"] in ["churn", "stay"]


def test_predict_batch(trained_client):
    response = trained_client.post("/predict", json=[VALID_FEATURES, VALID_FEATURES])
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_predict_empty_list(trained_client):
    response = trained_client.post("/predict", json=[])
    assert response.status_code == 200
    assert response.json() == []


def test_predict_invalid_data(trained_client):
    invalid = {**VALID_FEATURES, "monthly_fee": "дорого"}
    response = trained_client.post("/predict", json=[invalid])
    assert response.status_code == 422
    data = response.json()
    assert data["code"] == "VALIDATION_ERROR"


# --- model/metrics ---

def test_metrics_empty_history(client):
    from app.services.history import history_service
    history_service._records = []

    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["latest"] is None
    assert data["history"] == []
    assert data["total_runs"] == 0


def test_metrics_after_train(trained_client):
    from app.services.history import history_service
    history_service._records = []

    trained_client.post("/model/train", json={
        "model_type": "logreg",
        "hyperparameters": {"class_weight": "balanced"},
    })

    response = trained_client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] == 1
    assert data["latest"]["model_type"] == "logreg"


def test_metrics_filter_by_model_type(trained_client):
    from app.services.history import history_service
    history_service._records = []

    trained_client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {}})
    trained_client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}})

    response = trained_client.get("/model/metrics?model_type=logreg")
    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] == 1
    assert data["latest"]["model_type"] == "logreg"