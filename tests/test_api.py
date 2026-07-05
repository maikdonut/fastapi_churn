from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}


def test_dataset_preview():
    response = client.get("/dataset/preview?n=2")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_model_train():
    response = client.get("/model/train")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "f1" in data
