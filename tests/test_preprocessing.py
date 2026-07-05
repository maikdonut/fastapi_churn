from app.services.preprocessing import ChurnPreprocessingService


def test_split_info_returns_train_and_test_sizes():
    service = ChurnPreprocessingService(test_size=0.2, random_state=42)
    info = service.split_info()

    assert info["train_size"] == 1600
    assert info["test_size"] == 400
    assert info["train_churn_distribution"]["churn_0"] + info["train_churn_distribution"]["churn_1"] == 1600
    assert info["test_churn_distribution"]["churn_0"] + info["test_churn_distribution"]["churn_1"] == 400
