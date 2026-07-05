from app.services.dataset import ChurnDatasetService


def test_dataset_preview_returns_requested_rows():
    service = ChurnDatasetService()
    rows = service.preview(n=3)

    assert len(rows) == 3
    assert rows[0].monthly_fee == 9.99
    assert rows[0].churn in (0, 1)


def test_dataset_info_contains_expected_keys():
    service = ChurnDatasetService()
    info = service.info()

    assert info["n_rows"] == 2000
    assert "churn" not in info["feature_names"]
    assert set(info["churn_distribution"].keys()) == {0, 1}
