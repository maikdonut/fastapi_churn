import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    n = 100
    return pd.DataFrame({
        "monthly_fee": np.random.uniform(10, 100, n),
        "usage_hours": np.random.uniform(0, 200, n),
        "support_requests": np.random.randint(0, 10, n),
        "account_age_months": np.random.randint(1, 60, n),
        "failed_payments": np.random.randint(0, 5, n),
        "region": np.random.choice(["europe", "asia", "america", "africa"], n),
        "device_type": np.random.choice(["mobile", "desktop", "tablet"], n),
        "payment_method": np.random.choice(["card", "paypal", "crypto"], n),
        "autopay_enabled": np.random.randint(0, 2, n),
        "churn": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })