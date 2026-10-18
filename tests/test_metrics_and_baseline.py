import numpy as np
import pandas as pd
import pytest

from rossmann.model import accuracy, median_baseline, rmspe


def test_rmspe_skips_days_without_sales():
    # errors of 10% and 10%; the day with 0 sales is ignored, as in the Kaggle metric
    assert rmspe([100, 200, 0], [110, 180, 50]) == pytest.approx(0.1)


def test_accuracy_has_all_metrics():
    out = accuracy([100, 200], [100, 200])
    assert out == {"rmspe": 0.0, "mape": 0.0, "rmse": 0.0}


def test_median_baseline_falls_back_when_a_combination_is_new():
    train = pd.DataFrame({
        "Store": [1, 1, 1, 1, 2],
        "DayOfWeek": [1, 1, 1, 2, 1],
        "Promo": [1, 1, 0, 0, 0],
        "Sales": [100.0, 120.0, 80.0, 50.0, 300.0],
    })
    future = pd.DataFrame({
        "Store": [1, 1, 2, 3],
        "DayOfWeek": [1, 2, 3, 1],
        "Promo": [1, 1, 0, 0],
    })
    pred = median_baseline(train, future)
    # same store/weekday/promo; same store/weekday; same store; overall median
    np.testing.assert_allclose(pred, [110.0, 50.0, 300.0, 100.0])
