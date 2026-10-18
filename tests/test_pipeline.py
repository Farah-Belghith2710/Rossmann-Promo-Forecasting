"""End to end on synthetic data where the true promotion effect is known."""
import pytest

from rossmann.pipeline import run
from rossmann.sample_data import TRUE_PROMO_UPLIFT, make_sample_data


@pytest.fixture(scope="module")
def results(spark, tmp_path_factory):
    data = make_sample_data(tmp_path_factory.mktemp("data"), n_stores=40)
    reports = tmp_path_factory.mktemp("reports")
    return run(data, reports, max_rounds=600), reports


def test_forecast_is_accurate_and_beats_the_baseline(results):
    metrics, _ = results
    acc = metrics["accuracy_on_test_weeks"]
    assert acc["lightgbm"]["rmspe"] < 0.10
    assert acc["lightgbm"]["rmspe"] < acc["median_baseline"]["rmspe"]


def test_promo_uplift_matches_the_truth(results):
    metrics, _ = results
    assert metrics["promo_uplift"]["weekdays_used"] == [1, 2, 3, 4, 5]
    for row in metrics["promo_uplift"]["by_store_type"]:
        true_pct = 100 * TRUE_PROMO_UPLIFT[row["StoreType"]]
        assert row["model_uplift_pct"] == pytest.approx(true_pct, abs=5), row


def test_reports_are_written(results):
    _, reports = results
    assert (reports / "metrics.json").exists()
    assert (reports / "promo_uplift_by_store_type.csv").exists()
    assert len(list((reports / "figures").glob("*.png"))) == 8  # 4 charts, light and dark
