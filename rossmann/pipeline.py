"""Run everything: load -> check -> features (PySpark) -> forecast (LightGBM) -> promo uplift -> report.

    python -m rossmann.pipeline                      # real data in data/raw
    python -m rossmann.pipeline --sample             # small synthetic data, no Kaggle account needed
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

from . import etl, features, model, plots, report
from .sample_data import make_sample_data
from .spark import SetupError, get_spark

log = logging.getLogger("rossmann")


def run(data_dir: str | Path = "data/raw", reports_dir: str | Path = "reports", holdout_weeks: int = 6,
        max_rounds: int = 4000, make_charts: bool = True, readme: str | Path | None = None) -> dict:
    """Run the whole pipeline. If `readme` is given, its results section is rewritten too."""
    started = time.time()
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and check the raw files, then build the features with Spark
    already_running = SparkSession.getActiveSession() is not None
    spark = get_spark()
    try:
        sales, stores = etl.load_raw(spark, data_dir)
        sales = sales.cache()
        quality = etl.check_data(sales, stores)
        log.info("Data: %s rows, %s stores, %s to %s", f"{quality['rows']:,}", quality["stores"],
                 quality["first_day"], quality["last_day"])

        last_day = dt.date.fromisoformat(quality["last_day"])
        holdout_start = last_day - dt.timedelta(weeks=holdout_weeks) + dt.timedelta(days=1)
        train_end = holdout_start - dt.timedelta(days=1)
        table = model.to_pandas(features.build_features(sales, stores, train_end))
        sales.unpersist()
    finally:
        if not already_running:  # leave a session we didn't start alone (e.g. in the tests)
            spark.stop()
    log.info("Features: %s open store-days, %s features (%.0fs)", f"{len(table):,}", len(model.FEATURES),
             time.time() - started)

    # 2. Time-based split: the last weeks are only used to measure the forecast
    train = table[table["Date"] < np.datetime64(holdout_start)]
    hold = table[table["Date"] >= np.datetime64(holdout_start)].copy()

    hold["Baseline"] = model.median_baseline(train, hold)
    booster, rounds = model.train_lgbm(train, valid_weeks=holdout_weeks, max_rounds=max_rounds)
    hold["LightGBM"] = model.predict(booster, hold)
    log.info("LightGBM: %d trees (%.0fs)", rounds, time.time() - started)

    # 3. Promotion uplift on the weekdays that had promotions in the training data
    promo_weekdays = sorted(int(d) for d in train.loc[train["Promo"] == 1, "DayOfWeek"].unique())
    days = model.promo_counterfactual(booster, hold, promo_weekdays)
    uplift_all = model.summarize_uplift(days)
    by_type = model.summarize_uplift(days, "StoreType")
    by_weekday = model.summarize_uplift(days, "DayOfWeek")
    by_assortment = model.summarize_uplift(days, "Assortment")
    importance = model.feature_importance(booster)

    # 4. Save the numbers and the charts
    metrics = {
        "data": quality,
        "split": {
            "train_period": f"{quality['first_day']} to {train_end}",
            "test_period": f"{holdout_start} to {last_day}",
            "train_store_days": int(len(train)),
            "test_store_days": int(len(hold)),
            "test_stores": int(hold["Store"].nunique()),
        },
        "accuracy_on_test_weeks": {
            "median_baseline": model.accuracy(hold["Sales"], hold["Baseline"]),
            "lightgbm": model.accuracy(hold["Sales"], hold["LightGBM"]),
        },
        "lightgbm": {"trees": rounds, "params": model.LGB_PARAMS, "features": model.FEATURES},
        "promo_uplift": {
            "weekdays_used": promo_weekdays,
            "overall": uplift_all.drop(columns="group").iloc[0].to_dict(),
            "by_store_type": by_type.to_dict(orient="records"),
            "by_weekday": by_weekday.to_dict(orient="records"),
            "by_assortment": by_assortment.to_dict(orient="records"),
        },
        "top_features": importance.head(10).to_dict(orient="records"),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    for name, df in [("promo_uplift_by_store_type", by_type), ("promo_uplift_by_weekday", by_weekday),
                     ("promo_uplift_by_assortment", by_assortment), ("feature_importance", importance)]:
        df.round(4).to_csv(reports_dir / f"{name}.csv", index=False)
    if make_charts:
        plots.make_all(hold, by_type, by_weekday, importance, reports_dir / "figures")

    if readme and make_charts and report.update_readme(metrics, readme, reports_dir):
        log.info("Results written into %s", readme)

    acc = metrics["accuracy_on_test_weeks"]
    log.info("RMSPE on the test weeks: LightGBM %.3f vs baseline %.3f | promo uplift %+.1f%% | %.0fs total",
             acc["lightgbm"]["rmspe"], acc["median_baseline"]["rmspe"],
             metrics["promo_uplift"]["overall"]["model_uplift_pct"], metrics["runtime_seconds"])
    return metrics


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rossmann sales forecast and promotion uplift.")
    parser.add_argument("--data-dir", default="data/raw", help="folder with train.csv and store.csv")
    parser.add_argument("--reports-dir", default=None, help="where results go (default: reports/)")
    parser.add_argument("--holdout-weeks", type=int, default=6, help="weeks kept aside to test the forecast")
    parser.add_argument("--sample", action="store_true", help="generate and use a small synthetic dataset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    data_dir, reports_dir, readme = args.data_dir, args.reports_dir or "reports", "README.md"
    if args.sample:
        data_dir = make_sample_data("data/sample")
        reports_dir, readme = args.reports_dir or "reports/sample", None
        log.info("Using synthetic sample data in %s (results go to %s)", data_dir, reports_dir)
    try:
        run(data_dir, reports_dir, holdout_weeks=args.holdout_weeks, readme=readme)
    except (FileNotFoundError, SetupError) as problem:  # missing data or Java: a clear message, no traceback
        log.error("%s", problem)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
