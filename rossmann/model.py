"""Forecast model, baseline, accuracy metrics and promotion uplift."""
from __future__ import annotations

import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

FEATURES = [
    # calendar
    "DayOfWeek", "Day", "Month", "Year", "WeekOfYear", "DayOfYear",
    # promotions, holidays and closures (all planned in advance)
    "Promo", "SchoolHoliday", "StateHoliday",
    "DaysToHoliday", "DaysAfterHoliday", "DaysToClosure", "DaysAfterClosure",
    # store profile and competition
    "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenMonths", "CompetitionActive",
    "Promo2", "Promo2Weeks", "IsPromo2Month",
    # store history, from the training period only
    "StoreAvgSales", "StoreAvgCustomers", "StoreSalesPerCustomer", "StorePromoRatio", "StoreDowRatio",
]
CATEGORIES = {"StoreType": list("abcd"), "Assortment": list("abc"), "StateHoliday": ["0", "a", "b", "c"]}

LGB_PARAMS = {
    "objective": "regression",  # on log(sales), so errors count in percent, like RMSPE
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def to_pandas(features: DataFrame) -> pd.DataFrame:
    """Bring the Spark features into pandas (about 850k rows for the full data)."""
    pdf = features.select("Store", "Date", "Sales", *FEATURES).toPandas()
    pdf["Date"] = pd.to_datetime(pdf["Date"])
    for col, cats in CATEGORIES.items():
        pdf[col] = pd.Categorical(pdf[col], categories=cats)
    return pdf.sort_values(["Date", "Store"], ignore_index=True)


# ---- accuracy ------------------------------------------------------------------------------

def rmspe(actual, predicted) -> float:
    """Root mean squared percentage error, the Kaggle metric (days without sales are skipped)."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    keep = actual > 0
    return float(np.sqrt(np.mean(((actual[keep] - predicted[keep]) / actual[keep]) ** 2)))


def accuracy(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    keep = actual > 0
    return {
        "rmspe": rmspe(actual, predicted),
        "mape": float(np.mean(np.abs((actual[keep] - predicted[keep]) / actual[keep]))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }


# ---- baseline ------------------------------------------------------------------------------

def median_baseline(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Median past sales of the same store, weekday and promo status (a common simple forecast)."""
    pred = pd.Series(np.nan, index=future.index)
    for keys in (["Store", "DayOfWeek", "Promo"], ["Store", "DayOfWeek"], ["Store"]):
        medians = train.groupby(keys, observed=True)["Sales"].median().rename("_m")
        filled = future[keys].merge(medians, left_on=keys, right_index=True, how="left")["_m"]
        pred = pred.fillna(pd.Series(filled.to_numpy(), index=future.index))
    return pred.fillna(train["Sales"].median()).to_numpy()


# ---- LightGBM ------------------------------------------------------------------------------

def _dataset(df: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(df[FEATURES], label=np.log1p(df["Sales"]), categorical_feature=list(CATEGORIES),
                       reference=reference, free_raw_data=False)


def train_lgbm(train: pd.DataFrame, valid_weeks: int = 6, max_rounds: int = 4000,
               early_stopping_rounds: int = 100, params: dict | None = None) -> tuple[lgb.Booster, int]:
    """Pick the number of trees on the last weeks of the training data, then refit on all of it."""
    params = {**LGB_PARAMS, **(params or {})}
    cut = train["Date"].max() - dt.timedelta(weeks=valid_weeks)
    fit_part, valid_part = train[train["Date"] <= cut], train[train["Date"] > cut]
    fit_set = _dataset(fit_part)
    probe = lgb.train(params, fit_set, num_boost_round=max_rounds,
                      valid_sets=[_dataset(valid_part, reference=fit_set)],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    best_rounds = probe.best_iteration or max_rounds
    booster = lgb.train(params, _dataset(train), num_boost_round=best_rounds)
    return booster, best_rounds


def predict(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return np.expm1(booster.predict(df[FEATURES]))


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    gain = booster.feature_importance(importance_type="gain")
    return (pd.DataFrame({"feature": booster.feature_name(), "gain_share": gain / gain.sum()})
            .sort_values("gain_share", ascending=False, ignore_index=True))


# ---- promotion uplift ----------------------------------------------------------------------

def promo_counterfactual(booster: lgb.Booster, df: pd.DataFrame, promo_weekdays: list[int]) -> pd.DataFrame:
    """Forecast every store-day twice, with and without the promotion, everything else unchanged.

    Only weekdays that had promotions in the past are used: on other days (like Saturdays in
    the Rossmann data) the model never saw a promotion, so its answer would be a guess.
    """
    days = df[df["DayOfWeek"].isin(promo_weekdays)].copy()
    days["SalesWithPromo"] = predict(booster, days.assign(Promo=1))
    days["SalesWithoutPromo"] = predict(booster, days.assign(Promo=0))
    return days


def summarize_uplift(days: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Uplift per group: model estimate, and the plain difference seen in the data for comparison."""
    grouped = days.groupby(by, observed=True) if by else days.groupby(lambda _: "all")
    out = grouped.apply(lambda g: pd.Series({
        "store_days": len(g),
        "model_uplift_pct": 100 * (g["SalesWithPromo"].sum() / g["SalesWithoutPromo"].sum() - 1),
        "extra_sales_per_store_day": (g["SalesWithPromo"] - g["SalesWithoutPromo"]).mean(),
        "observed_uplift_pct": 100 * (g.loc[g["Promo"] == 1, "Sales"].mean()
                                      / g.loc[g["Promo"] == 0, "Sales"].mean() - 1),
    }), include_groups=False)
    out.index.name = by or "group"
    return out.reset_index()
