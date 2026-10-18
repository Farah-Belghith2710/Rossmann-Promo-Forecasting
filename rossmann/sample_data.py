"""A small synthetic dataset with the same files and columns as the Kaggle Rossmann data.

It exists for the tests and for trying the pipeline without a Kaggle account. Because the
data is generated, the real promotion effect per store type is known (TRUE_PROMO_UPLIFT),
so the tests can check that the pipeline finds it back.

Every number in the README comes from the real Kaggle data, not from this file.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# Promotion effect used to generate the data, per store type (0.30 = +30% sales on promo days).
TRUE_PROMO_UPLIFT = {"a": 0.30, "b": 0.10, "c": 0.20, "d": 0.25}

# Sales level per weekday, Monday = 1 ... Sunday = 7 (only type "b" stores open on Sundays).
_WEEKDAY_FACTOR = np.array([1.18, 1.03, 0.97, 0.96, 1.04, 0.88, 0.80])

_EASTER = {2013: "2013-03-31", 2014: "2014-04-20", 2015: "2015-04-05", 2016: "2016-03-27"}
_PROMO_INTERVALS = ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"]
_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]


def _state_holidays(years) -> dict[pd.Timestamp, str]:
    """Public holidays with the Kaggle codes: a = public holiday, b = Easter, c = Christmas."""
    days: dict[pd.Timestamp, str] = {}
    for y in years:
        for month, day in [(1, 1), (5, 1), (10, 3)]:
            days[pd.Timestamp(y, month, day)] = "a"
        easter = pd.Timestamp(_EASTER[y])
        days[easter - dt.timedelta(days=2)] = "b"
        days[easter + dt.timedelta(days=1)] = "b"
        days[pd.Timestamp(y, 12, 25)] = "c"
        days[pd.Timestamp(y, 12, 26)] = "c"
    return days


def make_sample_data(out_dir: str | Path, n_stores: int = 60, start: str = "2013-01-01",
                     end: str = "2015-07-31", seed: int = 7) -> Path:
    """Write train.csv and store.csv (Kaggle format) into out_dir and return out_dir."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    stores = np.arange(1, n_stores + 1)

    # ---- stores -------------------------------------------------------------------------
    store_type = rng.choice(list("abcd"), size=n_stores, p=[0.5, 0.1, 0.15, 0.25])
    store_type[:8] = list("abcdabcd")  # every type appears at least twice
    assortment = np.where(store_type == "b", "b", rng.choice(["a", "c"], size=n_stores, p=[0.55, 0.45]))
    base_sales = rng.lognormal(np.log(6000), 0.35, n_stores)
    spend_per_customer = rng.normal(9.5, 1.2, n_stores).clip(6, 14)

    comp_distance = np.round(rng.lognormal(np.log(2500), 1.0, n_stores), -1)
    comp_known = rng.random(n_stores) < 0.7
    comp_year = np.where(comp_known, rng.integers(2005, 2016, n_stores), -1)
    comp_month = np.where(comp_known, rng.integers(1, 13, n_stores), -1)

    promo2 = rng.random(n_stores) < 0.5
    promo2_year = np.where(promo2, rng.integers(2009, 2015, n_stores), -1)
    promo2_week = np.where(promo2, rng.integers(1, 50, n_stores), -1)
    promo2_interval = np.where(promo2, rng.choice(_PROMO_INTERVALS, n_stores), "")

    # ---- calendar -----------------------------------------------------------------------
    weekday = dates.dayofweek.to_numpy()  # Monday = 0
    monday = dates - pd.to_timedelta(weekday, unit="D")
    week_index = ((monday - monday[0]).days // 7).to_numpy()
    promo_day = (week_index % 2 == 0) & (weekday < 5)  # Monday-Friday, every other week

    holidays = _state_holidays(range(dates[0].year, dates[-1].year + 1))
    state_holiday = np.array([holidays.get(d, "0") for d in dates])
    is_holiday = state_holiday != "0"
    day_before_holiday = np.roll(is_holiday, -1) & ~is_holiday

    month, day = dates.month.to_numpy(), dates.day.to_numpy()
    school_holiday = ((month == 7) & (day >= 20)) | (month == 8) | ((month == 12) & (day >= 22)) | ((month == 1) & (day <= 5))

    season = 1 + 0.06 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    christmas = np.where((month == 12) & (day <= 23), 1 + 0.35 * day / 23, 1.0)
    trend = 1.03 ** ((dates - dates[0]).days.to_numpy() / 365.25)
    common = season * christmas * trend * np.where(day_before_holiday, 1.08, 1.0) * np.where(school_holiday, 1.05, 1.0)

    # ---- daily sales, one row per store and day ----------------------------------------
    rows = []
    for i, store in enumerate(stores):
        is_open = ~is_holiday & ((weekday < 6) | (store_type[i] == "b"))

        boost = np.ones(n_days)
        if rng.random() < 0.15:  # a refurbishment: closed 2-3 weeks, busy week after reopening
            first = int(rng.integers(60, n_days - 90))
            last = first + int(rng.integers(14, 22))
            is_open[first:last] = False
            boost[last:last + 7] = 1.25

        months_in_promo2 = {_MONTH_ABBR.index(m) + 1 for m in promo2_interval[i].split(",") if m}
        promo2_started = (dates.year.to_numpy() * 100 + dates.isocalendar().week.to_numpy()) >= (promo2_year[i] * 100 + promo2_week[i])
        promo2_month = promo2[i] & promo2_started & np.isin(month, list(months_in_promo2))

        sales = (base_sales[i] * _WEEKDAY_FACTOR[weekday] * common * boost
                 * (1 + TRUE_PROMO_UPLIFT[store_type[i]] * promo_day)
                 * np.where(promo2_month, 1.03, 1.0)
                 * rng.lognormal(0, 0.07, n_days))
        sales = np.where(is_open, np.round(sales), 0).astype(int)
        customers = np.where(is_open, np.round(sales / spend_per_customer[i] * rng.lognormal(0, 0.03, n_days)), 0).astype(int)

        rows.append(pd.DataFrame({
            "Store": store,
            "DayOfWeek": weekday + 1,
            "Date": dates.strftime("%Y-%m-%d"),
            "Sales": sales,
            "Customers": customers,
            "Open": is_open.astype(int),
            "Promo": promo_day.astype(int),
            "StateHoliday": state_holiday,
            "SchoolHoliday": school_holiday.astype(int),
        }))

    train = pd.concat(rows, ignore_index=True).sort_values(["Date", "Store"], ascending=[False, True])
    train.to_csv(out_dir / "train.csv", index=False)

    def nullable(values):
        return pd.array(np.where(values < 0, pd.NA, values), dtype="Int64")

    pd.DataFrame({
        "Store": stores,
        "StoreType": store_type,
        "Assortment": assortment,
        "CompetitionDistance": comp_distance.astype(int),
        "CompetitionOpenSinceMonth": nullable(comp_month),
        "CompetitionOpenSinceYear": nullable(comp_year),
        "Promo2": promo2.astype(int),
        "Promo2SinceWeek": nullable(promo2_week),
        "Promo2SinceYear": nullable(promo2_year),
        "PromoInterval": pd.array(np.where(promo2, promo2_interval, None), dtype="string"),
    }).to_csv(out_dir / "store.csv", index=False)
    return out_dir
