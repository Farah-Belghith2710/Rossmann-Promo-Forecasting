"""Feature engineering with PySpark.

Every feature is something a retailer knows before the day it forecasts: the calendar, the
planned promotions, holidays and closures, the store's profile, and averages from the past.
The daily number of customers is never used as a feature, because nobody knows tomorrow's
customer count when they plan a promotion.
"""
from __future__ import annotations

import datetime as dt

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from .etl import clean_sales, clean_stores

DAYS_CAP = 30  # distances to events stop at 30 days ("a month or more away")
_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]


def add_calendar(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("Year", F.year("Date"))
        .withColumn("Month", F.month("Date"))
        .withColumn("Day", F.dayofmonth("Date"))
        .withColumn("WeekOfYear", F.weekofyear("Date"))
        .withColumn("DayOfYear", F.dayofyear("Date"))
    )


def days_until_next(flag: Column, cap: int = DAYS_CAP) -> Column:
    """Days until the next day where `flag` is true, per store (0 on the day itself)."""
    # A running minimum over the days in reverse order finds the next flagged date in one pass.
    w = (Window.partitionBy("Store").orderBy(F.col("Date").desc())
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    next_day = F.min(F.when(flag, F.col("Date"))).over(w)
    return F.coalesce(F.least(F.datediff(next_day, F.col("Date")), F.lit(cap)), F.lit(cap))


def days_since_last(flag: Column, cap: int = DAYS_CAP) -> Column:
    """Days since the last day where `flag` was true, per store (0 on the day itself)."""
    w = (Window.partitionBy("Store").orderBy(F.col("Date").asc())
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    last_day = F.max(F.when(flag, F.col("Date"))).over(w)
    return F.coalesce(F.least(F.datediff(F.col("Date"), last_day), F.lit(cap)), F.lit(cap))


def add_event_distances(df: DataFrame) -> DataFrame:
    """Days to and after public holidays and unusual closures (like a refurbishment).

    This needs every day of every store, closed days included, so it runs before the
    closed days are removed. Sundays don't count as closures: most stores never open then.
    """
    holiday = F.col("StateHoliday") != "0"
    closure = (F.col("Open") == 0) & (F.col("DayOfWeek") != 7) & ~holiday
    return (
        df.withColumn("DaysToHoliday", days_until_next(holiday))
        .withColumn("DaysAfterHoliday", days_since_last(holiday))
        .withColumn("DaysToClosure", days_until_next(closure))
        .withColumn("DaysAfterClosure", days_since_last(closure))
    )


def add_store_features(df: DataFrame, stores: DataFrame) -> DataFrame:
    """Store profile, nearby competition and the long-running "Promo2" campaign."""
    df = df.join(F.broadcast(clean_stores(stores)), on="Store", how="left")
    competition_months = ((F.col("Year") - F.col("CompetitionOpenSinceYear")) * 12
                          + (F.col("Month") - F.col("CompetitionOpenSinceMonth")))
    promo2_weeks = ((F.col("Year") - F.col("Promo2SinceYear")) * 52
                    + (F.col("WeekOfYear") - F.col("Promo2SinceWeek")))
    month_name = F.element_at(F.array(*[F.lit(m) for m in _MONTH_ABBR]), F.col("Month"))
    promo2_month = F.array_contains(F.split(F.col("PromoInterval"), ","), month_name)
    return (
        # empty when the competitor's opening date is unknown
        df.withColumn("CompetitionOpenMonths", competition_months)
        .withColumn("CompetitionActive",
                    F.when(competition_months.isNull() | (competition_months >= 0), 1).otherwise(0))
        .withColumn("Promo2Weeks", F.when(F.col("Promo2") == 1, promo2_weeks))
        .withColumn("IsPromo2Month",
                    F.when((F.col("Promo2") == 1) & (promo2_weeks >= 0) & promo2_month, 1).otherwise(0))
    )


def add_store_history(df: DataFrame, train_end: dt.date) -> DataFrame:
    """Store averages computed on the training period only, so the test weeks never leak in."""
    past = df.filter(F.col("Date") <= F.lit(train_end))
    weekday = F.col("DayOfWeek") <= 5  # promo vs non-promo compared on Mon-Fri, the days promos run
    per_store = past.groupBy("Store").agg(
        F.avg("Sales").alias("StoreAvgSales"),
        F.avg("Customers").alias("StoreAvgCustomers"),
        (F.sum("Sales") / F.sum("Customers")).alias("StoreSalesPerCustomer"),
        (F.avg(F.when(weekday & (F.col("Promo") == 1), F.col("Sales")))
         / F.avg(F.when(weekday & (F.col("Promo") == 0), F.col("Sales")))).alias("StorePromoRatio"),
    )
    per_weekday = past.groupBy("Store", "DayOfWeek").agg(F.avg("Sales").alias("_weekday_avg"))
    return (
        df.join(per_store, on="Store", how="left")
        .join(per_weekday, on=["Store", "DayOfWeek"], how="left")
        .withColumn("StoreDowRatio", F.col("_weekday_avg") / F.col("StoreAvgSales"))
        .drop("_weekday_avg")
    )


def build_features(sales: DataFrame, stores: DataFrame, train_end: dt.date) -> DataFrame:
    """All features, one row per store and open day with sales."""
    df = add_calendar(clean_sales(sales))
    df = add_event_distances(df)
    df = df.filter((F.col("Open") == 1) & (F.col("Sales") > 0))
    df = add_store_features(df, stores)
    return add_store_history(df, train_end)
