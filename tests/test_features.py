import datetime as dt

from pyspark.sql import Row

from rossmann import etl, features


def _days(spark, rows):
    """rows: (date, open, state_holiday) for store 1, every other column filled in."""
    data = [Row(Store=1, DayOfWeek=d.isoweekday(), Date=d, Sales=100.0 if o else 0.0, Customers=10.0 if o else 0.0,
                Open=o, Promo=0, StateHoliday=h, SchoolHoliday=0) for d, o, h in rows]
    return spark.createDataFrame(data, schema=etl.TRAIN_SCHEMA)


def test_days_to_and_after_holidays_and_closures(spark):
    mon = dt.date(2015, 3, 2)
    day = [mon + dt.timedelta(days=i) for i in range(7)]  # Mon 2 ... Sun 8 March 2015
    df = _days(spark, [
        (day[0], 1, "0"),
        (day[1], 1, "0"),
        (day[2], 0, "a"),  # Wednesday: public holiday, closed (not an "unusual closure")
        (day[3], 1, "0"),
        (day[4], 0, "0"),  # Friday: closed on a normal day -> unusual closure
        (day[5], 1, "0"),
        (day[6], 0, "0"),  # Sunday: closed as usual, not a closure
    ])
    out = {r["Date"]: r for r in features.add_event_distances(df).collect()}
    cap = features.DAYS_CAP

    assert [out[d]["DaysToHoliday"] for d in day] == [2, 1, 0, cap, cap, cap, cap]
    assert [out[d]["DaysAfterHoliday"] for d in day] == [cap, cap, 0, 1, 2, 3, 4]
    assert [out[d]["DaysToClosure"] for d in day] == [4, 3, 2, 1, 0, cap, cap]
    assert [out[d]["DaysAfterClosure"] for d in day] == [cap, cap, cap, cap, 0, 1, 2]


def test_competition_and_promo2_features(spark):
    stores = spark.createDataFrame([
        # competitor opened May 2015; Promo2 since week 10 of 2014 in Mar/Jun/Sept/Dec
        (1, "a", "a", 500.0, 5.0, 2015.0, 1, 10.0, 2014.0, "Mar,Jun,Sept,Dec"),
        # competitor with unknown opening date, no Promo2, missing distance
        (2, "c", "c", None, None, None, 0, None, None, None),
    ], schema=etl.STORE_SCHEMA)
    sales = spark.createDataFrame([
        Row(Store=s, DayOfWeek=d.isoweekday(), Date=d, Sales=100.0, Customers=10.0, Open=1, Promo=0,
            StateHoliday="0", SchoolHoliday=0)
        for s, d in [(1, dt.date(2015, 3, 10)), (1, dt.date(2015, 7, 14)), (1, dt.date(2015, 9, 15)),
                     (1, dt.date(2013, 9, 10)), (2, dt.date(2015, 7, 14))]
    ], schema=etl.TRAIN_SCHEMA)
    rows = features.add_store_features(features.add_calendar(sales), stores).collect()
    out = {(r["Store"], r["Date"]): r for r in rows}

    march = out[(1, dt.date(2015, 3, 10))]
    assert (march["CompetitionOpenMonths"], march["CompetitionActive"]) == (-2, 0)  # not open yet
    assert march["IsPromo2Month"] == 1

    july = out[(1, dt.date(2015, 7, 14))]
    assert (july["CompetitionOpenMonths"], july["CompetitionActive"], july["IsPromo2Month"]) == (2, 1, 0)

    assert out[(1, dt.date(2015, 9, 15))]["IsPromo2Month"] == 1  # "Sept", as written in the Kaggle file
    assert out[(1, dt.date(2013, 9, 10))]["IsPromo2Month"] == 0  # before Promo2 started

    unknown = out[(2, dt.date(2015, 7, 14))]
    assert unknown["CompetitionOpenMonths"] is None and unknown["CompetitionActive"] == 1
    assert unknown["CompetitionDistance"] == etl.NO_COMPETITOR_DISTANCE
    assert unknown["Promo2Weeks"] is None and unknown["IsPromo2Month"] == 0


def test_store_history_uses_only_the_training_period(spark):
    rows = [Row(Store=1, DayOfWeek=d.isoweekday(), Date=d, Sales=s, Customers=10.0, Open=1, Promo=0,
                StateHoliday="0", SchoolHoliday=0)
            for d, s in [(dt.date(2015, 6, 1), 100.0), (dt.date(2015, 6, 2), 300.0),
                         (dt.date(2015, 7, 1), 10_000.0)]]  # after train_end: must be ignored
    df = spark.createDataFrame(rows, schema=etl.TRAIN_SCHEMA)
    out = features.add_store_history(df, dt.date(2015, 6, 30)).collect()
    assert {r["StoreAvgSales"] for r in out} == {200.0}
