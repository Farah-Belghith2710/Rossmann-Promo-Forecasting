"""Load the Kaggle CSV files with PySpark, check them, and clean them."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

TRAIN_SCHEMA = T.StructType([
    T.StructField("Store", T.IntegerType()),
    T.StructField("DayOfWeek", T.IntegerType()),
    T.StructField("Date", T.DateType()),
    T.StructField("Sales", T.DoubleType()),
    T.StructField("Customers", T.DoubleType()),
    T.StructField("Open", T.IntegerType()),
    T.StructField("Promo", T.IntegerType()),
    T.StructField("StateHoliday", T.StringType()),
    T.StructField("SchoolHoliday", T.IntegerType()),
])

# Numbers with gaps are read as doubles, so both "9" and "9.0" work, then turned into integers.
STORE_SCHEMA = T.StructType([
    T.StructField("Store", T.IntegerType()),
    T.StructField("StoreType", T.StringType()),
    T.StructField("Assortment", T.StringType()),
    T.StructField("CompetitionDistance", T.DoubleType()),
    T.StructField("CompetitionOpenSinceMonth", T.DoubleType()),
    T.StructField("CompetitionOpenSinceYear", T.DoubleType()),
    T.StructField("Promo2", T.IntegerType()),
    T.StructField("Promo2SinceWeek", T.DoubleType()),
    T.StructField("Promo2SinceYear", T.DoubleType()),
    T.StructField("PromoInterval", T.StringType()),
])

# Stores with no known competitor get this distance (in meters): "nobody close by".
NO_COMPETITOR_DISTANCE = 100_000.0


def unzip_kaggle_files(data_dir: Path) -> None:
    """If only the Kaggle zip is there (rossmann-store-sales.zip), take train.csv and store.csv out of it."""
    wanted = ("train.csv", "store.csv")
    if all((data_dir / name).exists() for name in wanted):
        return
    for archive in sorted(data_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            members = {Path(n).name: n for n in zf.namelist()}
            if all(name in members for name in wanted):
                for name in wanted:
                    with zf.open(members[name]) as src, open(data_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                return


def load_raw(spark: SparkSession, data_dir: str | Path) -> tuple[DataFrame, DataFrame]:
    """Read train.csv (daily sales) and store.csv (one row per store)."""
    data_dir = Path(data_dir)
    if data_dir.exists():
        unzip_kaggle_files(data_dir)
    for name in ("train.csv", "store.csv"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"{data_dir / name} is missing. Download the data from "
                "https://www.kaggle.com/competitions/rossmann-store-sales/data "
                "and put the zip (or train.csv and store.csv) in that folder."
            )
    sales = spark.read.csv(str(data_dir / "train.csv"), header=True, schema=TRAIN_SCHEMA, dateFormat="yyyy-MM-dd")
    stores = spark.read.csv(str(data_dir / "store.csv"), header=True, schema=STORE_SCHEMA)
    return sales, stores


def check_data(sales: DataFrame, stores: DataFrame) -> dict:
    """Basic checks before modeling. Stops if rows could not be read."""
    s = sales.agg(
        F.count("*").alias("rows"),
        F.countDistinct("Store").alias("stores"),
        F.min("Date").alias("first_day"),
        F.max("Date").alias("last_day"),
        F.sum(F.when(F.col("Store").isNull() | F.col("Date").isNull() | F.col("Sales").isNull(), 1)
              .otherwise(0)).alias("unreadable_rows"),
        F.sum(F.when((F.col("Open") == 1) & (F.col("Sales") == 0), 1).otherwise(0)).alias("open_days_without_sales"),
        F.sum(F.when(F.col("Open") == 0, 1).otherwise(0)).alias("closed_days"),
    ).first().asDict()
    if s["unreadable_rows"]:
        raise ValueError(f"{s['unreadable_rows']} rows of train.csv could not be read. Is it the Kaggle file?")

    per_store = sales.groupBy("Store").agg(F.min("Date").alias("first"), F.max("Date").alias("last"),
                                          F.count("*").alias("days"))
    s["stores_with_missing_days"] = per_store.filter(F.datediff("last", "first") + 1 > F.col("days")).count()
    s["duplicate_store_days"] = sales.groupBy("Store", "Date").count().filter("count > 1").count()
    s["stores_without_competition_distance"] = stores.filter(F.col("CompetitionDistance").isNull()).count()
    s["stores_in_store_file"] = stores.count()
    s["first_day"], s["last_day"] = str(s["first_day"]), str(s["last_day"])
    return s


def clean_sales(sales: DataFrame) -> DataFrame:
    """One row per store and day, and the same four holiday codes everywhere (0, a, b, c)."""
    return (
        sales.dropDuplicates(["Store", "Date"])
        .withColumn("StateHoliday",
                    F.when(F.col("StateHoliday").isin("a", "b", "c"), F.col("StateHoliday")).otherwise(F.lit("0")))
    )


def clean_stores(stores: DataFrame) -> DataFrame:
    """Turn the store numbers into integers and fill the few missing competition distances."""
    out = stores.withColumn("CompetitionDistance",
                            F.coalesce(F.col("CompetitionDistance"), F.lit(NO_COMPETITOR_DISTANCE)))
    for col in ("CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"):
        out = out.withColumn(col, F.col(col).cast("int"))
    return out.withColumn("PromoInterval", F.coalesce(F.col("PromoInterval"), F.lit("")))
