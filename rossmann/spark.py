"""The local Spark session used by the pipeline."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pyspark.sql import SparkSession

class SetupError(RuntimeError):
    """Something to fix on the computer (like Java), explained in plain words."""


JAVA_HELP = "Install Java 17 or 21 (free: https://adoptium.net, 'Temurin'), then open a new terminal and try again."


def check_java(min_major: int = 17) -> int:
    """PySpark 4 runs on Java 17+. Fail early with a clear message instead of a long Java error."""
    java_home = os.environ.get("JAVA_HOME")
    exe = "java.exe" if os.name == "nt" else "java"
    java = str(Path(java_home) / "bin" / exe) if java_home else shutil.which("java")
    if not java or not Path(java).exists():
        where = f"JAVA_HOME points to {java_home}, but there is no {exe} there. " if java_home else ""
        raise SetupError(f"Java was not found. {where}{JAVA_HELP}")
    out = subprocess.run([java, "-version"], capture_output=True, text=True)
    found = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
    major = int(found.group(1)) if found else 0
    if major == 1 and found and found.group(2):  # old style: "1.8.0" means Java 8
        major = int(found.group(2))
    if major < min_major:
        raise SetupError(f"PySpark needs Java {min_major} or newer, but {java} is Java {major}. {JAVA_HELP}")
    if major > 21:
        logging.getLogger("rossmann").warning(
            "Java %d is newer than the versions Spark is tested with (17 and 21). "
            "If Spark fails to start, install Java 21 and set JAVA_HOME to it.", major)
    return major


def get_spark(app_name: str = "rossmann-promo-forecasting", cores: str = "*",
              driver_memory: str = "4g") -> SparkSession:
    """Start (or reuse) a Spark session that runs on this computer."""
    if SparkSession.getActiveSession() is None:
        check_java()
    # Spark's Python side uses this same interpreter (avoids "python3 not found" on Windows).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
