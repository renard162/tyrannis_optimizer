import sys
from collections.abc import Generator

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    builder = SparkSession.builder.appName("tyrannis-test").config(
        "spark.ui.enabled", "false"
    )

    if sys.platform == "win32":
        builder = (
            builder.master("spark://172.22.76.60:7077")
            .config("spark.driver.bindAddress", "0.0.0.0")
            .config("spark.driver.host", "172.22.64.1")
            .config("spark.driver.port", "6060")
            .config("spark.blockManager.port", "6061")
        )
    else:
        builder = builder.master("local[2]")

    try:
        session = builder.getOrCreate()
    except Exception as exc:
        if sys.platform == "win32":
            raise RuntimeError(
                "Unable to create the Spark session.\n\n"
                "Spark integration tests on Windows require a fully functional "
                "Spark environment running in WSL.\n"
                "Start the Tyrannis Spark cluster in WSL and ensure that the "
                "Spark master is available at spark://172.22.76.60:7077.\n\n"
                "The Spark environment must be fully operational and use the same "
                "Python version and the same Python packages and package versions "
                "as the driver's Python environment."
            ) from exc

        raise

    session.sparkContext.setLogLevel("ERROR")

    try:
        yield session
    finally:
        session.stop()
