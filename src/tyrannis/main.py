from __future__ import annotations

import time
from collections.abc import Iterator

import pandas as pd
from pyspark.sql import SparkSession


def empty_worker(
    batches: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    for batch in batches:
        yield pd.DataFrame(
            {
                "value": batch["value"],
            }
        )


def benchmark_spark_overhead(
    spark: SparkSession,
    n_particles: int = 100,
    n_iterations: int = 30,
) -> None:

    print("\n" + "=" * 72)
    print("SPARK OVERHEAD BENCHMARK")
    print("=" * 72)

    # Cria uma população fictícia.
    particle_ids = list(range(n_particles))

    total_start = time.perf_counter()

    for iteration in range(n_iterations):
        start = time.perf_counter()

        df = spark.createDataFrame(
            [(particle_id,) for particle_id in particle_ids],
            ["value"],
        )

        create_time = time.perf_counter() - start

        start = time.perf_counter()

        result = df.mapInPandas(
            empty_worker,
            schema="value long",
        )

        map_time = time.perf_counter() - start

        start = time.perf_counter()

        rows = result.collect()

        collect_time = time.perf_counter() - start

        iteration_time = create_time + map_time + collect_time

        print(
            f"iteration={iteration:02d} | "
            f"create={create_time:.4f}s | "
            f"map={map_time:.4f}s | "
            f"collect={collect_time:.4f}s | "
            f"total={iteration_time:.4f}s | "
            f"rows={len(rows)}"
        )

    total_time = time.perf_counter() - total_start

    print("=" * 72)
    print(f"TOTAL: {total_time:.4f}s")
    print(f"AVERAGE/STAGE: {total_time / n_iterations:.4f}s")
    print("=" * 72)
    print()


import json
from time import perf_counter, sleep

import numpy as np

from .algorithm import PSO
from .backend import Spark, SparkParallel
from .backend.processor import ProcessPool, Serial, ThreadsPool
from .examples.many_local_minima import ackley


def generate_spark_session():
    from pyspark.sql import SparkSession

    # Use spark.stop() to terminate spark

    spark = (
        SparkSession.builder.appName("SparkClusterTest")
        .master("spark://172.22.76.60:7077")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.host", "172.22.64.1")
        .config("spark.driver.port", "6060")
        .config("spark.blockManager.port", "6061")
        .getOrCreate()
    )

    print("Spark version:", spark.version)
    print("Master:", spark.sparkContext.master)
    print("Application ID:", spark.sparkContext.applicationId)
    return spark


def test_function(x):
    result = ackley(x)
    for _ in range(1_000_001):
        result = result * 1.0000001
    result /= np.exp(1)
    # sleep(0.25)
    return float(result)


def spark_parallel_test():
    algo = PSO()
    algo.initialize_context(
        fitness_function=test_function,
        boundaries={f"{n}": (-32.768, 32.768) for n in range(2)},
    )

    spark = generate_spark_session()

    backend = SparkParallel(spark)
    backend.initialize_context(
        algorithm=algo,
        n_iter=30,
        n_particles=500,
        seed=42,
    )

    start = perf_counter()
    backend.run()
    total_time = perf_counter() - start

    results = backend.algorithm.population

    print(f"{total_time=}")
    print("Breakpoint here")


def main():
    algo = PSO()
    algo.initialize_context(
        fitness_function=test_function,
        boundaries={f"{n}": (-32.768, 32.768) for n in range(2)},
    )

    # processor = Serial()
    # processor = ThreadsPool()
    processor = ProcessPool(n_process=7)
    processor.initialize_context(
        algorithm=algo,
        n_iter=30,
        n_particles=500,
        seed=42,
    )
    processor.create_processors_pool(1)
    executor = next(iter(processor.processors_pool.values()))
    executor.initialize_execution_context()
    # spark = generate_spark_session()

    # backend = Spark(
    #     spark=spark,
    #     processor=processor,
    # )

    start = perf_counter()
    executor.run()
    total_time = perf_counter() - start

    print(f"{total_time=}")
    print("Breakpoint here")


def test():
    import os
    import socket
    import time

    import pandas as pd
    from pyspark.sql import Row

    spark = generate_spark_session()
    df = spark.createDataFrame([Row(value=i) for i in range(300)]).repartition(10)

    def test_parallel(iterator):
        for batch in iterator:
            start = time.perf_counter()

            print(
                f"START pid={os.getpid()} host={socket.gethostname()} rows={len(batch)}"
            )

            time.sleep(5)

            elapsed = time.perf_counter() - start

            print(
                f"END "
                f"pid={os.getpid()} "
                f"host={socket.gethostname()} "
                f"rows={len(batch)} "
                f"elapsed={elapsed:.3f}s"
            )

            yield pd.DataFrame(
                {
                    "pid": [os.getpid()],
                    "hostname": [socket.gethostname()],
                    "rows": [len(batch)],
                    "elapsed": [elapsed],
                }
            )

    start = time.perf_counter()

    result = df.mapInPandas(
        test_parallel,
        "pid long, hostname string, rows long, elapsed double",
    ).collect()

    print(f"TOTAL: {time.perf_counter() - start:.3f}s")

    for row in result:
        print(row)
    print("Breakpoint here")


if __name__ == "__main__":
    # main()
    spark_parallel_test()
    # test()
    # generate_spark_session().stop()
    # spark = generate_spark_session()
    # benchmark_spark_overhead(
    #     spark,
    #     n_particles=100,
    #     n_iterations=30,
    # )
