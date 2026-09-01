import json
from time import perf_counter, sleep

import numpy as np

from .algorithm import PSO
from .backend.distributed import SparkDistributed
from .backend.parallel import SparkParallel
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
    n_iter = 1_000_001  # Benchmark com 30 iter e 500 partículas
    sleep_time = 0.0  # Benchmark utiliza apenas tempo em consumo de CPU
    # n_iter = 10_001
    # sleep_time = 0.005  # Aproximadamente 75s a mais com 500 partículas e 30 iterações
    for _ in range(n_iter):
        result = result * 1.0000001
    result /= np.exp(1)
    sleep(sleep_time)
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
    processor = ThreadsPool()
    # processor = ProcessPool()
    processor.initialize_context(
        algorithm=algo,
        n_iter=30,
        n_particles=500,
        seed=42,
    )

    # processor.create_processors_pool(1)
    # executor = next(iter(processor.processors_pool.values()))
    # executor.initialize_execution_context()

    spark = generate_spark_session()
    backend = SparkDistributed(
        spark=spark,
        processor=processor,
        n_executors=3,
    )

    start = perf_counter()
    # executor.run()
    backend.execute()
    total_time = perf_counter() - start

    print(f"{total_time=}")
    print("Breakpoint here")


if __name__ == "__main__":
    main()
    # spark_parallel_test()
