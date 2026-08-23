if __name__ == "__main__":
    from pyspark.sql import SparkSession

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

    df = spark.range(0, 1000000)

    print("Partitions:", df.rdd.getNumPartitions())
    print("Count:", df.count())

    print("Breakpoint here")

    spark.stop()
