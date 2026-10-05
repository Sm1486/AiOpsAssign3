import time
import os
import psutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.types import DoubleType

TRIP_DATA_DIR = "TrafficData/Subsample"
ZONE_CSV = "TrafficData/taxi_zone_lookup.csv"
OUTPUT_DIR = "output/spark_output"
CLEAN_DIR = f"{OUTPUT_DIR}/trips_clean"
SPARK_TMP = "/home/vboxuser/spark-tmp"

os.makedirs(SPARK_TMP, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Spark-NYC-Taxi")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "32")
    .config("spark.local.dir", "/home/vboxuser/spark-tmp")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.files.maxPartitionBytes", "64m")
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    .config("spark.hadoop.parquet.hadoop.vectored.io.enabled", "false")
    .config("spark.sql.parquet.compression.codec", "gzip")
    .config("spark.io.compression.codec", "lz4") 
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("local dir:", spark.sparkContext.getConf().get("spark.local.dir", "NOT SET"))

time_start = time.perf_counter()
proc = psutil.Process(os.getpid())

REQUIRED_COLS = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime","passenger_count", "trip_distance", "PULocationID","DOLocationID", "fare_amount", "total_amount"]

trips_raw = spark.read.parquet(f"{TRIP_DATA_DIR}/*.parquet").select(*REQUIRED_COLS)
zones_raw = spark.read.option("header", True).csv(ZONE_CSV)

trips_raw.printSchema()

trips_clean = (
    trips_raw
    .dropna(subset=REQUIRED_COLS)
    .filter(F.col("trip_distance") > 0)
    .filter(F.col("fare_amount") > 0)
    .filter(F.col("passenger_count") >= 1)
    .withColumn("tpep_pickup_datetime", F.to_timestamp(F.col("tpep_pickup_datetime")))
    .withColumn("tpep_dropoff_datetime", F.to_timestamp(F.col("tpep_dropoff_datetime")))
    .filter(F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
    .dropDuplicates(["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime","PULocationID", "DOLocationID", "total_amount"])
    .withColumn("PULocationID", F.col("PULocationID").cast("int"))
    .withColumn("DOLocationID", F.col("DOLocationID").cast("int"))
)

trips_featured = (
    trips_clean
    .withColumn("trip_duration_min", (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0)
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn("pickup_dow", F.dayofweek("tpep_pickup_datetime"))
    .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
)

trips_featured.write.mode("overwrite").parquet(CLEAN_DIR)
trips_featured = spark.read.parquet(CLEAN_DIR)

trips_featured.explain(True)

native_start = time.perf_counter()

trips_with_speed = (
    trips_featured
    .withColumn("avg_speed_mph", F.col("trip_distance") / (F.col("trip_duration_min") / 60.0))
    .filter(F.col("avg_speed_mph") <= 150)
)

row_count = trips_with_speed.count()
native_elapsed = time.perf_counter() - native_start
print(f"Rows after speed filter: {row_count}")

zones_pu = (
    zones_raw
    .withColumnRenamed("LocationID", "PULocationID")
    .withColumnRenamed("Borough", "pickup_borough")
    .withColumnRenamed("Zone", "pickup_zone")
    .withColumnRenamed("service_zone", "pickup_service_zone")
    .withColumn("PULocationID", F.col("PULocationID").cast("int"))
    .select("PULocationID", "pickup_borough", "pickup_zone", "pickup_service_zone")
)

zones_do = (
    zones_raw
    .withColumnRenamed("LocationID", "DOLocationID")
    .withColumnRenamed("Borough", "dropoff_borough")
    .withColumnRenamed("Zone", "dropoff_zone")
    .withColumnRenamed("service_zone", "dropoff_service_zone")
    .withColumn("DOLocationID", F.col("DOLocationID").cast("int"))
    .select("DOLocationID", "dropoff_borough", "dropoff_zone", "dropoff_service_zone")
)

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 8 * 1024 * 1024)

trips_joined = (
    trips_with_speed
    .join(broadcast(zones_pu), on="PULocationID", how="left")
    .join(broadcast(zones_do), on="DOLocationID", how="left")
)

trips_joined.explain("formatted")

speed_by_hour = (
    trips_joined
    .groupBy("pickup_hour")
    .agg(
        F.count("*").alias("trip_count"),
        F.avg("avg_speed_mph").alias("mean_speed_mph"),
        F.avg("trip_distance").alias("mean_distance_miles"),
        F.avg("trip_duration_min").alias("mean_duration_min"),
    )
    .orderBy("pickup_hour")
)

speed_by_hour.show(24, truncate=False)

(trips_joined
    .repartition("pickup_date")
    .write.mode("overwrite").partitionBy("pickup_date")
    .parquet(f"{OUTPUT_DIR}/trips_enriched"))
speed_by_hour.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/speed_by_hour")

time_elapsed = time.perf_counter() - time_start
mem_mb = proc.memory_info().rss / (1024 ** 2)

def compute_avg_speed_mph(distance_miles, duration_min):
    if duration_min is None or duration_min <= 0:
        return None
    return float(distance_miles / (duration_min / 60.0))

avg_speed_udf = F.udf(compute_avg_speed_mph, DoubleType())

udf_sample_elapsed = None
native_sample_elapsed = None
try:
    sample = trips_featured.sample(fraction=0.01, seed=42).cache()
    sample.count()

    t0 = time.perf_counter()
    sample.withColumn("s", avg_speed_udf(F.col("trip_distance"), F.col("trip_duration_min"))) \
          .agg(F.sum("s")).collect()
    udf_sample_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    sample.withColumn("s", F.col("trip_distance") / (F.col("trip_duration_min") / 60.0)) \
          .agg(F.sum("s")).collect()
    native_sample_elapsed = time.perf_counter() - t0

    sample.unpersist()
except Exception as e:
    print("UDF timing failed (Python worker issue, likely Python 3.14):", str(e)[:200])

print(f"The total time taken (main pipeline): {time_elapsed:.2f}s")
print(f"Native speed step on full data: {native_elapsed:.2f}s")
if udf_sample_elapsed is not None:
    print(f"UDF on 1% sample: {udf_sample_elapsed:.2f}s | native on same sample: {native_sample_elapsed:.2f}s")
print(f"Driver RSS memory: {mem_mb:.1f} MB")

spark.stop()