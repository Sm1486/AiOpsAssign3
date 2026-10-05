import time
import os
import psutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.types import DoubleType

TRIP_DATA_DIR = "TripData"
ZONE_CSV      = "TripData/taxi_zone_lookup.csv"
OUTPUT_DIR    = "output/spark_output"

spark = (
    SparkSession.builder
    .appName("DA3408-A3-Spark-NYC-Taxi")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "16") 
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.sql.autoBroadcastJoinThreshold", -1)
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

sc = spark.sparkContext

time_start = time.perf_counter()
proc       = psutil.Process(os.getpid())

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — INGESTION
# Pattern: Lecture 7 -- Spark_ETL_Exercise.ipynb  Step 1
#   df = spark.read.parquet(DATA_DIR)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — INGESTION")
print("=" * 60)

trips_raw = spark.read.parquet(TRIP_DATA_DIR)
zones_raw = spark.read.option("header", True).csv(ZONE_CSV)

trips_raw.printSchema()


REQUIRED_COLS = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime","passenger_count", "trip_distance", "PULocationID","DOLocationID", "fare_amount", "total_amount"]

trips_clean = (
    trips_raw
    .dropna(subset=REQUIRED_COLS)                              
    .dropDuplicates()                      
    .filter(F.col("trip_distance") > 0)
    .filter(F.col("fare_amount")   > 0)
    .filter(F.col("passenger_count") >= 1)
    .withColumn("tpep_pickup_datetime", F.to_timestamp(F.col("tpep_pickup_datetime")))
    .withColumn("tpep_dropoff_datetime", F.to_timestamp(F.col("tpep_dropoff_datetime")))
    .filter(F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
    .withColumn("PULocationID", F.col("PULocationID").cast("int"))
    .withColumn("DOLocationID", F.col("DOLocationID").cast("int"))
)

trips_featured = (
    trips_clean
    .withColumn("trip_duration_min",(F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0)
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn("pickup_dow",  F.dayofweek("tpep_pickup_datetime"))
    .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
)

def compute_avg_speed_mph(distance_miles, duration_min):
    if duration_min is None or duration_min <= 0:
        return None
    return float(distance_miles / (duration_min / 60.0))

avg_speed_udf = F.udf(compute_avg_speed_mph, DoubleType())

trips_featured.explain(True)
udf_start = time.perf_counter()

trips_with_speed = (
    trips_featured
    .withColumn(
        "avg_speed_mph",
        avg_speed_udf(F.col("trip_distance"), F.col("trip_duration_min"))
    )
    .filter(F.col("avg_speed_mph").isNull() | (F.col("avg_speed_mph") <= 150))
)

udf_row_count = trips_with_speed.count()
udf_elapsed   = time.perf_counter() - udf_start

zones_pu = (
    zones_raw
    .withColumnRenamed("LocationID",   "PULocationID")
    .withColumnRenamed("Borough",      "pickup_borough")
    .withColumnRenamed("Zone",         "pickup_zone")
    .withColumnRenamed("service_zone", "pickup_service_zone")
    .withColumn("PULocationID", F.col("PULocationID").cast("int"))
    .select("PULocationID", "pickup_borough", "pickup_zone", "pickup_service_zone")
)

zones_do = (
    zones_raw
    .withColumnRenamed("LocationID",   "DOLocationID")
    .withColumnRenamed("Borough",      "dropoff_borough")
    .withColumnRenamed("Zone",         "dropoff_zone")
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
trips_joined.write.mode("overwrite").partitionBy("pickup_date").parquet(f"{OUTPUT_DIR}/trips_enriched")
speed_by_hour.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/speed_by_hour")

time_elapsed = time.perf_counter() - time_start
mem_mb       = proc.memory_info().rss / (1024 ** 2)

print(f"The total time taken: {time_elapsed:.2f}s")
print(f"UDF only time taken: {udf_elapsed:.2f}s  "
      f"({(udf_elapsed / time_elapsed) * 100:.1f}% of total)")
print(f"Peak RSS memory: {mem_mb:.1f} MB")

spark.stop()
