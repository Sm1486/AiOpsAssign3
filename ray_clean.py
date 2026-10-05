import time
import os
import psutil
import pandas as pd
import numpy as np

import ray
import ray.data

os.environ["RAY_AUTH_MODE"] = "disabled"
TRIP_DATA_DIR = "TrafficData/Subsample"
ZONE_CSV = "TrafficData/taxi_zone_lookup.csv"
OUTPUT_DIR = "output/ray_output"

if ray.is_initialized():
    ray.shutdown()

ray.init(num_cpus=2, object_store_memory= 1024**3, ignore_reinit_error=True)
ctx = ray.data.DataContext.get_current()
ctx.target_max_block_size = 64 * 1024**2

time_start = time.perf_counter()
proc = psutil.Process(os.getpid())

REQUIRED_COLS = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime","passenger_count", "trip_distance", "PULocationID","DOLocationID", "fare_amount", "total_amount"]

trips_ds = ray.data.read_parquet(TRIP_DATA_DIR, columns=REQUIRED_COLS)
zones_pd = pd.read_csv(ZONE_CSV)


def cleanse_batch(batch):
    batch = batch.dropna(subset=REQUIRED_COLS)
    batch = batch.drop_duplicates()
    batch = batch[batch["trip_distance"] > 0]
    batch = batch[batch["fare_amount"]   > 0]
    batch = batch[batch["passenger_count"] >= 1]
    batch["tpep_pickup_datetime"]  = pd.to_datetime(batch["tpep_pickup_datetime"],  errors="coerce")
    batch["tpep_dropoff_datetime"] = pd.to_datetime(batch["tpep_dropoff_datetime"], errors="coerce")
    batch = batch[batch["tpep_dropoff_datetime"] > batch["tpep_pickup_datetime"]]
    batch["PULocationID"] = batch["PULocationID"].astype(int)
    batch["DOLocationID"] = batch["DOLocationID"].astype(int)
    return batch

trips_clean = trips_ds.map_batches(cleanse_batch, batch_format="pandas", batch_size=10_000)
udf_start = time.perf_counter()

def add_features_and_speed(batch):
    batch["trip_duration_min"] = (batch["tpep_dropoff_datetime"] - batch["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    batch["pickup_hour"] = batch["tpep_pickup_datetime"].dt.hour
    batch["pickup_dow"]  = batch["tpep_pickup_datetime"].dt.dayofweek
    batch["pickup_date"] = batch["tpep_pickup_datetime"].dt.date.astype(str)
    batch["avg_speed_mph"] = (batch["trip_distance"] / (batch["trip_duration_min"] / 60.0))
    batch = batch[batch["avg_speed_mph"].isna() | (batch["avg_speed_mph"] <= 160)]
    return batch

trips_with_speed = trips_clean.map_batches(add_features_and_speed, batch_format="pandas", batch_size=10_000)

udf_row_count = trips_with_speed.count()
udf_elapsed   = time.perf_counter() - udf_start
zones_ref = ray.put(zones_pd)

def join_zones(batch: pd.DataFrame):
    zones = ray.get(zones_ref)
    zones_pu = zones.rename(columns={"LocationID":"PULocationID","Borough":"pickup_borough","Zone":"pickup_zone","service_zone":"pickup_service_zone",})[["PULocationID", "pickup_borough", "pickup_zone", "pickup_service_zone"]]
    zones_do = zones.rename(columns={"LocationID":"DOLocationID","Borough":"dropoff_borough","Zone":"dropoff_zone","service_zone":"dropoff_service_zone",})[["DOLocationID","dropoff_borough", "dropoff_zone", "dropoff_service_zone"]]

    batch = batch.merge(zones_pu, on="PULocationID", how="left")
    batch = batch.merge(zones_do, on="DOLocationID", how="left")
    return batch

trips_joined = trips_with_speed.map_batches(join_zones,batch_format="pandas",batch_size=10_000)

@ray.remote
class BenchmarkTracker:
    def __init__(self):
        self.results = []

    def record(self, hour, mean_speed, count, mean_dist, mean_dur):
        self.results.append({
            "pickup_hour": hour,
            "mean_speed_mph": mean_speed,
            "trip_count": count,
            "mean_distance_miles": mean_dist,
            "mean_duration_min": mean_dur,
        })

    def get_results(self):
        return self.results

tracker = BenchmarkTracker.remote()

@ray.remote
def compute_hour_stats(hour, df_all, tracker_ref):
    subset = df_all[df_all["pickup_hour"] == hour]
    if len(subset) == 0:
        return {"pickup_hour": hour, "trip_count": 0, "mean_speed_mph": None, "mean_distance_miles": None,"mean_duration_min": None}
    stats = {
        "pickup_hour": hour,
        "mean_speed_mph": float(subset["avg_speed_mph"].mean()),
        "trip_count": int(len(subset)),
        "mean_distance_miles":float(subset["trip_distance"].mean()),
        "mean_duration_min": float(subset["trip_duration_min"].mean()),
    }
    ray.get(tracker_ref.record.remote(stats["pickup_hour"], stats["mean_speed_mph"], stats["trip_count"], stats["mean_distance_miles"], stats["mean_duration_min"]))
    return stats

os.makedirs(OUTPUT_DIR, exist_ok=True)

trips_joined.write_parquet(f"{OUTPUT_DIR}/trips_enriched")
stats_pd = ray.data.read_parquet(
    f"{OUTPUT_DIR}/trips_enriched",
    columns=["pickup_hour", "avg_speed_mph", "trip_distance", "trip_duration_min"],
).to_pandas()
trips_ref = ray.put(stats_pd)

futures = [compute_hour_stats.remote(h, trips_ref, tracker) for h in range(24)]

not_ready = futures
speed_by_hour_rows = []
while not_ready:
    ready, not_ready = ray.wait(not_ready, num_returns=1)
    for r in ready:
        row = ray.get(r)
        speed_by_hour_rows.append(row)

actor_results = ray.get(tracker.get_results.remote())

speed_by_hour = (pd.DataFrame(speed_by_hour_rows).sort_values("pickup_hour").reset_index(drop=True))
speed_by_hour.to_parquet(f"{OUTPUT_DIR}/speed_by_hour.parquet", index=False, engine="pyarrow")

time_elapsed = time.perf_counter() - time_start
mem_mb = proc.memory_info().rss / (1024 ** 2)

print(f"The total time taken: {time_elapsed:.2f}s")
print(f"UDF only time: {udf_elapsed:.2f}s  "
      f"({(udf_elapsed / time_elapsed) * 100:.1f}% of total)")
print(f"Peak RSS memory: {mem_mb:.1f} MB")

ray.shutdown()