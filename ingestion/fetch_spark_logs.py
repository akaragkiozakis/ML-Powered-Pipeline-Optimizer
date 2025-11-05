import os
import pandas as pd
import numpy as np


# define the path 
RAW_DATA_PATH = os.path.join("data", "raw")
os.makedirs(RAW_DATA_PATH, exist_ok=True)
OUTPUT_FILE = os.path.join(RAW_DATA_PATH, "spark_job_logs.csv")


# set the number of jobs
NUM_JOBS = 500


# generate random values
np.random.seed(42)  

input_size_mb = np.random.randint(100, 3000, NUM_JOBS) 
executors = np.random.choice([4, 8, 12, 16], NUM_JOBS)
executor_memory_gb = np.random.choice([4, 6, 8, 10, 12], NUM_JOBS)
shuffle_mb = (input_size_mb * np.random.uniform(0.1, 0.35, NUM_JOBS)).round(2)



# calculate runtime and cost $
# runtime μειώνεται με executors & memory, αυξάνεται με input_size
runtime_sec = (
    input_size_mb / (executors * np.random.uniform(3.5, 6.0, NUM_JOBS)) *
    (12 / executor_memory_gb)
).round(2)

# κόστος αυξάνεται με executors, memory, runtime
cost_usd = (
    (executors * executor_memory_gb * 0.015) +
    (runtime_sec / 900)
).round(2)


# create dataframe
df = pd.DataFrame({
    "job_id": range(1, NUM_JOBS + 1),
    "input_size_mb": input_size_mb,
    "executors": executors,
    "executor_memory_gb": executor_memory_gb,
    "shuffle_mb": shuffle_mb,
    "runtime_sec": runtime_sec,
    "cost_usd": cost_usd
})



# save to csv
df.to_csv(OUTPUT_FILE, index=False)
print(f"Created {NUM_JOBS} Spark job logs at: {OUTPUT_FILE}")
print(df.head(10))
