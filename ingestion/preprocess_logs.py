import pandas as pd
from datetime import datetime, time 
import numpy as np

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


try:
    csvfile = pd.read_csv(r"C:\courses\ml_pipeline_optimizer\data\raw\spark_job_logs.csv")
    log_message(f"Csv file found: {csvfile}")
except Exception as e:
    log_message(f"Csv file not found: {e}")

# check for null values
log_message("Checking for NULL values")
for columns in csvfile.columns:
    null_count = csvfile[columns].isnull().sum()
    
    if null_count > 0:
        print(f" {columns} -> {null_count} nulls")
    else:
        print(f" {columns} -> No nulls")
        
        

# check for outliers in runtime and cost columns
thresholds = {
    'runtime_sec': 100,
    'cost_usd': 10
}


for column, thresholds in thresholds.items():
    outliers = csvfile[csvfile[column] > thresholds]
    log_message(f"Values over {thresholds}: {len(outliers)}")
    
    
# check for job_id duplicates
duplicates = csvfile[csvfile.duplicated(subset=['job_id'], keep=False)]
log_message(f"Duplicate IDs found: {len(duplicates)}")


# check for negative values
for col in csvfile.select_dtypes(include=[np.number]).columns:
    neg_count = (csvfile[col] < 0).sum()
    if neg_count > 0:
        log_message(f"{col} has {neg_count} negative values")
    else:
        log_message(f"{col}: No negative values")
        
try:        
    csvfile.to_csv(r"C:\courses\ml_pipeline_optimizer\data\raw\processed\clean_logs.csv", index=False)
    log_message("Succeed to save the clean csv file")
except Exception as e:
    log_message(f"Failed to save the clean csv file {e}")