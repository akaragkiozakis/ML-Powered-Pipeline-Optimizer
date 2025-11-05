import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.preprocess_logs import log_message

try:
    df = pd.read_csv(r"C:\courses\ml_pipeline_optimizer\data\raw\processed\clean_logs.csv")
    log_message("Succeed to read clean_logs file")
except Exception as e:
    log_message(f"Failed to read clean logs csv file {e}")
    

# displaying columns of the dataset    
print(f"Columns in dataset:", df.columns.tolist())

# define features (X)
feature_cols = ['input_size_mb', 'executors', 'executor_memory_gb', 'shuffle_mb']
x = df[feature_cols]

# define target (Y)
target_col = 'runtime_sec'
y = df[target_col]

log_message(f"\nx shape: {x.shape}")
log_message(f"y shape: {y.shape}")


processed_df = pd.concat([x,y], axis=1)

try:
    output_path = r"C:\courses\ml_pipeline_optimizer\data\raw\processed\features.csv"
    processed_df.to_csv(output_path, index=False)
    log_message(f"Features dataset successfully saved at: {output_path}")
except Exception as e:
    log_message(f"Failed to save the Features dataset {e}")