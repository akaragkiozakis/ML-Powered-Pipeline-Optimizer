import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.preprocess_logs import log_message

# read the dataset
df = pd.read_csv(r"C:\courses\ml_pipeline_optimizer\data\raw\processed\features.csv")

log_message(f"Columns in dataset: {df.columns.tolist()}")
log_message(f"Shape: {df.shape}")


df["shuffle_ratio"] = df["shuffle_mb"] / df["input_size_mb"]
df["mem_per_executor"] = df["executor_memory_gb"] / df["executors"]
df["size_per_executor"] = df["input_size_mb"] / df["executors"]


# define features (X) and target (Y)
x = df[[
    "input_size_mb", "executors", "executor_memory_gb", "shuffle_mb",
    "shuffle_ratio", "mem_per_executor", "size_per_executor"
]]
y = df["runtime_sec"]


print("\nFeature sample:")
print(x.head())

print("\nTarget sample:")
print(y.head())


# separate to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# display the shapes for confirmation 
print(f"\nX_train shape: {x_train.shape}")
print(f"X_test shape:  {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


# model training (Liner Regression)
model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
model.fit(x_train, y_train)


# prediction of test set
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"R²   = {r2:.3f}")


# save then model 
joblib.dump(model, r"data\raw\processed\models\runtime_predictor.pkl")
log_message(r"Model saved successfully at data\raw\processed\models")
