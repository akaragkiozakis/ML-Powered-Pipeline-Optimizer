import os
import json
from pathlib import Path
import joblib
import pandas as pd
import itertools

MODEL_DIR = Path("data/raw/processed/models")
MODEL_PATH = MODEL_DIR / "runtime_predictor.pkl"

def log(msg: str):
    print(f"[optimizer] {msg}")

def load_trained_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path.resolve()}")
    model = joblib.load(model_path)
    log(f"Model loaded: {type(model).__name__}")
    return model

feature_cols = [
    "input_size_mb", "executors", "executor_memory_gb", "shuffle_mb",
    "shuffle_ratio", "mem_per_executor", "size_per_executor"
]

def create_candidate_configs():
    executors = [4, 8, 12, 16]
    executor_memory_gb = [4, 8, 16]
    shuffle_mb = [100, 300, 500]
    input_size_mb = [1000, 1500]

    combos = list(itertools.product(
        executors,
        executor_memory_gb,
        shuffle_mb,
        input_size_mb
    ))

    df = pd.DataFrame(combos, columns=[
        "executors", "executor_memory_gb", "shuffle_mb", "input_size_mb"
    ])

    df["shuffle_ratio"] = df["shuffle_mb"] / df["input_size_mb"]
    df["mem_per_executor"] = df["executor_memory_gb"] / df["executors"]
    df["size_per_executor"] = df["input_size_mb"] / df["executors"]

    log(f"✅ Created {len(df)} candidate configurations.")
    return df

def rank_configurations(model, df_candidates, feature_cols):
    df_candidates["pred_runtime_sec"] = model.predict(df_candidates[feature_cols])
    df_sorted = df_candidates.sort_values(by="pred_runtime_sec", ascending=True).reset_index(drop=True)
    best_config = df_sorted.iloc[0]

    print("\nTop 5 predicted configurations (fastest first):")
    print(df_sorted.head(5).to_string(index=False))

    print("\n🏆 Best configuration found:")
    print({
        "executors": int(best_config["executors"]),
        "executor_memory_gb": int(best_config["executor_memory_gb"]),
        "shuffle_mb": int(best_config["shuffle_mb"]),
        "input_size_mb": int(best_config["input_size_mb"]),
        "predicted_runtime_sec": round(float(best_config["pred_runtime_sec"]), 2)
    })

    return df_sorted


    
def rank_by_tradeoff(df_candidates, alpha=0.5):
    """
    Υπολογίζει συνδυαστικό score: runtime + α * (executors * executor_memory_gb).
    Όσο μικρότερο το score, τόσο καλύτερο το configuration.
    """
    df = df_candidates.copy()

    # calculate cost 
    df["cost_proxy"] = df["executors"] * df["executor_memory_gb"]

    
    df["tradeoff_score"] = df["pred_runtime_sec"] + alpha * df["cost_proxy"]

    
    df_sorted = df.sort_values(by="tradeoff_score", ascending=True).reset_index(drop=True)

    # best config
    best = df_sorted.iloc[0]

    print("\nTop 5 Cost-Efficient Configurations:")
    print(df_sorted.head(5)[[
        "executors", "executor_memory_gb", "input_size_mb", "shuffle_mb",
        "pred_runtime_sec", "cost_proxy", "tradeoff_score"
    ]].to_string(index=False))

    print("\n💰 Best Cost-Efficient Configuration:")
    print({
        "executors": int(best["executors"]),
        "executor_memory_gb": int(best["executor_memory_gb"]),
        "input_size_mb": int(best["input_size_mb"]),
        "pred_runtime_sec": round(float(best["pred_runtime_sec"]), 2),
        "tradeoff_score": round(float(best["tradeoff_score"]), 2)
    })

    return df_sorted




def main():
    # Load model
    model = load_trained_model(MODEL_PATH)

    # Quick test prediction
    sample = pd.DataFrame([{
        "input_size_mb": 1500,
        "executors": 8,
        "executor_memory_gb": 8,
        "shuffle_mb": 300,
        "shuffle_ratio": 300 / 1500,
        "mem_per_executor": 8 / 8,
        "size_per_executor": 1500 / 8
    }])
    pred = model.predict(sample)
    log(f"✅ Model test prediction: {pred[0]:.2f} sec")

    
    df_candidates = create_candidate_configs()
    df_candidates["pred_runtime_sec"] = model.predict(df_candidates[feature_cols])
    
    log("Ranking by pure runtime:")
    df_runtime = rank_configurations(model, df_candidates, feature_cols)
    
    log("\nRanking by cost-efficiency (alpha = 0.5):")
    df_tradeoff = rank_by_tradeoff(df_candidates, alpha=0.5)


if __name__ == "__main__":
    main()
