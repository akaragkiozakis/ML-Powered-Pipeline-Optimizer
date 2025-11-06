from fastapi import FastAPI , HTTPException 
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path 


# Metrics
REQUEST_COUNT = Counter("optimizer_requests_total", "Total requests to the optimizer API", ["endpoint"])
REQUEST_LATENCY = Histogram("optimizer_request_latency_seconds", "Latency of API requests", ["endpoint"])


# initialize app
app = FastAPI(
    title="ML Pipeline Optimizer API",
    description="Predicts pipeline runtime and suggests optimal configuration",
    version="1.0.0"
)


# load model at startup 
MODEL_PATH = Path("data/raw/processed/models/runtime_predictor.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[API] Model loaded successfully: {type(model).__name__}")
except Exception as e:
    print(f"[API] Failed to load model: {e}")
    model = None
    
# define input schema 
class ConfigInput(BaseModel):
    input_size_mb: float
    executors: int
    executor_memory_gb: float
    shuffle_mb: float
    

# define endpoints
@app.get("/")
def root():
    return {
        "message": "🚀 ML Pipeline Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "optimize": "/optimize",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)  # No Content - eliminates the 404 error

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None 
    }
    
    
@app.get("/predict")
def predict_runtime(config: ConfigInput):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    data = pd.DataFrame([{
        "input_size_mb": config.input_size_mb,
        "executors": config.executors,
        "executor_memory_gb": config.executor_memory_gb,
        "shuffle_mb": config.shuffle_mb,
        "shuffle_ratio": config.shuffle_mb / config.input_size_mb,
        "mem_per_executor": config.executor_memory_gb / config.executors,
        "size_per_executor": config.input_size_mb / config.executors
    }])
    
    pred = model.predict(data)[0]
    
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time()- start_time)
    return {"predicted_runtime_sec": round(float(pred), 2)}


from optimizer.suggest_config import create_candidate_configs, rank_configurations, rank_by_tradeoff

@app.get("/optimize")
def optimize_pipeline(mode: str = "runtime", alpha: float = 0.5):
    """
    Suggest best configuration based on trained model.
    Mode options:
    - runtime: fastest predicted configuration
    - cost: cost-efficient trade-off configuration
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # create candidate config
    df_candidates = create_candidate_configs()
    df_candidates["pred_runtime_sec"] = model.predict(df_candidates[[
        "input_size_mb", "executors", "executor_memory_gb", "shuffle_mb",
        "shuffle_ratio", "mem_per_executor", "size_per_executor"
    ]])

    # select mode
    if mode == "runtime":
        df_ranked = rank_configurations(model, df_candidates, [
            "input_size_mb", "executors", "executor_memory_gb", "shuffle_mb",
            "shuffle_ratio", "mem_per_executor", "size_per_executor"
        ])
    elif mode == "cost":
        df_ranked = rank_by_tradeoff(df_candidates, alpha=alpha)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'runtime' or 'cost'.")

    # return best config
    best = df_ranked.iloc[0].to_dict()

    return {
        "mode": mode,
        "best_configuration": {
            "executors": int(best["executors"]),
            "executor_memory_gb": int(best["executor_memory_gb"]),
            "shuffle_mb": int(best["shuffle_mb"]),
            "input_size_mb": int(best["input_size_mb"]),
            "predicted_runtime_sec": round(float(best["pred_runtime_sec"]), 2)
        },
        "note": "Results generated using ML runtime optimizer"
    }


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# run manually (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)