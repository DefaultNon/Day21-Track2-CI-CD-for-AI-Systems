from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os
import sys

app = FastAPI()

# Lay ten bucket tu bien moi truong (dong bo voi mlops.yml)
GCS_BUCKET = os.getenv("CLOUD_BUCKET")
MODEL_DIR = os.path.expanduser("~/models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

class PredictRequest(BaseModel):
    features: list

# Bien toan cuc de luu model sau khi load
model = None

def load_model():
    """Tai model tu GCS va nạp vao bo nho."""
    global model
    try:
        if not GCS_BUCKET:
            print("ERROR: CLOUD_BUCKET environment variable is not set.")
            return

        print(f"Attempting to download model from bucket: {GCS_BUCKET}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("models/latest/model.pkl")
        
        if not blob.exists():
            print(f"ERROR: Model blob 'models/latest/model.pkl' not found in bucket '{GCS_BUCKET}'")
            return

        blob.download_to_filename(MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")
        
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully into memory.")
    except Exception as e:
        print(f"CRITICAL ERROR during model loading: {str(e)}")

# Goi nạp model ngay khi khoi dong server
load_model()

@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """Endpoint suy luan."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features")

    try:
        pred = model.predict([req.features])[0]
        labels = {0: "thap", 1: "trung_binh", 2: "cao"}
        return {"prediction": int(pred), "label": labels.get(int(pred), "unknown")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
