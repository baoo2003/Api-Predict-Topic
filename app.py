import joblib
import os, time, uvicorn
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from phobert_svm_pipeline import load_phobert_onnx, predict_topic
from proccessvitext import *

MODEL_DIR = os.getenv("MODEL_DIR", "models")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clf = joblib.load(MODEL_DIR + "/svm_cso_optimized.joblib")
    app.state.le = joblib.load(MODEL_DIR + "/label_encoder.joblib")
    app.state.tokenizer, app.state.model = load_phobert_onnx()
    yield

app = FastAPI(title="PhoBERT+SVM Topic API", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class InText(BaseModel):
    title: Optional[str] = ""
    content: str

class Out(BaseModel):
    label: str
    latency_ms: int

@app.get("/health")
def health():
    le = getattr(app.state, "le", None)
    return {"status": "ok", "model_dir": MODEL_DIR,
            "num_classes": len(getattr(le, "classes_", [])) if le else 0}

@app.post("/predict")
def predict(p: InText):
    t = time.time()
    print("Received payload:", p)

    p.title = preprocess_topic(p.title)
    p.content = preprocess_text(p.content, set())

    result = predict_topic(
        app.state.model, app.state.tokenizer,
        p.title or "", p.content or "",
        app.state.clf, app.state.le,
        batch_size=4, max_length=128
    )

    return {
        "best_label": result["best_label"],
        "confidence": round(result["confidence"] * 100, 2),
        "all_predictions": result["all_predictions"],
        "latency_ms": int((time.time() - t) * 1000)
    }
