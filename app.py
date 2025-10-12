import joblib
import os, time, torch, uvicorn
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from phobert_svm_pipeline import predict_topic
from proccessvitext import *

MODEL_DIR = os.getenv("MODEL_DIR", "models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clf = joblib.load(MODEL_DIR + "/svm_cso_optimized.joblib")
    app.state.le = joblib.load(MODEL_DIR + "/label_encoder.joblib")
    yield

app = FastAPI(title="PhoBERT+SVM Topic API", lifespan=lifespan)
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
    return {"status": "ok", "device": str(device), "model_dir": MODEL_DIR,
            "num_classes": len(getattr(le, "classes_", [])) if le else 0}

@app.post("/predict", response_model=Out)
def predict(p: InText):
    t = time.time()
    print("Received payload:", p)
    print("Original title:", p.title)
    print("Original content:", p.content)
    p.title = preprocess_topic(p.title)
    p.content = preprocess_text(p.content, set())
    print("Preprocessed title:", p.title)
    print("Preprocessed content:", p.content)
    lbl = predict_topic(p.title or "", p.content or "", app.state.clf, app.state.le)
    return {"label": lbl, "latency_ms": int((time.time()-t)*1000)}

@app.post("/predict-batch")
def batch(payload: List[InText]):
    t = time.time()
    clf, le = app.state.clf, app.state.le
    res = [{"label": predict_topic(p.title or "", p.content or "", clf, le)} for p in payload]
    return {"results": res, "latency_ms": int((time.time()-t)*1000)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=True)
