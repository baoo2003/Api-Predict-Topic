import joblib
import os, time, uvicorn
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from phobert_svm_pipeline import load_phobert_onnx, predict_topic
from proccessvitext import *
import numpy as np

LABEL_NAMES = [
    "Công nghệ",     # 0
    "Du lịch",       # 1
    "Giáo dục",      # 2
    "Giải trí",      # 3
    "Khoa học",      # 4
    "Kinh doanh",    # 5
    "Pháp luật",     # 6
    "Sức khỏe",      # 7
    "Thế giới",      # 8
    "Thể thao",      # 9
    "Thời sự",       # 10
    "Xe",            # 11
    "Đời sống",      # 12
]

class SimpleLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes, dtype=object)

    def inverse_transform(self, idx_list):
        # idx_list: [0,1,2] -> ["Công nghệ", ...]
        return self.classes_[np.array(idx_list, dtype=int)]


MODEL_DIR = os.getenv("MODEL_DIR", "models")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clf = joblib.load(MODEL_DIR + "/linear_svc_phobert.joblib")
    app.state.le = SimpleLabelEncoder(LABEL_NAMES)
    app.state.tokenizer, app.state.model = load_phobert_onnx()
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
    return {"status": "ok", "model_dir": MODEL_DIR,
            "num_classes": len(getattr(le, "classes_", [])) if le else 0}

@app.post("/predict")
def predict(p: InText):
    t = time.time()
    print("Received payload:", p)

    p.title = preprocess_text(p.title)
    p.content = preprocess_text(p.content)

    result = predict_topic(
        app.state.model, app.state.tokenizer,
        p.title or "", p.content or "",
        app.state.clf, app.state.le
    )

    return {
        "best_label": result["best_label"],
        "confidence": round(result["confidence"] * 100, 2),
        "all_predictions": result["all_predictions"],
        "latency_ms": int((time.time() - t) * 1000)
    }
