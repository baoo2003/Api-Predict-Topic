from io import BytesIO
import numpy as np
import os
import onnxruntime as ort
from transformers import AutoTokenizer
import requests

def load_phobert_onnx(
    tokenizer_dir: str = "phobert-base/tokenizer",
    model_path: str = "phobert-base/model.onnx",
):
    """
    Load PhoBERT ONNX từ local (đã có sẵn trong container).
    Trả về: (tokenizer, onnxruntime.InferenceSession)
    """

    print(">> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

    if not (os.path.exists(model_path) and os.path.getsize(model_path) > 0):
        raise FileNotFoundError(f"❌ ONNX model not found at: {model_path}")

    print(f">> Loading ONNX model from: {model_path}")
    # dùng cấu hình mặc định, không tối ưu RAM gì cả
    session = ort.InferenceSession(model_path)

    print(">> PhoBERT ONNX loaded ✅")
    return tokenizer, session

def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., np.newaxis].astype(np.float32)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(axis=1)            # [B,H]
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None) # [B,1]
    return summed / counts

def phobert_embed(session, tokenizer, texts, max_length: int = 256,
                  batch_size: int = 16, l2norm: bool = True) -> np.ndarray:
    """
    Trả về embedding PhoBERT (numpy) sử dụng ONNX.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=max_length)

        input_ids = encoded["input_ids"].astype("int64", copy=False)
        attention_mask = encoded["attention_mask"].astype("int64", copy=False)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        ort_outputs = session.run(["last_hidden_state"], ort_inputs)
        pooled = _mean_pool(ort_outputs[0], encoded["attention_mask"])

        if l2norm:
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norm, a_min=1e-9, a_max=None)

        embs.append(pooled)

    return np.vstack(embs)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def predict_topic(session, tokenizer, title: str, content: str, clf, le,
                  batch_size: int = 8, max_length: int = 128):
    # 1) build input y như lúc train
    text = (title or "") + " </s> " + (content or "")
    vec = phobert_embed(session, tokenizer, [text],
                        batch_size=batch_size, max_length=max_length)

    # 2) Với model đã calibrate: phải có predict_proba
    if not hasattr(clf, "predict_proba"):
        raise ValueError(
            "Model không có predict_proba(). Hãy load model CalibratedClassifierCV "
            "(vd: linear_svc_phobert_calibrated.joblib)."
        )

    # 3) Lấy xác suất chuẩn (độ tin cậy)
    proba = clf.predict_proba(vec)          # shape: (1, K)
    proba = np.atleast_2d(proba)[0]         # -> (K,)

    # 4) classes_ mapping (CalibratedClassifierCV có classes_)
    classes = getattr(clf, "classes_", None)
    if classes is None:
        raise ValueError("Không tìm thấy classes_ trong model calibrate.")
    classes = np.asarray(classes, dtype=int)

    # 5) Chọn best theo proba
    best_pos = int(np.argmax(proba))
    best_class = int(classes[best_pos])
    best_label = le.inverse_transform([best_class])[0]
    best_conf = float(proba[best_pos])      # ✅ confidence đúng

    # 6) gap giữa top1-top2 (từ proba) để xem top1 có vượt trội không
    sorted_p = np.sort(proba)
    gap = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) >= 2 else 0.0

    # 7) (Tuỳ chọn) vẫn lấy decision_function nếu có để debug margin
    margin = None
    if hasattr(clf, "decision_function"):
        raw = clf.decision_function(vec)
        raw = np.atleast_2d(raw)[0]
        margin = float(raw[best_pos])

    # 8) all predictions (theo xác suất thật)
    all_preds = [
        {
            "label": le.inverse_transform([int(c)])[0],
            "confidence": float(proba[pos]),   # ✅ proba thật
            "class_id": int(c),
        }
        for pos, c in enumerate(classes)
    ]
    all_preds.sort(key=lambda x: x["confidence"], reverse=True)

    out = {
        "best_label": best_label,
        "confidence": best_conf,
        "gap": gap,
        "all_predictions": all_preds
    }
    if margin is not None:
        out["margin"] = margin

    return out
