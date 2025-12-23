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

def predict_topic(session, tokenizer, title: str, content: str, clf, le,
                  batch_size: int = 8, max_length: int = 128):
    text = (title or "") + " </s> " + (content or "")
    vec = phobert_embed(session, tokenizer, [text], batch_size=batch_size, max_length=max_length)

    # Dự đoán xác suất (nếu model có hỗ trợ)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(vec)[0]
    else:
        # Nếu model không có predict_proba, chuyển sang decision_function và chuẩn hóa
        if hasattr(clf, "decision_function"):
            raw = clf.decision_function(vec)[0]
            # Chuẩn hóa về 0-1 bằng softmax
            exp_raw = np.exp(raw - np.max(raw))
            probs = exp_raw / np.sum(exp_raw)
        else:
            probs = np.ones(len(le.classes_)) / len(le.classes_)

    # Lấy nhãn tốt nhất
    best_idx = int(np.argmax(probs))
    best_label = le.inverse_transform([best_idx])[0]
    best_conf = float(probs[best_idx])

    # Danh sách dự đoán tất cả nhãn
    all_preds = [
        {"label": le.inverse_transform([i])[0], "confidence": float(probs[i])}
        for i in range(len(probs))
    ]

    return {
        "best_label": best_label,
        "confidence": best_conf,
        "all_predictions": sorted(all_preds, key=lambda x: x["confidence"], reverse=True)
    }
