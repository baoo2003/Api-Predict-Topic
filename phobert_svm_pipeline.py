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

    # 2) lấy decision scores (LinearSVC / Pipeline đều có decision_function)
    if not hasattr(clf, "decision_function"):
        raise ValueError("Model không có decision_function. Bạn đang load đúng LinearSVC/Pipeline chưa?")

    raw = clf.decision_function(vec)  # shape: (1, K) hoặc (1,) nếu binary
    raw = np.atleast_2d(raw)[0]        # -> shape (K,)

    # 3) classes_ (an toàn cho Pipeline vs estimator)
    classes = getattr(clf, "classes_", None)
    if classes is None and hasattr(clf, "named_steps"):
        classes = clf.named_steps["svm"].classes_
    if classes is None:
        raise ValueError("Không tìm thấy classes_ trong clf.")

    classes = np.asarray(classes, dtype=int)

    # 4) pseudo-prob để HIỂN THỊ (không phải xác suất thật)
    probs = _softmax(raw)

    # 5) chọn best theo raw hoặc probs đều như nhau
    best_pos = int(np.argmax(raw))
    best_class = int(classes[best_pos])          # label id thật (0..12)
    best_label = le.inverse_transform([best_class])[0]
    best_conf = float(probs[best_pos])

    # 6) margin/gap để debug (thường hữu ích hơn conf)
    sorted_raw = np.sort(raw)
    gap = float(sorted_raw[-1] - sorted_raw[-2]) if len(sorted_raw) >= 2 else 0.0
    margin = float(raw[best_pos])

    # 7) all predictions (map theo classes_)
    all_preds = [
        {
            "label": le.inverse_transform([int(c)])[0],
            "confidence": float(probs[pos]),
            "margin": float(raw[pos]),
            "class_id": int(c),
        }
        for pos, c in enumerate(classes)
    ]
    all_preds.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "best_label": best_label,
        "confidence": best_conf,
        "margin": margin,
        "gap": gap,
        "all_predictions": all_preds
    }
