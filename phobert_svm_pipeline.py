from io import BytesIO
import numpy as np
import os
import onnxruntime as ort
from transformers import AutoTokenizer
import requests

def load_phobert_onnx(
    tokenizer_dir: str = "phobert-base/tokenizer",
    # Đường dẫn tới model INT8 bạn đã upload lên Hugging Face
    onnx_url: str = "https://huggingface.co/Qbao/phobert-onnx/resolve/main/model_int8.onnx",
    model_path: str = "phobert-base/model_int8.onnx",
    allow_download: bool = True,
    timeout: int = 600,
    chunk_size: int = 16 * 1024 * 1024  # 16 MB mỗi chunk để tiết kiệm RAM
):
    """
    Load PhoBERT ONNX INT8:
      - Ưu tiên load từ local (model_int8.onnx)
      - Nếu chưa có và allow_download=True thì tải về (streaming)
    Trả về: (tokenizer, onnxruntime.InferenceSession)
    """

    print(">> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

    # Nếu model INT8 đã có sẵn trong container -> load luôn
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f">> Found local INT8 ONNX model: {model_path}")
        so = ort.SessionOptions()
        # tối ưu RAM:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # giảm busy-wait
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")

        providers = [("CPUExecutionProvider", {
            "arena_extend_strategy": "kSameAsRequested"  # tránh bành trướng arena
        })]

        session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        print(">> Loaded INT8 ONNX model from local ✅")
        return tokenizer, session

    if not allow_download:
        raise FileNotFoundError(f"❌ Model not found locally and downloading disabled: {model_path}")

    # Nếu chưa có -> tải từ Hugging Face
    print(">> Downloading PhoBERT INT8 ONNX from Hugging Face (streaming)...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp_path = model_path + ".part"

    try:
        with requests.get(onnx_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, model_path)
        print(f">> Downloaded PhoBERT INT8 ONNX to {model_path}")
    except Exception as e:
        print(f"❌ Failed to download ONNX model: {e}")
        raise

    # Load session sau khi tải
    so = ort.SessionOptions()
    # tối ưu RAM:
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # giảm busy-wait
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")

    providers = [("CPUExecutionProvider", {
        "arena_extend_strategy": "kSameAsRequested"  # tránh bành trướng arena
    })]

    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    print(">> PhoBERT INT8 model loaded successfully ✅")

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

        ort_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        ort_outputs = session.run(["last_hidden_state"], ort_inputs)
        pooled = _mean_pool(ort_outputs[0], encoded["attention_mask"])

        if l2norm:
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norm, a_min=1e-9, a_max=None)

        embs.append(pooled)

    return np.vstack(embs)

def predict_topic(session, tokenizer, title: str, content: str, clf, le,
                  batch_size: int = 8, max_length: int = 256):
    text = (title or "") + " " + (content or "")
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
