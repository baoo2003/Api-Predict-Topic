from io import BytesIO
import numpy as np
import os
import onnxruntime as ort
from transformers import AutoTokenizer
import requests

def load_phobert_onnx(
    tokenizer_dir: str = "phobert-base/tokenizer",
    onnx_url: str = "https://huggingface.co/Qbao/phobert-onnx/resolve/main/model.onnx",
    model_path: str = "phobert-base/model.onnx",
    allow_download: bool = True,
    timeout: int = 600,
    chunk_size: int = 16 * 1024 * 1024  # 16MB mỗi lần đọc để tiết kiệm RAM
):
    """
    Load PhoBERT ONNX:
      - Ưu tiên load từ file local (model_path)
      - Nếu chưa có và allow_download=True thì tải về (streaming, tiết kiệm RAM)
    Trả về: (tokenizer, onnxruntime.InferenceSession)
    """

    # 1️⃣ Load tokenizer (nhẹ, đã có trong repo)
    print(">> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

    # 2️⃣ Nếu đã có file model local -> load ngay
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f">> Found local ONNX model: {model_path}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(">> ONNX model loaded from local ✅")
        return tokenizer, session

    # 3️⃣ Nếu chưa có và không cho phép tải
    if not allow_download:
        raise FileNotFoundError(f"❌ ONNX model not found at {model_path} and downloading disabled.")

    # 4️⃣ Tải ONNX model từ Hugging Face (streaming)
    print(">> Local ONNX model not found. Downloading from Hugging Face (streaming)...")
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
        print(f">> Downloaded ONNX model to {model_path}")
    except Exception as e:
        print(f"❌ Failed to download ONNX model: {e}")
        raise

    # 5️⃣ Load model vừa tải
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(">> ONNX model loaded successfully ✅")

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
                  batch_size: int = 8, max_length: int = 256) -> str:
    text = (title or "") + " " + (content or "")
    vec = phobert_embed(session, tokenizer, [text], batch_size=batch_size, max_length=max_length)
    pred = clf.predict(vec)[0]
    return le.inverse_transform([pred])[0]