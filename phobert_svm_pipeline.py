import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_phobert_onnx(local_tokenizer_dir="phobert-base/tokenizer",
                      onnx_model_path="phobert-base/model.onnx"):
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_dir, use_fast=False)
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
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
