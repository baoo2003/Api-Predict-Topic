import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
import os

# =============================
# 1) PhoBERT: Tự động tải từ Hugging Face
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_phobert(model_name: str = "vinai/phobert-base"):
    """
    Luôn tải PhoBERT từ Hugging Face về cache (không cần local_dir).
    """
    print(">> Loading PhoBERT from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model

_tokenizer, _model = load_phobert()

# =============================
# 2) Embedding PhoBERT
# =============================
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B,1]
    return summed / counts

def phobert_embed(texts, max_length: int = 256, batch_size: int = 16, l2norm: bool = True) -> np.ndarray:
    """
    Trả về embedding PhoBERT (numpy) cho list[str].
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    embs = []
    _model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = _tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(device)

            outputs = _model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if l2norm:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            embs.append(pooled.cpu().numpy())
    return np.vstack(embs)

# =============================
# 3) Predict Topic
# =============================
def predict_topic(title: str, content: str, clf, le,
                  batch_size: int = 8, max_length: int = 256) -> str:
    text = (title or "") + " " + (content or "")
    vec = phobert_embed([text], batch_size=batch_size, max_length=max_length)
    pred = clf.predict(vec)[0]
    return le.inverse_transform([pred])[0]
