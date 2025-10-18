FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# ================================
# 🧠 TẢI MODEL ONNX TRONG LÚC BUILD
# ================================
RUN mkdir -p phobert-base/tokenizer && \
    echo ">> Downloading PhoBERT ONNX from Hugging Face..." && \
    python - <<'PYCODE'
import requests, os
url = "https://huggingface.co/Qbao/phobert-onnx/resolve/main/model.onnx"
path = "phobert-base/model.onnx"
os.makedirs(os.path.dirname(path), exist_ok=True)
with requests.get(url, stream=True, timeout=600) as r:
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=16*1024*1024):
            if chunk:
                f.write(chunk)
print(">> Model ONNX saved to", path)
PYCODE

# copy tokenizer nhẹ (đã có trong repo)
COPY phobert-base/tokenizer ./phobert-base/tokenizer

# copy code chính
COPY . .

ENV MODEL_DIR=/app/models
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
