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
RUN python - <<'PYCODE'
import os, requests
os.makedirs("phobert-base", exist_ok=True)
url = "https://huggingface.co/Qbao/phobert-onnx/resolve/main/model_int8.onnx"
path = "phobert-base/model_int8.onnx"
print(f">> Downloading INT8 model from {url} ...")
with requests.get(url, stream=True, timeout=600) as r:
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=16*1024*1024):
            if chunk: f.write(chunk)
print("✅ PhoBERT INT8 ONNX downloaded successfully.")
PYCODE

# copy tokenizer nhẹ (đã có trong repo)
COPY phobert-base/tokenizer ./phobert-base/tokenizer

# copy code chính
COPY . .

ENV MODEL_DIR=/app/models
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
