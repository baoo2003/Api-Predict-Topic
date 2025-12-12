FROM python:3.11-slim

# Cơ bản: không ghi .pyc, log ra stdout, tắt cache pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Cài thư viện hệ thống cần cho onnxruntime + HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài deps Python
# (đảm bảo trong requirements.txt có: onnxruntime, transformers, requests, uvicorn, fastapi/... tuỳ bạn dùng gì)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# 🧠 TẢI MODEL ONNX TRONG LÚC BUILD
# (runtime KHÔNG cần tải lại nữa)
# ================================
RUN python - <<'PYCODE'
import os, requests

os.makedirs("phobert-base", exist_ok=True)

url = "https://huggingface.co/Qbao/phobert-onnx/resolve/main/model.onnx"
path = "phobert-base/model.onnx")

print(f">> Downloading ONNX model from {url} ...")

with requests.get(url, stream=True, timeout=600) as r:
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=16*1024*1024):
            if chunk: f.write(chunk)

print(f"✅ PhoBERT ONNX downloaded successfully to {path}")
PYCODE

# copy tokenizer nhẹ (bạn đã có trong repo)
COPY phobert-base/tokenizer ./phobert-base/tokenizer

# copy toàn bộ code ứng dụng
COPY . .

ENV MODEL_DIR=/app/models
EXPOSE 8000

# Chạy 1 worker + log nhẹ để tiết kiệm RAM
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level warning"]
