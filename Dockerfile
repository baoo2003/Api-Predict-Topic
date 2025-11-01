FROM python:3.11-slim

# Giảm ghi .pyc, flush stdout, tắt cache pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Giới hạn thread/arena để giảm RAM nền
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_WAIT_POLICY=PASSIVE \
    MALLOC_ARENA_MAX=2

# (tuỳ chọn) tắt telemetry của HF
ENV HF_HUB_DISABLE_TELEMETRY=1

# Chỉ cài những thư viện hệ thống thực sự cần cho onnxruntime + HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài deps Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# 🧠 TẢI MODEL ONNX TRONG LÚC BUILD
# (giúp runtime không phải tải, giảm RAM và tránh spike)
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

# copy tokenizer nhẹ (bạn đã có trong repo)
COPY phobert-base/tokenizer ./phobert-base/tokenizer

# copy toàn bộ code (giữ sau khi đã có deps/model)
COPY . .

ENV MODEL_DIR=/app/models
EXPOSE 8000

# Chạy 1 worker + log nhẹ để tiết kiệm RAM
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}", "--workers", "1", "--log-level", "warning"]
