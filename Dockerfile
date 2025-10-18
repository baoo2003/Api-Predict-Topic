# ====== Stage 1: Runtime nhỏ gọn ======
FROM python:3.11-slim AS runtime

# Giảm kích thước và lỗi biên dịch
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Một số lib động thời chạy cho onnxruntime/transformers
# libgomp1 thường cần cho onnxruntime; ca-certificates để requests tải model ONNX qua HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đúng dependency từ requirements.txt của bạn (đã bỏ torch nên nhẹ)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Sao chép MỖI file cần thiết, tránh .venv, __pycache__,…
# Code
COPY app.py .
COPY phobert_svm_pipeline.py .
COPY proccessvitext.py .

# Model/label cục bộ
# Giữ đúng cấu trúc "models" như local để app.py load qua MODEL_DIR
COPY models ./models

# Chỉ copy tokenizer (không copy model.onnx nặng; app sẽ tải ONNX từ Hugging Face lúc chạy)
COPY phobert-base/tokenizer ./phobert-base/tokenizer

# Tuỳ chọn: copy .env nếu bạn thực sự cần biến mặc định trong container (không bắt buộc)
# COPY .env .

# Thiết lập biến cho app
ENV MODEL_DIR=/app/models

# Railway sẽ truyền PORT; nếu không có thì dùng 8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
