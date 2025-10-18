import os
import requests
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def download_and_quantize_onnx(
    onnx_url: str,
    fp32_path: str = "phobert-base/model.onnx",
    int8_path: str = "phobert-base/model_int8.onnx",
    timeout: int = 600,
    chunk_mb: int = 16
):
    """
    Tải ONNX từ Hugging Face -> lưu file -> quantize INT8.
    - onnx_url: link file .onnx trên Hugging Face (ví dụ: https://huggingface.co/.../model.onnx)
    - fp32_path: đường dẫn lưu model float32
    - int8_path: đường dẫn lưu model INT8
    """

    os.makedirs(os.path.dirname(fp32_path), exist_ok=True)
    tmp_path = fp32_path + ".part"
    chunk_size = chunk_mb * 1024 * 1024

    # 1) Download ONNX (streaming)
    print(f">> Downloading ONNX from: {onnx_url}")
    with requests.get(onnx_url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, fp32_path)
    print(f"✅ Saved float32 ONNX to: {fp32_path}")

    # 2) Kiểm tra hợp lệ (khuyến nghị)
    print(">> Checking float32 ONNX model...")
    onnx_model = onnx.load(fp32_path)
    onnx.checker.check_model(onnx_model)
    print("✅ Float32 ONNX is valid.")

    # 3) Quantize sang INT8 (dynamic quantization)
    print(f">> Quantizing to INT8 -> {int8_path} ...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,  # giảm size mạnh, RAM thấp hơn khi load
    )
    print(f"✅ Saved INT8 ONNX to: {int8_path}")

    # 4) Kiểm tra lại INT8 (khuyến nghị)
    print(">> Checking INT8 ONNX model...")
    onnx_int8 = onnx.load(int8_path)
    onnx.checker.check_model(onnx_int8)
    print("✅ INT8 ONNX is valid.")

    # 5) In size
    size_fp32 = os.path.getsize(fp32_path) / (1024 * 1024)
    size_int8 = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"📦 Size float32: {size_fp32:.2f} MB  →  INT8: {size_int8:.2f} MB")

if __name__ == "__main__":
    # Ví dụ dùng:
    download_and_quantize_onnx(
        onnx_url="https://huggingface.co/Qbao/phobert-onnx/resolve/main/model.onnx",
        fp32_path="phobert-base/model.onnx",
        int8_path="phobert-base/model_int8.onnx",
    )
