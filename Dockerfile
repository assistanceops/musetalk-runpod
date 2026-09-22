FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    TORCH_HOME=/opt/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg git libgl1 libglib2.0-0 python3.10 python3-pip python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt requirements-serverless.txt ./
RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && python3.10 -m pip install \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && python3.10 -m pip install -r requirements-inference.txt -r requirements-serverless.txt \
    && python3.10 -m pip install -U openmim \
    && mim install "mmengine==0.10.7" "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

COPY . .

# Bake all weights into the image so cold workers never download models.
RUN chmod +x download_weights.sh \
    && HF_ENDPOINT=https://huggingface.co ./download_weights.sh \
    && test -s models/musetalkV15/unet.pth \
    && test -s models/whisper/pytorch_model.bin \
    && test -s models/face-parse-bisent/79999_iter.pth

CMD ["python3.10", "-u", "handler.py"]
