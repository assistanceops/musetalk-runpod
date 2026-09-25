FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/hundevmode/musetalk-runpod" \
      org.opencontainers.image.description="MuseTalk 1.5 worker for RunPod Serverless" \
      org.opencontainers.image.licenses="MIT"

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
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python3.10 -m pip install -r requirements-inference.txt -r requirements-serverless.txt \
    && python3.10 -m pip install --no-build-isolation "chumpy==0.70" \
    && python3.10 -m pip install \
        "mmengine==0.10.7" "mmcv==2.1.0" "mmdet==3.1.0" "mmpose==1.1.0" \
        --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    && python3.10 -m pip check
COPY . .

# Bake all weights into the image so cold workers never download models.
RUN chmod +x download_weights.sh \
    && HF_ENDPOINT=https://huggingface.co ./download_weights.sh \
    && test -s models/musetalkV15/unet.pth \
    && test -s models/whisper/pytorch_model.bin \
    && test -s models/face-parse-bisent/79999_iter.pth

CMD ["python3.10", "-u", "handler.py"]
