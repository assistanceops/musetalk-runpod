#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalkV15 models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# Use the official Hugging Face endpoint unless the builder explicitly overrides it.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# Download pinned runtime artifacts without depending on a CLI version.
python3.10 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    local_dir="models",
    allow_patterns=["musetalkV15/musetalk.json", "musetalkV15/unet.pth"],
)
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir="models/sd-vae",
    allow_patterns=["config.json", "diffusion_pytorch_model.bin"],
)
snapshot_download(
    repo_id="openai/whisper-tiny",
    local_dir="models/whisper",
    allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"],
)
snapshot_download(
    repo_id="yzd-v/DWPose",
    local_dir="models/dwpose",
    allow_patterns=["dw-ll_ucoco_384.pth"],
)
PY

# Download Face Parse Bisent weights
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

# Face detector weight otherwise fetched during the first cold start.
curl -L https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth \
  -o musetalk/utils/face_detection/detection/sfd/s3fd.pth

echo "✅ All weights have been downloaded successfully!" 
