import argparse
import base64
import binascii
import ipaddress
import mimetypes
import os
import socket
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import runpod
import torch
from omegaconf import OmegaConf
from runpod.serverless.utils import upload_file_to_bucket

from scripts import inference


ROOT = Path(__file__).resolve().parent
MAX_INPUT_BYTES = int(os.getenv("MAX_INPUT_BYTES", str(100 * 1024 * 1024)))
MAX_BASE64_OUTPUT_BYTES = int(os.getenv("MAX_BASE64_OUTPUT_BYTES", str(7 * 1024 * 1024)))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
_runtime = None
_runtime_lock = threading.Lock()


def _args(config_path: Path, result_dir: Path, output_name: str, payload: dict):
    return argparse.Namespace(
        ffmpeg_path="/usr/bin",
        gpu_id=0,
        vae_type="sd-vae",
        unet_config=str(ROOT / "models/musetalkV15/musetalk.json"),
        unet_model_path=str(ROOT / "models/musetalkV15/unet.pth"),
        whisper_dir=str(ROOT / "models/whisper"),
        inference_config=str(config_path),
        bbox_shift=0,
        result_dir=str(result_dir),
        extra_margin=int(payload.get("extra_margin", 10)),
        fps=int(payload.get("fps", 25)),
        audio_padding_length_left=int(payload.get("audio_padding_left", 2)),
        audio_padding_length_right=int(payload.get("audio_padding_right", 2)),
        batch_size=int(payload.get("batch_size", 8)),
        output_vid_name=output_name,
        use_saved_coord=False,
        saved_coord=False,
        use_float16=True,
        parsing_mode=payload.get("parsing_mode", "jaw"),
        left_cheek_width=int(payload.get("left_cheek_width", 90)),
        right_cheek_width=int(payload.get("right_cheek_width", 90)),
        version="v15",
    )


def _get_runtime(args):
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                if not torch.cuda.is_available() and os.getenv("ALLOW_CPU", "false").lower() != "true":
                    raise RuntimeError("MuseTalk requires a CUDA GPU; no CUDA device was detected")
                _runtime = inference.load_runtime(args)
    return _runtime


def _validate_public_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Media URL must use http or https")
    if os.getenv("ALLOW_PRIVATE_URLS", "false").lower() == "true":
        return
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve media host: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private, loopback, and link-local media URLs are not allowed")


def _extension(url: str, content_type: str, fallback: str):
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix and len(suffix) <= 6:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return guessed or fallback


def _download(url: str, directory: Path, stem: str, fallback_ext: str):
    current = url
    response = None
    for _ in range(4):
        _validate_public_url(current)
        response = requests.get(current, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=False)
        if response.is_redirect:
            current = urljoin(current, response.headers["Location"])
            response.close()
            continue
        response.raise_for_status()
        break
    else:
        raise ValueError("Too many redirects while downloading media")

    output = directory / f"{stem}{_extension(current, response.headers.get('Content-Type', ''), fallback_ext)}"
    total = 0
    with output.open("wb") as handle:
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_INPUT_BYTES:
                raise ValueError(f"{stem} exceeds the {MAX_INPUT_BYTES} byte input limit")
            handle.write(chunk)
    response.close()
    if total == 0:
        raise ValueError(f"{stem} download was empty")
    return output


def _decode_media(value: str, directory: Path, stem: str, fallback_ext: str):
    if value.startswith("data:"):
        header, value = value.split(",", 1)
        media_type = header[5:].split(";", 1)[0]
        fallback_ext = mimetypes.guess_extension(media_type) or fallback_ext
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{stem}_base64 is not valid base64") from exc
    if not data or len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"{stem}_base64 is empty or exceeds the input limit")
    output = directory / f"{stem}{fallback_ext}"
    output.write_bytes(data)
    return output


def _materialize(payload: dict, directory: Path, stem: str, fallback_ext: str):
    url = payload.get(f"{stem}_url")
    encoded = payload.get(f"{stem}_base64")
    if bool(url) == bool(encoded):
        raise ValueError(f"Provide exactly one of {stem}_url or {stem}_base64")
    return _download(url, directory, stem, fallback_ext) if url else _decode_media(encoded, directory, stem, fallback_ext)


def _serialize_output(path: Path, job_id: str, output_format: str):
    bucket_ready = all(os.getenv(name) for name in (
        "BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY", "BUCKET_NAME"
    ))
    if output_format == "auto":
        output_format = "url" if bucket_ready else "base64"
    if output_format == "url":
        if not bucket_ready:
            raise ValueError("output_format=url requires BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY, and BUCKET_NAME")
        url = upload_file_to_bucket(
            path.name,
            str(path),
            bucket_name=os.environ["BUCKET_NAME"],
            prefix=f"musetalk/{job_id}",
            extra_args={"ContentType": "video/mp4"},
        )
        return {"video_url": url}
    if output_format != "base64":
        raise ValueError("output_format must be auto, url, or base64")
    size = path.stat().st_size
    if size > MAX_BASE64_OUTPUT_BYTES:
        raise ValueError("Generated video is too large for an inline response; configure S3 and use output_format=url")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"video_base64": f"data:video/mp4;base64,{encoded}"}


def handler(event):
    payload = event.get("input") or {}
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    job_id = str(event.get("id") or uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix="musetalk-") as temp:
        temp_dir = Path(temp)
        source_payload = dict(payload)
        if "source_url" in payload:
            source_payload["video_url"] = payload["source_url"]
        if "source_base64" in payload:
            source_payload["video_base64"] = payload["source_base64"]
        video_path = _materialize(source_payload, temp_dir, "video", ".mp4")
        audio_path = _materialize(payload, temp_dir, "audio", ".wav")
        result_dir = temp_dir / "results"
        config_path = temp_dir / "job.yaml"
        output_name = "musetalk-output.mp4"
        OmegaConf.save(
            OmegaConf.create({"job": {"video_path": str(video_path), "audio_path": str(audio_path), "result_name": output_name}}),
            config_path,
        )
        args = _args(config_path, result_dir, output_name, payload)
        outputs = inference.main(args, runtime=_get_runtime(args))
        if not outputs:
            raise RuntimeError("MuseTalk did not produce an output video")
        output = Path(outputs[0])
        response = _serialize_output(output, job_id, str(payload.get("output_format", "auto")).lower())
        response.update({"model": "MuseTalk 1.5", "content_type": "video/mp4"})
        return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
