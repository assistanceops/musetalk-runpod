# MuseTalk 1.5 on RunPod Serverless

The worker accepts a source image/video plus an audio track and returns a lip-synced MP4. Models are loaded once per warm worker.

## Request

```json
{
  "input": {
    "video_url": "https://public.example/avatar.mp4",
    "audio_url": "https://public.example/speech.wav",
    "output_format": "auto",
    "fps": 25,
    "batch_size": 8
  }
}
```

`source_url` is accepted as an alias for `video_url`. Public HTTP(S) URLs and base64 inputs (`video_base64`, `audio_base64`) are supported. Private-network URLs are rejected by default.

With `output_format=auto`, the worker uploads to S3-compatible storage when all bucket variables are configured; otherwise it returns an inline `data:video/mp4;base64,...` value. Inline outputs are capped at 7 MiB by default.

## Optional endpoint variables

- `BUCKET_ENDPOINT_URL`
- `BUCKET_ACCESS_KEY_ID`
- `BUCKET_SECRET_ACCESS_KEY`
- `BUCKET_NAME`
- `MAX_INPUT_BYTES` (default: 100 MiB per input)
- `MAX_BASE64_OUTPUT_BYTES` (default: 7 MiB)

Use a CUDA GPU with at least 8 GB VRAM. A 16 GB GPU is the safer default for longer clips or larger batches. Endpoint idle timeout and maximum execution time should account for face preprocessing and video encoding.
