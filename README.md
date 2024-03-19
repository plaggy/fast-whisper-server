Use with a prebuilt image:
```
docker run --gpus all -p 7860:7860 --env-file .env ghcr.io/plaggy/asrdiarization-server:latest
```
and parametrize via `.env`:
```
ASR_MODEL=
FLASH_ATTN2=
DIARIZATION_MODEL=
ASSISTANT_MODEL=
HF_TOKEN=
```

Or build your own in a usual way
