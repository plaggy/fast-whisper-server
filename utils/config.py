import os

from pydantic import BaseModel
from typing import Optional, Literal

audio_types = {"audio/mpeg", "audio/wav", "audio/webm"}


class ModelConfig(BaseModel):
    asr_model: str = os.getenv("ASR_MODEL")
    flash_attn2: bool = bool(os.getenv("FLASH_ATTN2", 0))
    assistant_model: Optional[str] = os.getenv("ASSISTANT_MODEL", None)
    diarization_model: Optional[str] = os.getenv("DIARIZATION_MODEL", None)
    hf_token: Optional[str] = os.getenv("HF_TOKEN")


class InferenceConfig(BaseModel):
    task: Literal["transcribe", "translate"] = "transcribe"
    batch_size: int = 24
    chunk_length_s: int = 30
    sampling_rate: int = 16000
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


model_config = ModelConfig()
inference_config = InferenceConfig()
