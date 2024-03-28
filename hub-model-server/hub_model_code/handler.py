import logging
import torch

from fastapi import HTTPException
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForCausalLM
from diarization_utils import diarize
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


class EndpointHandler():

    def __init__(self, model_settings):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Using device: {device.type}")
        torch_dtype = torch.float32 if device.type == "cpu" else torch.float16

        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            model_settings.assistant_model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ) if model_settings.assistant_model else None

        if self.assistant_model:
            self.assistant_model.to(device)

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_settings.asr_model,
            torch_dtype=torch_dtype,
            device=device
        )

        if model_settings.diarization_model:
            # diarization pipeline doesn't raise if there is no token
            HfApi().whoami(model_settings.hf_token)
            self.diarization_pipeline = Pipeline.from_pretrained(
                checkpoint_path=model_settings.diarization_model,
                use_auth_token=model_settings.hf_token,
            )
            self.diarization_pipeline.to(device)
        else:
            self.diarization_pipeline = None

    async def __call__(self, file, parameters):

        logger.info(f"inference parameters: {parameters}")

        generate_kwargs = {
            "task": parameters.task, 
            "language": parameters.language,
            "assistant_model": self.assistant_model if parameters.assisted else None
        }

        try:
            asr_outputs = self.asr_pipeline(
                file,
                chunk_length_s=parameters.chunk_length_s,
                batch_size=parameters.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
        except RuntimeError as e:
            logger.error(f"ASR inference error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"ASR inference error: {str(e)}")
        except Exception as e:
            logger.error(f"Unknown error diring ASR inference: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unknown error diring ASR inference: {str(e)}")

        if self.diarization_pipeline:
            try:
                transcript = diarize(self.diarization_pipeline, file, parameters, asr_outputs)
            except RuntimeError as e:
                logger.error(f"Diarization inference error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Diarization inference error: {str(e)}")
            except Exception as e:
                logger.error(f"Unknown error during diarization: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unknown error during diarization: {str(e)}")
        else:
            transcript = []

        return {
            "speakers": transcript,
            "chunks": asr_outputs["chunks"],
            "text": asr_outputs["text"],
        }