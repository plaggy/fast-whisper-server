import logging
import torch

from typing import Annotated
from pydantic import Json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForCausalLM
from huggingface_hub import HfApi

from app.utils.validation_utils import validate_file, process_params
from app.utils.diarization_utils import diarize
from app.utils.config import model_settings

logger = logging.getLogger(__name__)
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device.type}")

    torch_dtype = torch.float32 if device.type == "cpu" else torch.float16

    # from pytorch 2.2 sdpa implements flash attention 2
    models["asr_pipeline"] = pipeline(
        "automatic-speech-recognition",
        model=model_settings.asr_model,
        torch_dtype=torch_dtype,
        device=device
    )
    
    models["assistant_model"] = AutoModelForCausalLM.from_pretrained(
        model_settings.assistant_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ) if model_settings.assistant_model else None

    if models["assistant_model"]:
        models["assistant_model"].to(device)

    if model_settings.diarization_model:
        # diarization pipeline doesn't raise if there is no token
        HfApi().whoami(model_settings.hf_token)
        models["diarization_pipeline"] = Pipeline.from_pretrained(
            checkpoint_path=model_settings.diarization_model,
            use_auth_token=model_settings.hf_token,
        )
        models["diarization_pipeline"].to(device)
    else:
        models["diarization_pipeline"] = None

    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=PlainTextResponse)
@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "OK"


@app.post("/")
@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File()],
    parameters: Annotated[Json , Form()] = {}
):
    parameters = process_params(parameters)
    file = await validate_file(file)

    logger.info(f"inference parameters: {parameters}")

    generate_kwargs = {
        "task": parameters.task, 
        "language": parameters.language,
        "assistant_model": models["assistant_model"] if parameters.assisted else None
    }

    try:
        logger.info("starting ASR pipeline")
        asr_outputs = models["asr_pipeline"](
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

    if models["diarization_pipeline"]:
        try:
            transcript = diarize(models["diarization_pipeline"], file, parameters, asr_outputs)
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
