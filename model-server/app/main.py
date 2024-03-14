import logging
import torch

from typing import Annotated
from pydantic import Json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForCausalLM

from app.utils.validation_utils import validate_file, process_params, check_cuda_fa2
from app.utils.diarization_utils import diarize
from app.utils.config import model_settings

logger = logging.getLogger(__name__)
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device.type}")

    torch_dtype = torch.float32 if device.type == "cpu" else torch.float16
    check_cuda_fa2(device)

    attn_implementation = "flash_attention_2" if model_settings.flash_attn2 else "sdpa"

    models["asr_pipeline"] = pipeline(
        "automatic-speech-recognition",
        model=model_settings.asr_model,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": attn_implementation},
    )

    models["assistant_model"] = AutoModelForCausalLM.from_pretrained(
        model_settings.assistant_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=attn_implementation,
    ) if model_settings.assistant_model else None

    if models["assistant_model"]:
        models["assistant_model"].to(device)

    models["diarization_pipeline"] = Pipeline.from_pretrained(
        checkpoint_path=model_settings.diarization_model,
        use_auth_token=model_settings.hf_token,
    ) if model_settings.diarization_model else None

    if models["diarization_pipeline"]:
        models["diarization_pipeline"].to(device)
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


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


# debugging

# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7860, log_config="app/utils/log_config.yaml")
