import logging
import torch

from typing import Annotated
from pydantic import Json
from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForCausalLM

from utils.validation_utils import validate_file, process_params
from utils.diarization_utils import diarize
from utils.config import model_config

logging.basicConfig(level="INFO")

models = {}
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("mps") if torch.backends.mps.is_available() \
    else torch.device("cpu")
attn_implementation = "flash_attention_2" if model_config.flash_attn2 else "sdpa"


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["assistant_model"] = AutoModelForCausalLM.from_pretrained(
        model_config.assistant_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=attn_implementation,
    ) if model_config.assistant_model else None

    if models["assistant_model"]:
        models["assistant_model"].to(device)

    models["asr_pipeline"] = pipeline(
        "automatic-speech-recognition",
        model=model_config.asr_model,
        torch_dtype=torch.float16,
        device=device,
        model_kwargs={"attn_implementation": attn_implementation},
    )

    models["diarization_pipeline"] = Pipeline.from_pretrained(
        checkpoint_path=model_config.diarization_model,
        use_auth_token=model_config.hf_token,
    ) if model_config.diarization_model else None

    if models["diarization_pipeline"]:
        models["diarization_pipeline"].to(device)
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(
    parameters: Annotated[Json, Form()],
    file: Annotated[UploadFile, File()]
):
    parameters = process_params(parameters)
    file = await validate_file(file)

    logging.info(f"inference parameters: {parameters}")

    generate_kwargs = {"task": parameters.task, "language": parameters.language}
    if model_config.asr_model.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    asr_outputs = models["asr_pipeline"](
        file,
        chunk_length_s=parameters.chunk_length_s,
        batch_size=parameters.batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )

    if models["diarization_pipeline"]:
        transcript = diarize(models["diarization_pipeline"], file, parameters, asr_outputs)
    else:
        transcript = []


    return {
        "speakers": transcript,
        "chunks": asr_outputs["chunks"],
        "text": asr_outputs["text"],
    }


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
