import logging
import os

from pathlib import Path
from typing import Annotated
from pydantic import Json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager

from app.utils.setup_utils import register_custom_pipeline_from_directory
from app.utils.validation_utils import validate_file, process_params, check_cuda_fa2

logger = logging.getLogger(__name__)
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    check_cuda_fa2()
    app_state["pipeline"] = register_custom_pipeline_from_directory(os.getenv("HF_MODEL_DIR", Path(__file__).parent))
    yield
    app_state.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/")
@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File()],
    parameters: Annotated[Json , Form()] = {}
):
    parameters = process_params(parameters)
    file = await validate_file(file)
    try:
        out = await app_state["pipeline"](file, parameters)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown error: {e}"
        )
    
    return out


# debugging

# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7860, log_config="hub-model-server/app/utils/log_config.yaml")
