import logging
import mimetypes

from pydantic import Json, BaseModel
from fastapi import UploadFile, File, Form
from typing import Type

from utils.config import inference_config, audio_types

logging.basicConfig(level="INFO")


def process_params(parameters: Json = Form(default=None)) -> Type[BaseModel]:
    default_fields = inference_config.model_fields
    unsupported = [parameters.pop(k) for k in default_fields if k not in default_fields]
    parameters = inference_config.model_copy(update=parameters)

    if unsupported:
        logging.warning(f"parameters are not supported and will not be used: {unsupported}")

    return parameters


async def validate_file(file: UploadFile = File()):
    content_type = file.content_type
    if not content_type:
        content_type = mimetypes.guess_type(file.filename)[0]
        logging.warning(f"content type was not provided, guessed as {content_type}")

    assert content_type in audio_types, \
        f"file type {file.content_type} not supported"

    return await file.read()
