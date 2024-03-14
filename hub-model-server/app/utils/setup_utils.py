import logging
import importlib.util
import sys

from fastapi import HTTPException
from pathlib import Path

from app.utils.config import model_settings

HF_DEFAULT_PIPELINE_NAME = "handler.py"
HF_MODULE_NAME = f"{Path(HF_DEFAULT_PIPELINE_NAME).stem}.EndpointHandler"

logger = logging.getLogger(__name__)

# by Philipp Schmid https://www.philschmid.de/

def register_custom_pipeline_from_directory(model_dir):
    """
    Checks if a custom pipeline is available and registers it if so.
    """
    # path to custom handler
    custom_module = Path(model_dir).joinpath(HF_DEFAULT_PIPELINE_NAME)
    if not custom_module.is_file():
        logger.error(f"Path {HF_DEFAULT_PIPELINE_NAME} not found")
        raise HTTPException(status_code=400, detail=f"Path {HF_DEFAULT_PIPELINE_NAME} not found")
    
    logger.info(f"Found custom pipeline at {custom_module}")
    spec = importlib.util.spec_from_file_location(HF_MODULE_NAME, custom_module)
    if not spec:
        logger.error(f"EndpointHandler not found in {HF_DEFAULT_PIPELINE_NAME}")
        raise HTTPException(status_code=400, detail=f"EndpointHandler not found in {HF_DEFAULT_PIPELINE_NAME}")
    
    # add the whole directory to path for submodlues
    sys.path.insert(0, model_dir)
    # import custom handler
    handler = importlib.util.module_from_spec(spec)
    sys.modules[HF_MODULE_NAME] = handler
    spec.loader.exec_module(handler)
    # init custom handler with model_dir
    custom_pipeline = handler.EndpointHandler(model_settings)
    
    return custom_pipeline
