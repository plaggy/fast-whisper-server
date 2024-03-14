import json
import os
import logging
import pytest
from fastapi.testclient import TestClient


logger = logging.getLogger(__name__)


@pytest.fixture
def mock_app(monkeypatch):
    # setting up environment before import
    monkeypatch.setenv("ASR_MODEL", "openai/whisper-small")
    monkeypatch.setenv("FLASH_ATTN2", "0")
    monkeypatch.setenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    monkeypatch.setenv("ASSISTANT_MODEL", "distil-whisper/distil-small.en")
    monkeypatch.setenv("HF_TOKEN", os.getenv("HF_TOKEN"))

    from app.main import app
    return app


def test_predict(mock_app):
    with TestClient(mock_app) as test_client:
        token = os.getenv("HF_TOKEN")
        logger.info(token)
        files = {"file": open("app/tests/polyai-minds14-0.wav", "rb")}
        data = {"parameters": json.dumps({"batch_size": 12, "sampling_rate": 24000, "non-existent": "here"})}
        resp = test_client.post("/predict", data=data, files=files) 
        resp_json = resp.json()
        assert resp.status_code == 200
        assert "speakers" in resp_json and "text" in resp_json


def test_predict_no_params(mock_app):
    with TestClient(mock_app) as test_client:
        files = {"file": open("app/tests/polyai-minds14-0.wav", "rb")}
        resp = test_client.post("/predict", files=files) 
        resp_json = resp.json()
        assert resp.status_code == 200
        assert "speakers" in resp_json and "text" in resp_json


def test_predict_assisted(mock_app):
    with TestClient(mock_app) as test_client:
        files = {"file": open("app/tests/polyai-minds14-0.wav", "rb")}
        data = {"parameters": json.dumps({"batch_size": 1, "assisted": True})}
        resp = test_client.post("/predict", data=data, files=files) 
        resp_json = resp.json()
        assert resp.status_code == 200
        assert "speakers" in resp_json and "text" in resp_json
