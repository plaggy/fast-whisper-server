import json
import os
import logging
import pytest
from fastapi.testclient import TestClient


logger = logging.getLogger(__name__)


@pytest.fixture
def mock_app(monkeypatch):
    # setting up environment before import
    # make sure to set env HF_TOKEN 
    monkeypatch.setenv("ASR_MODEL", "openai/whisper-small")
    monkeypatch.setenv("ASSISTANT_MODEL", "distil-whisper/distil-small.en")
    monkeypatch.setenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    monkeypatch.setenv("HF_TOKEN", os.getenv("HF_TOKEN"))

    from app.main import app
    return app


def test_predict(mock_app):
    with TestClient(mock_app) as test_client:
        files = {"file": open("app/tests/polyai-minds14-0.wav", "rb")}
        data = {"parameters": json.dumps({"batch_size": 12, "sampling_rate": 24000, "non-existent": "dummy"})}
        resp = test_client.post("/predict", data=data, files=files) 
        resp_json = resp.json()
        assert resp.status_code == 200
        assert resp_json["speakers"] and resp_json["text"]


def test_predict_no_params(mock_app):
    with TestClient(mock_app) as test_client:
        files = {"file": open("app/tests/polyai-minds14-0.wav", "rb")}
        resp = test_client.post("/predict", files=files) 
        resp_json = resp.json()
        assert resp.status_code == 200
        assert resp_json["speakers"] and resp_json["text"]


def test_predict_assisted(mock_app):
    with TestClient(mock_app) as test_client:
        files = {"file": open("app/tests/polyai-minds14-0.wav", "rb")}
        data = {"parameters": json.dumps({"batch_size": 1, "assisted": True})}
        resp = test_client.post("/predict", data=data, files=files) 
        resp_json = resp.json()
        assert resp.status_code == 200
        assert resp_json["speakers"] and resp_json["text"]


def test_predict_all_params(mock_app):
    with TestClient(mock_app) as test_client:
        files = {"file": open("app/tests/polyai-minds14-0.wav", "rb")}
        data = {
            "parameters": 
                json.dumps({
                    "batch_size": 1, 
                    "assisted": True,
                    "chunk_length_s": 24,
                    "sampling_rate": 24000,
                    "language": "en",
                    "min_speakers": 1,
                    "max_speakers": 2
                })
        }
        resp = test_client.post("/predict", data=data, files=files) 
        resp_json = resp.json()
        assert resp.status_code == 200
        assert resp_json["speakers"] and resp_json["text"]