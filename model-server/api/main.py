"""Anti-DeepVoice Guard — FastAPI 서버.

AASIST 모델을 사용하여 deepfake 음성을 탐지하는 REST API 서버.

Usage:
    uvicorn model_server.api.main:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_server.inference.aasist_model import AASISTDetector

# 전역 모델 인스턴스
_detector: AASISTDetector | None = None

# 기본 모델 경로
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "deep-fake-audio-detection"
    / "code"
    / "2_aasist_rawboost"
    / "models"
    / "weights"
    / "AASIST.pth"
)


def get_detector() -> AASISTDetector | None:
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    config_path = os.environ.get("CONFIG_PATH", None)
    device = os.environ.get("DEVICE", "cpu")

    print(f"Loading AASIST model from {model_path} (device={device})...")
    _detector = AASISTDetector(model_path=model_path, config_path=config_path, device=device)
    print("Model loaded successfully.")

    yield

    _detector = None


app = FastAPI(
    title="Anti-DeepVoice Guard API",
    version="1.0.0",
    description="Deepfake voice detection API using AASIST model",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _detector is not None,
    }


@app.get("/api/v1/model/info")
async def model_info():
    if _detector is None:
        return {"error": "Model not loaded"}
    return {
        "version": _detector.model_version,
        "sample_rate": 16000,
        "input_samples": 80000,
        "input_duration_sec": 5.0,
    }


@app.post("/api/v1/detect")
async def detect(audio: UploadFile):
    from model_server.api.routes.detect import detect_deepfake
    return await detect_deepfake(audio)
