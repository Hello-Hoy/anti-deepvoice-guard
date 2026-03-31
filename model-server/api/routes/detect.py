"""Deepfake 음성 탐지 API 엔드포인트."""

import io
import struct

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from model_server.inference.aasist_model import DetectionResult

router = APIRouter()

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 10
MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION_SEC


class DetectionResponse(BaseModel):
    fake_score: float
    real_score: float
    confidence: float
    model_version: str
    inference_ms: float


def _read_audio(data: bytes) -> np.ndarray:
    """오디오 바이트를 numpy float32 배열로 변환.

    지원 형식:
    - WAV/FLAC 등 soundfile 지원 형식
    - raw PCM 16-bit signed (폴백)
    """
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if sr != SAMPLE_RATE:
            # 간단한 리샘플링: 정확한 리샘플링이 필요하면 librosa 사용
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    except Exception:
        # raw PCM 16-bit signed little-endian으로 시도
        n_samples = len(data) // 2
        audio = np.array(struct.unpack(f"<{n_samples}h", data[:n_samples * 2]), dtype=np.float32)
        audio /= 32768.0  # normalize to [-1, 1]

    # mono로 변환
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 길이 제한
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    return audio


async def detect_deepfake(file: UploadFile) -> DetectionResponse:
    """오디오 파일을 받아 deepfake 여부를 분석한다."""
    from model_server.api.main import get_detector

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if len(data) > 10 * 1024 * 1024:  # 10MB 제한
        raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")

    try:
        audio = _read_audio(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio format: {e}")

    if len(audio) < SAMPLE_RATE:  # 최소 1초
        raise HTTPException(status_code=400, detail="Audio too short (min 1 second)")

    detector = get_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result: DetectionResult = detector.detect(audio)

    return DetectionResponse(
        fake_score=round(result.fake_score, 6),
        real_score=round(result.real_score, 6),
        confidence=round(result.confidence, 6),
        model_version=result.model_version,
        inference_ms=result.inference_ms,
    )
