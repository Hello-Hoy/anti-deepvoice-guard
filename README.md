# Anti-DeepVoice Guard

실시간 Deepfake 음성 탐지 Android 앱. 백그라운드에서 오디오를 모니터링하고 AI 생성 음성을 감지하여 사용자에게 경고합니다.

## Architecture

```
마이크(16kHz) → RingBuffer(30s) → Silero VAD → SegmentExtractor → AASIST 추론 → 알림
```

## Modules

| Module | Description |
|--------|-------------|
| `model-server/` | FastAPI 기반 추론 서버 (PyTorch AASIST) |
| `model-training/` | 모델 변환 (PyTorch → ONNX) |
| `android-app/` | Android 앱 (Kotlin + Jetpack Compose) |

## Quick Start

### 1. ONNX 모델 변환

```bash
cd model-training
pip install -r requirements.txt
python convert_to_onnx.py \
    --model-path ../../deep-fake-audio-detection/code/2_aasist_rawboost/models/weights/AASIST-L.pth \
    --output ../model-server/models/weights/aasist-l.onnx \
    --verify
```

### 2. 서버 실행

```bash
cd model-server
pip install -r requirements.txt
PYTHONPATH=.. uvicorn model_server.api.main:app --port 8000
```

### 3. API 테스트

```bash
curl -X POST http://localhost:8000/api/v1/detect \
    -F "audio=@test_audio.wav"
```

## Based On

[deep-fake-audio-detection](../deep-fake-audio-detection/) — SW중심대학 경진대회 AI부문 10위 (219팀 중)
- AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
- RawBoost data augmentation + DANN domain adaptation
