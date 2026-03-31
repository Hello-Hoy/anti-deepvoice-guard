# Anti-DeepVoice Guard

실시간 Deepfake 음성 탐지 Android 앱. 백그라운드에서 오디오를 모니터링하고 AI 생성 음성을 감지하여 사용자에게 경고합니다.

## Architecture

```
마이크(16kHz) → RingBuffer(30s) → Silero VAD → SegmentExtractor → AASIST 추론 → 알림
```

**핵심 설계:**
- **2단계 파이프라인**: 경량 VAD가 음성 구간만 필터링 → AASIST가 deepfake 판별 (배터리 효율)
- **추론 백엔드 추상화**: On-Device (ONNX Runtime) / Server (FastAPI) 런타임 스위칭
- **보안**: AES-256-GCM 암호화 저장, AndroidKeyStore 키 관리, 자동 삭제 정책

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `model-training/` | PyTorch → ONNX 모델 변환 | Done + Verified |
| `model-server/` | FastAPI 추론 서버 + Docker | Done |
| `android-app/` | Kotlin + Jetpack Compose 앱 | Done |

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

### 2. 서버 실행 (선택)

```bash
cd model-server
pip install -r requirements.txt
PYTHONPATH=.. uvicorn model_server.api.main:app --port 8000
```

### 3. Android 앱 빌드

```bash
# Android Studio에서 android-app/ 열기 또는:
cd android-app
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

자세한 빌드 가이드: [`android-app/BUILD_GUIDE.md`](android-app/BUILD_GUIDE.md)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | AASIST-L (107K params, 644KB ONNX) |
| VAD | Silero VAD v5 (2.3MB ONNX) |
| Android | Kotlin, Jetpack Compose, Material3 |
| DI | Hilt |
| Inference | ONNX Runtime Android |
| Network | Retrofit + OkHttp |
| Storage | Room DB + AES-256-GCM EncryptedFile |
| Server | FastAPI + PyTorch |

## Model Performance

| Metric | Value |
|--------|-------|
| ONNX 변환 정확도 | diff < 1e-8 (PyTorch vs ONNX) |
| On-Device 추론 | ~100ms per 5s segment |
| 모델 크기 | AASIST-L: 644KB, Silero VAD: 2.3MB |
| 메모리 사용 | ~12MB (추론 시) |

## Recent Changes (Phase 6 — Runtime Fix)

Codex adversarial review로 발견된 4개 핵심 블로커 수정:

| Issue | Fix |
|-------|-----|
| HomeScreen이 null 파이프라인 스냅샷 | 서비스 StateFlow를 live observable로 노출, UI가 실시간 구독 |
| 런타임 RECORD_AUDIO 권한 체크 없음 | START 전 권한 요청 게이팅 + AudioRecord 초기화 검증 |
| 감지 결과 메모리 전용 (DB/알림 데드코드) | detectionEvents → Room DB 저장 + 암호화 오디오 + 푸시 알림 |
| Settings UI가 서비스에 영향 없음 | DataStore 영속화 → 엔진 선택/threshold 서비스에 주입 |

## Based On

[deep-fake-audio-detection](../deep-fake-audio-detection/) — SW중심대학 경진대회 AI부문 10위 (219팀 중)
- AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
- RawBoost data augmentation + DANN domain adaptation
