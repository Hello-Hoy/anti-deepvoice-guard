# Anti-DeepVoice Guard — Agent Instructions

## Project Overview
실시간 deepfake 음성 탐지 Android 앱. AASIST 모델 기반.

## Repository Structure
```
anti-deepvoice-guard/
├── model-training/        # 모델 아키텍처, fine-tuning, ONNX 변환
│   ├── models/AASIST.py   # AASIST 모델 (자기완결적)
│   ├── configs/            # 모델 config (aasist.json, aasist_l.json)
│   ├── weights/            # official / legacy / finetuned
│   ├── fine_tune.py        # Fine-tuning 스크립트
│   └── convert_to_onnx.py  # PyTorch → ONNX 변환
├── model-server/          # FastAPI 추론 서버
├── android-app/           # Android 앱 (Kotlin + Jetpack Compose)
└── test-samples/          # 테스트 오디오 샘플
```

## For Code Review Agents
- AASIST 모델: `model-training/models/AASIST.py`
- 모델 설정: `model-training/configs/aasist.json`
- 공식 가중치: `model-training/weights/official/AASIST.pth`

## For Testing Agents
1. **모델 변환 검증**: `model-training/convert_to_onnx.py` — ONNX 변환 및 출력 비교
2. **서버 API 테스트**: `model-server/api/main.py` — FastAPI 서버 기능 테스트
3. **추론 정확도**: 기존 PyTorch 출력과 ONNX 출력 일치 확인

## Key Technical Decisions
- SincConv의 band_pass 필터를 고정 nn.Parameter로 변환하여 ONNX 호환
- DANN (domain_classifier, ReverseLayerF) 제거 — 추론 전용
- forward() 출력에 softmax 적용하여 확률값 반환 (CrossEntropyLoss 학습과 일관)
- Device: MPS (M4 Pro) > CUDA (NVIDIA) > CPU 자동 감지
