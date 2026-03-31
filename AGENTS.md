# Anti-DeepVoice Guard — Agent Instructions

## Project Overview
실시간 deepfake 음성 탐지 Android 앱. AASIST 모델 기반.

## Repository Structure
```
anti-deepvoice-guard/
├── model-server/          # FastAPI 추론 서버
├── model-training/        # PyTorch → ONNX 변환
├── android-app/           # Android 앱 (Phase 2에서 구현 예정)
└── docs/                  # 스펙 문서
```

## For Code Review Agents
- 스펙 문서: `../docs/superpowers/specs/2026-03-31-anti-deepvoice-guard-design.md`
- 기존 AASIST 모델: `../deep-fake-audio-detection/code/2_aasist_rawboost/models/AASIST.py`
- 모델 설정: `../deep-fake-audio-detection/code/2_aasist_rawboost/config/AASIST.conf`

## For Testing Agents
1. **모델 변환 검증**: `model-training/convert_to_onnx.py` — AASIST-L.pth → ONNX 변환 및 출력 비교
2. **서버 API 테스트**: `model-server/api/main.py` — FastAPI 서버 기능 테스트
3. **추론 정확도**: 기존 PyTorch 출력과 ONNX 출력 일치 확인

## Key Technical Decisions
- SincConv의 band_pass 필터를 고정 nn.Parameter로 변환하여 ONNX 호환
- DANN (domain_classifier, ReverseLayerF) 제거 — 추론 전용
- forward() 출력에 softmax 적용하여 확률값 반환
