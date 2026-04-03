# Anti-DeepVoice Guard — Project Instructions

## Project Overview
실시간 deepfake 음성 탐지 시스템. AASIST 모델 기반 Android 앱 + FastAPI 서버.
전화 수신 시 상대방 음성이 AI 생성(딥보이스)인지 실시간 판별하여 알림.

## Tech Stack
- **모델**: AASIST (PyTorch → ONNX), Silero VAD
- **Android**: Kotlin, Jetpack Compose, Hilt, ONNX Runtime, Room DB
- **서버**: Python, FastAPI, librosa, soundfile
- **학습**: PyTorch, MPS (M4 Pro 24GB), CUDA (RTX 4080 Super)

## Repository Structure
```
anti-deepvoice-guard/
├── android-app/           # Android 앱 (Kotlin + Jetpack Compose)
├── model-server/          # FastAPI 추론 서버
├── model-training/        # 모델 학습, fine-tuning, ONNX 변환
│   ├── models/AASIST.py   # AASIST 모델 아키텍처
│   ├── fine_tune.py       # Fine-tuning
│   ├── convert_to_onnx.py # PyTorch → ONNX
│   ├── generate_multi_tts.py    # 다중 TTS fake 데이터 생성
│   └── generate_clone_data.py   # XTTS v2 음성 클로닝 fake 생성
└── test-samples/          # 테스트 오디오 (real, fake, fake-diverse)
```

## Development Environment
- **주 작업**: M4 Pro 24GB (macOS) — 모델 학습(MPS), 서버, 앱 빌드
- **보조**: RTX 4080 Super — unfreeze 학습, 대규모 배치
- **Python venv**: `.venv/` (프로젝트 루트)
- **서버 실행**: `MODEL_PATH=model-training/weights/finetuned/aasist_best.pth .venv/bin/python -m uvicorn model_server.api.main:app --port 8000`

## Key Conventions
- ASVspoof label convention: index 0 = spoof/fake, index 1 = bonafide/real
- 오디오 입력: 16kHz mono PCM float32, 64600 samples (~4초)
- ONNX 모델은 embedded weights (단일 파일) 사용
- Android assets에 aasist.onnx + silero_vad.onnx 배포

## Review & Verification Process
모든 주요 변경사항은 다중 AI 에이전트 검토를 거칩니다:

### 1. Claude Code Agents (구현 + 1차 검토)
- `code-reviewer`: 코드 품질, 보안, 프로젝트 컨벤션 준수 확인
- `silent-failure-hunter`: 에러 핸들링, silent failure 탐지
- `pr-test-analyzer`: 테스트 커버리지 분석
- `type-design-analyzer`: 타입 설계 검토

### 2. Codex (GPT 5.4) 교차 검증
주요 구현 완료 후 Codex로 독립적 검증:
```bash
# 코드 리뷰
codex "review the changes in <files> for bugs, security issues, and correctness"

# 아키텍처 검증
codex "verify the architecture of <component> meets Android best practices"

# 테스트 생성
codex "generate test cases for <component> covering edge cases"
```

### 3. 검토 체크리스트
- [ ] Claude agents로 코드 리뷰 완료
- [ ] Codex로 교차 검증 완료
- [ ] 단위 테스트 통과
- [ ] 실제 오디오 샘플로 통합 테스트 완료
- [ ] 모델 정확도 regression 없음 확인

## Model Status (v3 fine-tuned)
- 학습: AIHub real 9,170 + multi-TTS/clone fake 8,684
- 테스트 정확도: WAV/MP3 Real 100%, Fake 기본 100%, Fake 다양한 TTS 88%
- 알려진 이슈: edge_ko_hyunsu/injoon 오탐, 특정 녹음환경 불확실

## Do NOT
- model-training/data/ 이하 대용량 데이터를 git에 커밋하지 말 것
- .env, 자격증명 파일을 커밋하지 말 것
- API 키를 코드에 하드코딩하지 말 것
