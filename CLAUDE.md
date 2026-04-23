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
│   ├── fine_tune.py       # Fine-tuning (repeat-pad + soundfile loader)
│   ├── convert_to_onnx.py # PyTorch → ONNX (dynamo=False, legacy path — ORT Android 호환)
│   ├── data/
│   │   ├── real/, fake/                  # 원본 AIHub + TTS (gitignored)
│   │   ├── real_extra_iphone/            # iPhone 녹음 real 878개 (gitignored)
│   │   ├── mobile_fake_capture/          # Android 마이크로 녹음한 TTS 445개 (gitignored)
│   │   └── train_v*.csv / val_v*.csv / test_*.csv  # 학습 iteration 매니페스트
│   ├── generate_multi_tts.py    # 다중 TTS fake 데이터 생성
│   └── generate_clone_data.py   # XTTS v2 음성 클로닝 fake 생성
└── test-samples/          # 테스트 오디오 (real, fake, fake-diverse, real-voice-data, voice-clone-raw)
```

## Development Environment
- **주 작업**: M4 Pro 24GB (macOS) — 모델 학습(MPS), 서버, 앱 빌드
- **보조**: RTX 4080 Super — unfreeze 학습, 대규모 배치
- **Python venv**: `.venv/` (프로젝트 루트)
- **서버 실행**: `MODEL_PATH=model-training/weights/finetuned_v4/aasist_best.pth .venv/bin/python -m uvicorn model_server.api.main:app --port 8000`
- **ONNX 재변환**: `.venv/bin/python model-training/convert_to_onnx.py --model-path model-training/weights/finetuned_v4/aasist_best.pth --output android-app/app/src/main/assets/aasist.onnx --verify`
- **Android 빌드**: `cd android-app && ./gradlew :app:assembleDebug` 또는 Android Studio에서 ▶ Run

## Key Conventions
- ASVspoof label convention: index 0 = spoof/fake, index 1 = bonafide/real
- 오디오 입력: 16kHz mono PCM float32, 64600 samples (~4초)
- Padding 정책: **repeat-pad** (training/inference 모두 동일). 학습 zero-pad였던 이전 정책 통일됨
- ONNX export는 **legacy TorchScript path (dynamo=False)** 사용 — 기본 dynamo path가 ORT Android 1.19.2 호환 안 되는 ops 생성
- Silero VAD v5 호출 시 **이전 프레임 context 64 샘플을 앞에 concat** 해서 (1, 576) 입력 (공식 wrapper 규약)
- Android assets: `aasist.onnx` (1.4 MB, v4 fine-tuned) + `silero_vad.onnx` (2.3 MB)
- NarrowbandPreprocessor는 **런타임 비활성화** (v4 모델은 raw PCM 입력으로 학습됨 — 적용 시 감지력 56→7% 하락 실측). Settings 토글 있지만 기본 OFF

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

## Model Status (v4 fine-tuned — 2026-04)

### 학습 데이터
- AIHub real 9,170 + multi-TTS/clone fake 8,684 (기존 v3 베이스)
- **+ iPhone 녹음 real 878개** (hyohee v1/v2, lee, baek, jeong — speaker-aware split, jeong 50개는 완전 홀드아웃)
- **+ Mobile-captured fake 445개** (Mac 스피커 → Xcover 5 마이크로 216개 TTS 재생 녹음)

### 실측 성능 (Android Xcover 5 기준)
- **실기기 한국어 real 오탐**: 33% → **0%** (홀드아웃 jeong 평균 fake score 0.014)
- **직접 파일 TTS 감지** (Python): **100%** (fake score 0.87~1.00)
- **스피커→마이크 TTS 감지** (Python 445개 홀드아웃): **56%** (> 0.5 fake)
- **스피커→마이크 TTS 감지** (실기기 소규모): **~20%** (라운드별 acoustic variance 큼)
- On-device 추론 latency: **~1.5초/세그먼트** (Xcover 5, 노트 — Pixel 7 대비 느림)

### 누적 해결된 버그
1. **VAD Silero context 누락**: prob 항상 ~0.001 → 정상 동작
2. **DANGER 단일 spike**: 오탐 33% → 연속 2회 요구로 제거
3. **Padding mismatch**: training zero-pad vs runtime repeat-pad → 통일
4. **Narrowband preprocessor 적용**: 감지력 56→7% 하락 → runtime 비활성화
5. **ONNX dynamo export**: ORT Android 호환 안 됨 → legacy path 강제

### 알려진 한계
- **스피커 재생 TTS 감지 미흡**: 실기기 ~20%. 훨씬 많은 데이터 + acoustic augmentation 필요
- **영어 TTS 2개 회귀**: `edge_en_neerja_in`, `edge_en_jenny` (한국어 중심 배포엔 허용 가능)
- **통화 오디오 경로(VOICE_COMMUNICATION) 미검증**: 실제 보이스피싱 공격 벡터. 별도 라이브 통화 테스트 필요
- **speaker-holdout 규모 작음**: jeong 1명만 완전 홀드아웃, 다양성 부족

## Do NOT
- `model-training/data/` 이하 **대용량 데이터를 git에 커밋하지 말 것** (real/, fake/, real_extra_iphone/, mobile_fake_capture/, train_v*.csv, val_v*.csv, test_*.csv)
- `model-training/weights/finetuned*/` 체크포인트(.pth)를 커밋하지 말 것 — 재현 가능한 산출물
- `.env`, 자격증명 파일을 커밋하지 말 것
- API 키를 코드에 하드코딩하지 말 것
- `torch.onnx.export` 호출 시 `dynamo=False` 명시 빠뜨리지 말 것 — 기본값 Android 런타임 비호환
- `NarrowbandPreprocessor`를 런타임에 켜지 말 것 — v4 모델 분포 밖으로 밀어냄 (레거시 호환 목적 외)
