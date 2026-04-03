# Anti-DeepVoice Guard — Agent Instructions

## Project Overview
실시간 deepfake 음성 탐지 Android 앱. AASIST 모델 기반.
전화 수신 시 상대방 음성의 AI 생성 여부를 실시간 판별.

## Repository Structure
```
anti-deepvoice-guard/
├── model-training/        # 모델 아키텍처, fine-tuning, ONNX 변환
│   ├── models/AASIST.py   # AASIST 모델 (자기완결적)
│   ├── configs/            # 모델 config (aasist.json, aasist_l.json)
│   ├── weights/            # official / legacy / finetuned
│   ├── fine_tune.py        # Fine-tuning 스크립트
│   ├── convert_to_onnx.py  # PyTorch → ONNX 변환
│   ├── generate_multi_tts.py    # 다중 TTS fake 데이터 생성
│   └── generate_clone_data.py   # XTTS v2 음성 클로닝 fake 생성
├── model-server/          # FastAPI 추론 서버
│   ├── api/main.py         # FastAPI 앱, 모델 로드
│   ├── api/routes/detect.py # 탐지 엔드포인트
│   └── inference/aasist_model.py # PyTorch 추론 래퍼
├── android-app/           # Android 앱 (Kotlin + Jetpack Compose)
│   ├── app/src/main/java/com/deepvoiceguard/app/
│   │   ├── service/        # AudioCaptureService, AudioPipeline
│   │   ├── inference/      # OnDeviceEngine (ONNX), ServerEngine
│   │   ├── audio/          # RingBuffer, VadEngine, SegmentExtractor
│   │   ├── storage/        # Room DB, EncryptedStorage, Settings
│   │   └── ui/             # Compose screens (Home, History, Settings)
│   └── app/src/main/assets/ # ONNX 모델 파일
└── test-samples/          # 테스트 오디오 샘플
```

## Agent Roles & Workflow

### Claude Code Agents (주 구현)
| Agent | 역할 | 사용 시점 |
|-------|------|----------|
| code-reviewer | 코드 품질, 보안, 컨벤션 검토 | 주요 코드 작성 후 |
| silent-failure-hunter | 에러 핸들링 누락 탐지 | catch/fallback 코드 작성 후 |
| pr-test-analyzer | 테스트 커버리지 분석 | PR 생성 전 |
| type-design-analyzer | 타입 설계 검토 | 새 타입 도입 시 |
| code-simplifier | 코드 간결화, 가독성 개선 | 기능 구현 완료 후 |
| code-architect | 아키텍처 설계 | 새 기능 설계 시 |

### Codex GPT 5.4 (교차 검증)
Claude Code 구현 완료 후 독립적 검증 수행:

```bash
# 사용법 — Claude Code 세션 내에서 codex-rescue 에이전트로 위임
# 또는 별도 터미널에서 직접 실행:

# 1. 코드 리뷰 (버그, 보안, 정확성)
codex "review <file_path> for bugs, security issues, and correctness. Focus on: error handling, edge cases, resource leaks."

# 2. Android 코드 검증
codex "verify that <component> follows Android best practices: lifecycle management, memory leaks, thread safety, permission handling."

# 3. 모델 추론 코드 검증
codex "verify the ONNX inference pipeline in <file> correctly handles: input preprocessing, output interpretation, resource cleanup."

# 4. 통합 테스트 케이스 생성
codex "generate comprehensive test cases for <component> covering: normal flow, edge cases, error conditions, concurrency."
```

### 검증 프로세스 (필수)
```
구현 완료
  ↓
[1] Claude code-reviewer 에이전트 실행
  ↓
[2] Claude silent-failure-hunter 에이전트 실행
  ↓
[3] Codex GPT 5.4 교차 검증
  ↓
[4] 실제 테스트 실행 (단위 + 통합)
  ↓
검증 완료 → 커밋
```

## For Code Review Agents
- AASIST 모델: `model-training/models/AASIST.py`
- 모델 설정: `model-training/configs/aasist.json`
- 공식 가중치: `model-training/weights/official/AASIST.pth`
- Fine-tuned 가중치: `model-training/weights/finetuned/aasist_best.pth`

## For Testing Agents
1. **모델 변환 검증**: `model-training/convert_to_onnx.py` — ONNX 변환 및 출력 비교
2. **서버 API 테스트**: `curl -F "audio=@test-samples/real/real_01.wav" http://localhost:8000/api/v1/detect`
3. **추론 정확도**: test-samples/ 디렉토리의 real/fake 샘플로 검증
4. **Android 빌드**: `cd android-app && ./gradlew assembleDebug`

## Key Technical Decisions
- SincConv의 band_pass 필터를 고정 nn.Parameter로 변환하여 ONNX 호환
- DANN (domain_classifier, ReverseLayerF) 제거 — 추론 전용
- forward() 출력에 softmax 적용하여 확률값 반환
- ASVspoof convention: index 0 = spoof/fake, index 1 = bonafide/real
- Device 우선순위: MPS (M4 Pro) > CUDA (NVIDIA) > CPU
- 오디오 로딩: soundfile 1차 → librosa(ffmpeg) 2차 폴백
- Android 통화 오디오: 스피커폰 모드 마이크 캡처 방식 (1차)

## Android Architecture
- **Audio Pipeline**: AudioRecord(16kHz) → RingBuffer(30s) → VAD(Silero) → SegmentExtractor → AASIST → Aggregator
- **Foreground Service**: TYPE_MICROPHONE, 알림 채널 2개 (모니터링 + 경고)
- **ThreatLevel**: SAFE → CAUTION(avg>0.6) → WARNING(3연속>0.7) → DANGER(단일>0.9)
- **Storage**: Room DB + AES-256-GCM 암호화 오디오 + 7일 보관 + 100MB 쿼터
