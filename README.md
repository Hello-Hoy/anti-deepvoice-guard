# Anti-DeepVoice Guard

알뜰폰 사용자를 위한 **실시간 보이스피싱 + Deepfake 음성 탐지** Android 앱.
AASIST 딥러닝 모델 기반 on-device 추론으로 통화 음성의 AI 합성 여부를 판별하고,
Korean STT + 보이스피싱 키워드 분석으로 피싱 위협을 동시에 감지합니다.

**프랙티컴 경진대회 출품작** (2026).

## 핵심 기능

1. **Deepfake 탐지** — AASIST 모델 (v4 fine-tuned, 1.4MB ONNX) 실시간 on-device 추론
2. **보이스피싱 키워드 탐지** — Korean STT(SpeechRecognizer) + 200+ 키워드 사전 + n-gram 구문 매칭
3. **통합 위협 분석** — 딥보이스 + 피싱 결과를 5단계(SAFE/CAUTION/WARNING/DANGER/CRITICAL) 결합
4. **Demo 모드** — 6개 사전 시나리오 파일 기반 시연 (경진대회 용)
5. **하이브리드 서버** — 메타데이터/규칙 동기화 FastAPI v2 API (옵션)

## 아키텍처

```
[라이브 마이크 경로 — 실기기 전용]
AudioRecord(16kHz, MIC) → RingBuffer(30s) → ┬→ Silero VAD (v5 + context-concat) → SegmentExtractor(64600) → AASIST v4
                                            └→ SpeechRecognizer → TranscriptionBuffer → PhishingKeywordDetector
                                            AASIST + Phishing → CombinedThreatAggregator → 알림 + Room DB

[Demo 경로 — 파일 기반, 에뮬레이터/실기기 모두 가능]
WAV asset → DemoAnalysisPipeline(sliding-window AASIST) + 사전 전사본 → PhishingKeywordDetector → UI
```

> v4 모델은 raw PCM 입력 기준으로 학습되어, 라이브 경로에서 `NarrowbandPreprocessor`는 **런타임 비활성화** 상태 (Settings 토글에 레거시 호환 옵션으로 유지).

### 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| VAD → AASIST 2단계 파이프라인 | VAD가 speech 구간만 필터링 → AASIST 호출 최소화 → 배터리 효율 |
| **v4 fine-tune (mobile-mic 녹음 추가)** | 기존 AIHub 학습 분포가 폰 마이크 신호와 맞지 않아 실기기 한국어 real 33% 오탐 → 878개 iPhone real + 445개 mobile-captured fake 추가 학습으로 **실기기 오탐 0%** |
| **Silero VAD v5 context-concat (64 + 512)** | 공식 wrapper 규약대로 이전 프레임 64 샘플을 앞에 concat해 (1, 576) 입력. 이걸 빠뜨리면 VAD 확률 ~0.001에 고정됨 |
| **DANGER 연속 2회 이상 요구** | 실기기 AASIST 점수는 같은 실음성에도 단일 세그먼트 기준 널뛰어(0.02~0.97) 오탐 발생 → 연속 스파이크가 있어야만 DANGER 승격 |
| **Training repeat-pad + ONNX legacy path** | training(zero-pad) vs runtime(repeat-pad) 분포 mismatch 및 dynamo path ORT Android 비호환 이슈 해결 |
| NarrowbandPreprocessor **런타임 OFF** | v4는 raw PCM 분포로 학습됨. runtime 적용 시 감지력 56→7% 하락 확인 → null 고정 |
| **Demo WAV pre-filtered 저장** | 런타임 per-segment normalization이 TTS 구분 특징까지 지워 정확도 저하 → 디스크에 global-filtered WAV 저장이 정확도 최대 |
| Combined threat 5단계 | 딥보이스 + 피싱을 독립 채널로 본 뒤 의사결정 테이블로 최종 레벨 산출 (CRITICAL = 양쪽 모두 high) |
| `audioSnapshot` 이벤트 결합 | 탐지 이벤트와 증거 audio가 `emit` 시점에 원자적 묶음 저장 (STT 지연 drift 방지) |

## 모듈

| Module | 역할 | 상태 |
|--------|-----|------|
| `android-app/` | Kotlin + Jetpack Compose 앱 | 완성 |
| `model-server/` | FastAPI v1/v2 추론/telemetry 서버 | 완성 |
| `model-training/` | PyTorch 학습, ONNX 변환 도구 | 완성 |
| `tools/` | 오디오 전처리, TTS 생성, 스코어링 유틸리티 | 완성 |

## 빠른 시작

### A. 경진대회 시연 (컴퓨터 Demo)

1. 저장소 클론 + JDK 17 + Android Studio 설치
2. Android Studio에서 `android-app/` 열기
3. Pixel 7 API 34 에뮬레이터 생성
4. **Run ▶** (자동 빌드+설치)
5. 앱에서 **Demo 탭** → 6개 시나리오 중 선택 → **분석** 버튼
6. 예상 결과:

| # | 시나리오 | 기대 결과 |
|---|---------|---------|
| 1 | 일상 통화 | SAFE |
| 2 | TTS 일상 | DANGER (AI 음성) |
| 3 | 실제 사람 피싱 | WARNING (피싱 키워드) |
| 4 | TTS 피싱 | CRITICAL (AI + 피싱) |
| 5 | 은행 정상 안내 | SAFE |
| 6 | 보험 사기 | WARNING |

상세 가이드: **[docs/DEMO_TESTING_GUIDE.md](docs/DEMO_TESTING_GUIDE.md)**

### B. 실기기 라이브 마이크 테스트

Apple Silicon Mac + Android 에뮬레이터는 호스트 마이크 라우팅 문제로 AudioRecord가 동작하지 않습니다. 라이브 마이크는 **USB 연결한 실제 Android 폰**에서 테스트해야 합니다.

1. Android 폰의 **개발자 옵션 → USB 디버깅** ON
2. USB로 Mac 연결
3. Android Studio 상단에서 실기기 선택 → **Run ▶**
4. 앱에서 **RECORD_AUDIO 권한 허용** (첫 실행 시)
5. **Home 탭 → START**
6. 스피커폰으로 다른 폰에서 피싱 대본 재생 → 실시간 CRITICAL 알림

상세 가이드: **[docs/REAL_DEVICE_TESTING.md](docs/REAL_DEVICE_TESTING.md)**

### C. ONNX 모델 / 서버 (옵션)

```bash
# ONNX 변환 (이미 생성된 aasist.onnx가 포함되어 있어 재변환 불필요)
# v4 체크포인트 (mobile-mic 녹음 추가 학습) 기준
cd model-training && python convert_to_onnx.py \
    --model-path weights/finetuned_v4/aasist_best.pth \
    --output ../android-app/app/src/main/assets/aasist.onnx --verify

# 서버 실행
cd model-server && pip install -r requirements.txt
PYTHONPATH=.. uvicorn model_server.api.main:app --port 8000
```

## 기술 스택

| Layer | Technology |
|-------|-----------|
| AI Model | AASIST-L (107K params, 644KB ONNX) |
| VAD | Silero VAD v5 (2.3MB ONNX) |
| STT | Android SpeechRecognizer (Korean) — capability-gated |
| 피싱 탐지 | 200+ 키워드 사전 + bigram/trigram 구문 매칭 + 한국어 정규화 |
| Android | Kotlin, Jetpack Compose, Material3, Hilt, Room |
| Inference | ONNX Runtime Android 1.17 |
| Network | Retrofit + OkHttp |
| Storage | Room DB v3 + AES-256-GCM EncryptedFile |
| Server | FastAPI + PyTorch |

## 모델 성능 (v4, 2026-04 기준)

| Metric | Value |
|--------|-------|
| On-Device 추론 latency | ~100ms per 4s segment (Pixel 7), ~1.5초 (Xcover 5) |
| AASIST 정확도 (Demo 시나리오) | 6/6 (파일 입력 기준) |
| AIHub real 분류 | fake score 0.02~0.15 (오탐 없음) |
| 한국어 TTS fake 감지 | 0.87~1.00 fake score (모두 DANGER) |
| **실기기 한국어 real 오탐** | 33% → **0%** (홀드아웃 jeong 50개 평균 0.014) |
| **Mobile-captured fake 감지** | 56% (Python 445개 홀드아웃, >0.5 fake) |
| 피싱 탐지 F1 | 정상 금융 10 + 피싱 15 → F1 > 0.75 |

## 알려진 한계 / 다음 개선 후보

- **스피커→마이크 공기 경로 재생 TTS 감지 약함**: 실기기 ~20%. 더 많은 mobile-fake 데이터 + acoustic augmentation (reverb/room IR/speaker 응답 시뮬) 필요.
- **영어 TTS 2건 회귀** (`edge_en_neerja_in`, `edge_en_jenny`): v3→v4에서 감지력 하락. 한국어 중심 시나리오면 영향 제한적이나 다국어 제품엔 추가 영어 fake 데이터 필요.
- **통화 오디오 경로 (VOICE_COMMUNICATION source) 미검증**: 실제 보이스피싱 공격 벡터. 실 통화 시나리오 검증 필요.
- **Speaker-holdout 규모 작음**: `jeong` 1명만 완전 홀드아웃. 더 많은 다양한 화자로 검증해야 진짜 generalization 판정 가능.

## 테스트

```bash
# Android 단위 테스트
cd android-app && ./gradlew :app:testDebugUnitTest

# Demo 정확도 검증 (Python)
python tools/score_demo_samples.py

# 서버 미들웨어 pytest
cd .. && .venv/bin/python -m pytest model-server/tests/
```

포함 테스트:
- `NarrowbandPreprocessorTest` (10건): 필터 응답, 정규화, idempotency
- `AudioPipelinePreprocessorIntegrationTest` (4건): 파이프라인 통합, audioSnapshot 원본 유지
- `CombinedEventAudioSnapshotTest` (4건): 이벤트-오디오 원자 묶음 회귀
- `CombinedThreatAggregatorTest` (16건): 의사결정 테이블
- `PhishingKeywordDetectorTest`: F1 ≥ 0.75 보장
- `KoreanNormalizerTest`: 조사/어미/동의어 처리
- `NotificationHelperTest`: 알림 dedupe + 쿨다운
- `test_body_size_middleware.py` (6건): HTTP middleware race conditions

## 문서

- **[Sprint 계획서 & 아키텍처 (Mermaid)](Anti-DeepVoice-Guard-프로젝트-가이드.pdf)**
- **[빌드 & 테스트 가이드 (Korean PDF)](Anti-DeepVoice-Guard-Build-Guide.pdf)**
- **[Demo 테스트 가이드](docs/DEMO_TESTING_GUIDE.md)**
- **[실기기 테스트 가이드](docs/REAL_DEVICE_TESTING.md)**
- **[이슈 요약 2026-04-04 (법률/기술)](ISSUE_SUMMARY_20260404.md)**
- **[이슈 요약 2026-04-06 (비즈니스/기술)](ISSUE_SUMMARY_20260406.md)**

## 기여 / 개발 이력

프로젝트 개발 과정에서 수행한 리뷰 및 검증:

- **Claude Code agents** (구현 + 1차 검토): code-reviewer, silent-failure-hunter, pr-test-analyzer, type-design-analyzer
- **Codex GPT-5.4 교차 검증**: 다중 adversarial review 사이클 (Sprint 1-4 각 단계마다)
- **ultrareview**: 전체 diff 대상 외부 에이전트 리뷰 (nit 수정 포함)
- **실제 오디오 샘플 검증**: AIHub real 9,170 + multi-TTS/clone fake 8,684

## 라이센스

Research/education용. 경진대회 시연 이후 상용 사용 시 AASIST/Silero VAD 원 저자 라이선스 확인 필요.

## Based On

[deep-fake-audio-detection](https://github.com/clovaai/aasist) — AASIST 원 논문 구현
- AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (Jung et al., ICASSP 2022)
- RawBoost augmentation + DANN domain adaptation
