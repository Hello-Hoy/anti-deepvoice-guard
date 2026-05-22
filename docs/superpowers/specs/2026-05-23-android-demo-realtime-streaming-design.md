# 안드로이드 데모 — GPT-SoVITS 음성 추가 + 5지표 실시간 스트리밍 표시 — 설계

- 날짜: 2026-05-23
- 대상: `android-app/` (package `com.deepvoiceguard.app`)
- 관련: [[project_practicum_competition]] 데모 시연, [[jeonhyunmoo-voice-extraction]](GPT-SoVITS 클론 음성)

## 1. 목표

GPT-SoVITS로 생성한 전현무 클론 음성 2개(`GPT-SoVITS/outputs/jhm/demo/01_call.wav`, `02_menu.wav`)를
안드로이드 앱 데모에 추가하고, 각 데모를 재생할 때 **5개 지표가 재생과 동기되어 실시간으로 점진 갱신**되도록
기존 데모 화면을 개편한다:
1. 실시간 음성 VAD (발화 활성)
2. 실시간 딥보이스 점수
3. 실시간 STT 전사
4. 실시간 피싱 점수
5. 통합 위협 (위 결과의 실시간 종합)

**앱 컨셉**(통화 실시간 탐지)에 맞춰, 현재처럼 결과를 끝에 한 번 보여주는 게 아니라 통화가 진행되며
위협도가 실시간으로 변하는 모습을 보여준다.

## 2. 현재 상태 (조사 결과)

- 데모 기능이 이미 존재: `ui/screens/DemoScreen.kt`(하드코딩 `demoScenarios` 7개) + `inference/DemoAnalysisPipeline.kt`.
- 현재 `analyze()`는 **일회성**: WAV 전체 로드 → AASIST 세그먼트 추론(집계) → 사전 전사 .txt 로드 → 피싱 → 통합. 단일 `DemoResult` 반환.
- 화면은 재생 progress + 전사 글자 점진표시만 "실시간"이고, 딥보이스/피싱/위협은 끝(REVEALING)에 카드로 1회 표시.
- **VAD는 데모 경로에서 미사용**(요청 5개 중 누락). `audio/VadEngine.kt`(silero_vad.onnx, `process(512f)→Float`, reset/close) 존재.
- 엔진은 화면에서 `remember`로 직접 생성(ViewModel 없음), Hilt 미주입. fail-closed 정책(asset/추론 실패 시 예외 노출).
- 데모 자산: `app/src/main/assets/demo/demo_01..07`. 포맷 = **16kHz mono 16-bit PCM WAV**, `tools/preprocess_demo_wav.py`로 100–3000Hz 협대역 전처리됨.
- 단위테스트: `app/src/test/.../{CombinedThreatAggregator,DetectionAggregator,PhishingKeywordDetector}Test.kt` 등 JUnit.

## 3. 접근 (확정: A안)

**타임라인 사전계산 → 재생 동기 재생.** 시나리오 시작 시 무거운 계산(VAD 프레임별 + AASIST 세그먼트별 +
구간별 피싱/통합)을 한 번에 끝내 **타임라인**으로 만들고, 재생 중 ticker가 현재 재생위치의 프레임을 골라
5개 지표를 갱신. 결정적·매끄럽고 재생과 완벽 동기. 기존 엔진/헬퍼 전부 재사용.
(대안 B 실시간 동시추론=싱크 불안정, C 하이브리드=일관성↓ → 기각.)

## 4. 컴포넌트 설계

### 4.1 데이터 모델 — `inference/DemoTimeline.kt` (신규)
```kotlin
data class DemoFrame(
    val offsetMs: Int,                       // 100ms 격자
    val vadActive: Boolean,                  // 그 시점 발화 중
    val fakeScore: Float,                    // 그 시점까지 누적 딥보이스(0..1)
    val deepfakeLevel: ThreatLevel,          // 그 시점 딥보이스 단계
    val phishingScore: Float,                // 그 시점까지 드러난 전사 기준(0..1)
    val threatLevel: CombinedThreatLevel,    // 그 시점 통합 위협
    val transcriptChars: Int,                // 그 시점까지 보여줄 전사 길이
)
data class DemoTimeline(
    val durationMs: Int,
    val frames: List<DemoFrame>,
    val transcript: String,
    val finalResult: DemoResult,             // 끝 요약(기존 DemoResult 재사용)
)
```

### 4.2 파이프라인 — `DemoAnalysisPipeline.buildTimeline(audioAsset, transcriptAsset): DemoTimeline` (신규 메서드)
- 기존 `analyze()`의 WAV 로드·세그먼트·AASIST·피싱·통합 로직 재사용.
- 추가 계산:
  1. **VAD 타임라인**: 16k 오디오를 512샘플(32ms) 프레임으로 순회, `VadEngine.process()`로 speechProb → 프레임별 활성(>0.5).
  2. **딥보이스 step 타임라인**: 세그먼트별 `detect()`를 `DetectionAggregator`에 누적하며 각 세그먼트의 시간범위(끝 시각)에 누적 fakeScore/level 기록.
  3. **100ms 격자 frame 생성**: 각 offset t에서 — vadActive(VAD 타임라인), fakeScore/level(t 이하 마지막 세그먼트 step의 누적값, step-hold), transcriptChars = round(len * t/dur), phishingScore/통합 = 그 시점까지 드러난 전사로 `phishingDetector.analyze()` + `CombinedThreatAggregator.combine()`.
- 순수 파생 로직(VAD prob 배열 + 세그먼트 step + 전사 → frames)은 `DemoTimeline.kt`의 순수 함수로 분리해 단위테스트.
- 기존 `analyze()`는 보존(호환). 화면은 `buildTimeline()` 사용.

### 4.3 화면 — `DemoScreen.kt` (개편)
- 시나리오 시작 → `buildTimeline()`(IO) 진행 중 "분석 준비" 스피너 → 완료 후 MediaPlayer 재생 시작.
- 기존 ticker(60ms)가 재생 위치(ms) → `frames`에서 해당 `DemoFrame` 선택(이진탐색/선형) → 아래 5개 카드를 실시간 갱신:
  1. **VAD 카드**: 🎙 발화중/⏸ 무음 인디케이터 + 활성 펄스(기존 WaveformView 진행색과 결합 가능).
  2. **딥보이스 카드**: `fakeScore` 게이지(>0.7 빨강), 단계 라벨. 재생 중 값이 오르내림.
  3. **STT 카드**: 기존 전사 점진표시 + 키워드 하이라이트(기존 헬퍼 재사용).
  4. **피싱 카드**: `phishingScore` 게이지 + 드러난 키워드(점진).
  5. **통합 위협 배너**: `threatLevel` 색상 배너(기존 `threatColor`/색 매핑 재사용), 재생 중 단계 전이.
- 종료(DONE) 시 `finalResult` 최종 요약 카드 표시(기존 `DemoResultCard` 재사용/축약).
- 동시성·생명주기(Mutex/generation 토큰/DisposableEffect/MediaPlayer 해제)는 기존 구조 유지.

### 4.4 데모 자산 (신규)
- `tools/preprocess_demo_wav.py`의 `decode_to_16k_mono`+`preprocess`+`to_int16_wav` 재사용해
  `01_call.wav`/`02_menu.wav`(32k) → **16k mono + 100–3000Hz 협대역** → `assets/demo/demo_08.wav`, `demo_09.wav`.
  (전화 사칭 시나리오라 협대역 전처리 적합. 스크립트에 GPT-SoVITS 입력용 매핑 추가 또는 보조 함수.)
- 전사 `.txt` 2개: faster-whisper 전사 정제본.
  - 08: "안녕하세요. 저 전현무입니다. 다름이 아니라 저희가 지금 전현무계획 촬영 중이거든요. 지금 스태프들이랑 다 같이 이동 중인데 촬영 가능할까요?"
  - 09: "저희 스태프들까지 다 해서 총 25명이고요. 방송 나가시기도 하고, 저희가 찐맛집만 골라 다니잖아요. 메뉴 중에 제일 유명한 고기 세트로 바로 세팅 좀 부탁드릴게요."
- `demoScenarios`에 2줄 추가: id 8/9, 제목(예: "전현무 사칭 촬영요청(AI)", "전현무 사칭 메뉴요청(AI)"), 예상=**DANGER**(AI 음성→딥보이스 高, 피싱키워드 低).

## 5. 검증 (empirical-first, 필수)

- **딥보이스 점수 사전 검증**: 전처리된 demo_08/09를 onnxruntime로 `aasist.onnx`에 통과시켜 fakeScore가 충분히 높은지(예: ≥0.7) 확인. 낮으면 전처리 강도 조정 또는 사실대로 보고(데모 오인 방지). 데모의 핵심이 "AI 음성 탐지"이므로 fake로 안 잡히면 데모 실패.
- 통합 위협이 재생 동안 SAFE→…→DANGER로 자연스럽게 전이하는지 수동 확인.

## 6. 에러 핸들링

- 기존 fail-closed 유지: 오디오/전사/엔진 누락·추론 실패 시 예외 표시.
- VAD 엔진 초기화 실패 시: VAD 지표만 "N/A"로 강등하고 나머지 4개는 정상 진행(딥보이스 데모 자체는 살림). 조용한 실패 금지.

## 7. 테스트

- 순수 타임라인 파생 로직 단위테스트(`app/src/test`): VAD prob 배열→frame.vadActive, 세그먼트 step 누적값의 step-hold 선택, transcriptChars=progress 비례, 빈/짧은 입력 경계.
- 기존 테스트 회귀(no break). Compose UI는 기존처럼 수동 검증.

## 8. 범위 제외 (YAGNI)

- 실제 on-device STT를 데모 파일에 구동(SpeechRecognizer는 마이크용) — 기존처럼 사전 전사 점진표시로 대체.
- 라이브 통화 경로(AudioCaptureService) 변경 없음 — 데모 한정.
- 데모 외 화면/네비게이션/DI 변경 없음.

## 9. 리스크

- 전처리 후에도 GPT-SoVITS 클론이 fake로 안 잡힐 수 있음 → §5 사전 검증으로 선제 확인.
- frame 격자(100ms)·세그먼트(~2s hop)로 딥보이스는 ~2s마다 계단식 갱신 — "실시간 느낌"엔 충분하나 매우 매끄럽진 않음(허용).
