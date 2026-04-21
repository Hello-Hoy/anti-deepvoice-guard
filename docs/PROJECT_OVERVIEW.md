# Anti-DeepVoice Guard — 프로젝트 개요

> **한 줄 요약**: 알뜰폰 사용자를 위한 **실시간 보이스피싱 + Deepfake 음성 동시 탐지** Android 앱.
> AASIST 딥러닝 모델(on-device)로 AI 합성 음성을 식별하고, Korean STT + 키워드 사전으로 피싱 문구를 함께 분석해 **통합 위협 레벨**을 사용자에게 알림.

---

## 1. 문제 정의 — 왜 만들었는가

### 배경
- **Deepfake 음성 기술**의 발전으로 가족·지인을 사칭한 보이스피싱이 증가 중.
- 통신 3사(SKT/KT/LGU+)는 자체 보이스피싱 차단 서비스를 운영하나, **알뜰폰 사용자는 사각지대**.
- 삼성 Galaxy S26부터 보이스피싱 방지 기능이 탑재되지만, **딥보이스(AI 합성 음성) 탐지는 미포함**.

### 우리의 접근
**두 개의 독립된 탐지 축을 결합**해 보완적 신호를 얻는다.

```mermaid
graph LR
    A[통화 음성] --> B[딥보이스 탐지<br/>AASIST]
    A --> C[피싱 키워드 탐지<br/>STT + 사전]
    B --> D[통합 위협 분석<br/>5단계 레벨]
    C --> D
    D --> E[사용자 알림]

    style B fill:#e1f5ff,stroke:#0277bd
    style C fill:#fff3e0,stroke:#e65100
    style D fill:#f3e5f5,stroke:#6a1b9a
```

| 단일 신호의 한계 | 결합 시 효과 |
|---|---|
| 딥보이스 단독: 실제 사람이 피싱하면 놓침 | AI 음성이 아니더라도 피싱 문구로 탐지 |
| 키워드 단독: 피싱 단어를 쓰지 않는 정교한 대본 대응 불가 | AI 음성이라는 사실 자체가 강력한 경고 신호 |
| **둘 다 탐지되면 = CRITICAL** (최고 경고) | False Positive 감소, 확신도 상승 |

---

## 2. 전체 시스템 아키텍처

### 2.1 컴포넌트 수준 (Bird's-eye view)

```mermaid
graph TB
    subgraph Phone["📱 Android 단말 (On-Device)"]
        direction TB
        MIC[AudioRecord<br/>16kHz mono PCM]
        RB[RingBuffer 30s]
        VAD[Silero VAD<br/>2.3MB ONNX]
        SEG[SegmentExtractor<br/>64600 samples ≈ 4s]
        NBP[NarrowbandPreprocessor<br/>100-3000Hz band-pass]
        AASIST[AASIST-L<br/>107K params / 644KB ONNX]
        STT[SpeechRecognizer<br/>Korean]
        TB[TranscriptionBuffer]
        PKD[PhishingKeywordDetector<br/>200+ 키워드 + n-gram]
        CTA[CombinedThreatAggregator]
        NOTIF[NotificationHelper]
        ROOM[(Room DB<br/>AES-256-GCM)]
    end

    subgraph Server["☁️ FastAPI 서버 (Optional)"]
        API[/v1/predict · v2/telemetry/]
        SRV_MODEL[AASIST PyTorch<br/>서버 측 재추론/검증]
        SYNC[키워드 사전<br/>원격 동기화]
    end

    MIC --> RB
    RB --> VAD
    VAD --> SEG
    SEG --> NBP
    NBP --> AASIST
    RB --> STT
    STT --> TB
    TB --> PKD
    AASIST --> CTA
    PKD --> CTA
    CTA --> NOTIF
    CTA --> ROOM

    CTA -.선택적 업로드.-> API
    API --> SRV_MODEL
    API --> SYNC
    SYNC -.갱신.-> PKD

    style AASIST fill:#e1f5ff,stroke:#0277bd,stroke-width:2px
    style PKD fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style CTA fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

### 2.2 레포지토리 구조

```
anti-deepvoice-guard/
├── android-app/          ← Kotlin + Jetpack Compose (메인 산출물)
├── model-training/       ← PyTorch 학습, ONNX 변환 파이프라인
├── model-server/         ← FastAPI 서버 (하이브리드 옵션)
├── test-samples/         ← 검증용 오디오 (real / fake / fake-diverse / Demo)
├── tools/                ← 오디오 전처리, TTS 생성, 스코어링 스크립트
└── docs/                 ← 문서 (본 문서 포함)
```

---

## 3. 🔵 Deepvoice 탐지 관점 (AASIST)

### 3.1 모델 선정 이유
**AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks, ICASSP 2022)

- **Raw waveform 입력** — MFCC/spectrogram 수작업 feature 없이 end-to-end.
- **Graph Attention** — 시간(temporal) × 주파수(spectral) 두 축의 그래프 attention을 통합.
- **경량화** — AASIST-L 버전은 파라미터 **107K**, ONNX 파일 **644KB** → 단말 추론 가능.

### 3.2 On-Device 추론 파이프라인

```mermaid
flowchart LR
    MIC[🎤 마이크<br/>16kHz PCM] --> BUF[RingBuffer<br/>최근 30초 보관]
    BUF --> VAD{Silero VAD<br/>음성 구간?}
    VAD -->|No speech| SKIP[배터리 절약<br/>추론 스킵]
    VAD -->|Speech| SEG[Segment 추출<br/>정확히 64600 samples]
    SEG --> NBP[NarrowbandPreprocessor<br/>100-3000Hz band-pass<br/>폰 마이크 wideband 보정]
    NBP --> ONNX[AASIST ONNX<br/>Runtime Android 1.17]
    ONNX --> SOFTMAX[Softmax<br/>index 0 = spoof<br/>index 1 = bonafide]
    SOFTMAX --> AGG[DetectionAggregator<br/>슬라이딩 윈도우 평균]
    AGG --> RESULT[ThreatLevel<br/>SAFE/CAUTION/WARNING/DANGER]

    style VAD fill:#fff9c4,stroke:#f57f17
    style NBP fill:#c8e6c9,stroke:#2e7d32
    style ONNX fill:#e1f5ff,stroke:#0277bd
```

### 3.3 핵심 설계 결정

| 결정 | 이유 |
|---|---|
| **2단계 파이프라인 (VAD → AASIST)** | VAD가 무음·노이즈 구간을 먼저 걸러내 AASIST 호출 횟수 ↓ → 배터리 효율 |
| **NarrowbandPreprocessor (신규 도입)** | 폰 마이크는 wideband(~8kHz) 신호를 수집하나, AASIST 학습 분포는 narrowband(전화 품질). 전처리 없이 입력하면 **real voice를 FAKE로 오판**. 100-3000Hz band-pass로 도메인 매칭. |
| **Room DB + AES-256-GCM** | 탐지 이력을 기기 내부에만 저장, 암호화. 프라이버시 보호. |
| **audioSnapshot 원자 결합** | 탐지 이벤트와 증거 audio를 emit 시점에 한 묶음으로 저장 → STT 지연으로 인한 시간 drift 방지 |

### 3.4 학습 데이터 & 성능

```mermaid
graph LR
    subgraph Real["Real 음성 9,176개"]
        R1[AIHub Korean 자유발화]
    end
    subgraph Fake["Fake 음성 9,184개"]
        F1[Multi-TTS 생성<br/>edge-tts, gTTS 등]
        F2[XTTS v2 Voice Cloning]
        F3[전화 품질 augmentation<br/>대역폭 제한·양자화·패킷 손실]
    end
    Real --> FT[Fine-tuning<br/>PyTorch MPS/CUDA]
    Fake --> FT
    FT --> ONNX_OUT[aasist.onnx<br/>Android assets 배포]

    style Real fill:#c8e6c9
    style Fake fill:#ffcdd2
```

| 테스트 | 정확도 |
|---|---|
| Real WAV | **100%** (3/3) |
| Real MP3 | **100%** (10/10) |
| Fake 기본 | **100%** (4/4) |
| Fake 다양한 TTS | **94%** (16/17) |
| **종합** | **97%** (33/34) · val_acc 92.95% |
| **추론 latency** | **~100ms** per 4초 세그먼트 (Pixel 7) |

---

## 4. 🟠 STT 기반 보이스피싱 키워드 탐지

### 4.1 왜 별도 축인가
- Deepfake 탐지만으로는 **실제 사람이 대본을 읽는 피싱**을 잡지 못함.
- 한국어 보이스피싱은 **패턴화된 어휘·구문**을 반복 사용 → 키워드 사전 + n-gram 매칭으로 효과적 탐지.

### 4.2 탐지 파이프라인

```mermaid
flowchart LR
    AUDIO[🎤 Audio] --> STTE[Android<br/>SpeechRecognizer]
    STTE --> |Korean 전사| TB[TranscriptionBuffer<br/>최근 N초 텍스트]
    TB --> KN[KoreanNormalizer<br/>조사·어미·동의어<br/>정규화]
    KN --> PKD{PhishingKeywordDetector}

    PKD --> KW[키워드 사전<br/>200+ entries]
    PKD --> PM[PhraseMatcher<br/>bigram · trigram]

    KW --> SCORE[위협 점수]
    PM --> SCORE
    SCORE --> LEVEL[PhishingThreatLevel<br/>NONE/LOW/MEDIUM/HIGH]

    style STTE fill:#fff3e0
    style KN fill:#ffe0b2
    style PKD fill:#ffcc80
```

### 4.3 구성 요소

| Component | 역할 |
|---|---|
| `SttCapabilityChecker` | 기기에 한국어 STT가 가능한지 사전 검증 (capability-gated) |
| `GoogleSttEngine` | Android SpeechRecognizer 래퍼, 연속 전사 |
| `TranscriptionBuffer` | 시간-정렬 전사 버퍼 (최근 발화 유지) |
| `KoreanNormalizer` | "수사관입니다" ↔ "수사관이에요" 등 어미 변형 흡수 |
| `PhishingKeywordDictionary` | 카테고리별 200+ 키워드 (검찰/금감원/대출/계좌이체 등) |
| `PhraseMatcher` | bigram/trigram 구문 매칭 — "계좌가 연루", "보안 카드 번호" 등 |
| `PhishingKeywordDetector` | 점수 합산 + 임계값 기반 `PhishingThreatLevel` 산출 |

### 4.4 성능
- 정상 금융 대화 10건 + 피싱 대본 15건 테스트 → **F1 > 0.75** 보장 (`PhishingKeywordDetectorTest`).

---

## 5. 🟣 통합 위협 분석 (CombinedThreatAggregator)

### 5.1 의사결정 테이블
두 채널의 결과를 **독립적으로 산출한 뒤 결합**.

```mermaid
graph TD
    DV[딥보이스 결과<br/>SAFE/CAUTION/WARNING/DANGER] --> CTA{CombinedThreat<br/>Aggregator}
    PH[피싱 결과<br/>NONE/LOW/MEDIUM/HIGH] --> CTA
    STT[STT 상태<br/>LISTENING / UNAVAILABLE] --> CTA

    CTA --> L1[🟢 SAFE]
    CTA --> L2[🟡 CAUTION]
    CTA --> L3[🟠 WARNING]
    CTA --> L4[🔴 DANGER]
    CTA --> L5[⚫ CRITICAL]

    style L5 fill:#b71c1c,color:#fff
    style L4 fill:#d32f2f,color:#fff
    style L3 fill:#f57c00,color:#fff
    style L2 fill:#fbc02d
    style L1 fill:#388e3c,color:#fff
```

### 5.2 결합 규칙 (주요 행)

| 딥보이스 | 피싱 | STT | 통합 결과 |
|---|---|---|---|
| DANGER | High | OK | **⚫ CRITICAL** ← AI + 피싱 양쪽 확증 |
| DANGER | Low/None | OK | 🔴 DANGER |
| DANGER | — | 불가 | 🔴 DANGER (딥보이스만 적용) |
| WARNING | High | OK | 🔴 DANGER (상향) |
| SAFE | High | OK | 🟠 WARNING |
| SAFE | Medium/Low | OK | 🟡 CAUTION |
| SAFE | None | OK | 🟢 SAFE |

> 실제 구현: `android-app/.../inference/CombinedThreatAggregator.kt` (13행 decision table)

---

## 6. 실시간 데이터 플로우 (Sequence)

```mermaid
sequenceDiagram
    participant U as 👤 사용자
    participant M as 🎤 AudioRecord
    participant RB as RingBuffer
    participant V as Silero VAD
    participant A as AASIST
    participant S as SpeechRecognizer
    participant P as PhishingDetector
    participant C as Combined<br/>Aggregator
    participant N as Notification

    U->>M: 통화 시작 (스피커폰)
    loop 매 20ms 프레임
        M->>RB: PCM 16kHz 프레임 push
    end

    par 딥보이스 경로
        RB->>V: 4초 윈도우
        V-->>A: 음성 구간만 전달
        A->>A: Narrowband 전처리 + 추론
        A-->>C: DeepfakeResult
    and STT 경로
        RB->>S: 연속 오디오
        S-->>P: 한국어 전사 텍스트
        P->>P: 정규화 + 키워드/구문 매칭
        P-->>C: PhishingResult
    end

    C->>C: 의사결정 테이블 적용
    alt 위협 레벨 ≥ WARNING
        C->>N: 알림 발송 (dedupe + 쿨다운)
        N-->>U: "⚠️ 의심 통화 감지"
    end
    C->>C: Room DB에 암호화 저장
```

---

## 7. 모델 학습 & 배포 파이프라인

```mermaid
graph LR
    subgraph Data["📦 데이터 수집"]
        A1[AIHub Real<br/>9,176건]
        A2[Multi-TTS 생성<br/>generate_multi_tts.py]
        A3[XTTS v2 Cloning<br/>generate_clone_data.py]
        A4[전화 품질 Aug<br/>bandwidth/quantize/loss]
    end

    subgraph Train["🧠 학습 (PyTorch)"]
        T1[fine_tune.py<br/>MPS M4 Pro<br/>또는 CUDA RTX 4080 Super]
        T2[weights/finetuned/<br/>aasist_best.pth]
    end

    subgraph Deploy["🚀 배포"]
        D1[convert_to_onnx.py<br/>embedded weights]
        D2[android-app/assets/<br/>aasist.onnx]
        D3[model-server/<br/>서버 재추론/검증]
    end

    A1 --> T1
    A2 --> T1
    A3 --> T1
    A4 --> T1
    T1 --> T2
    T2 --> D1
    D1 --> D2
    T2 --> D3

    style A1 fill:#c8e6c9
    style A2 fill:#ffcdd2
    style A3 fill:#ffcdd2
    style A4 fill:#ffcdd2
    style T2 fill:#e1f5ff
    style D2 fill:#f3e5f5
```

---

## 8. 기술 스택 요약

| Layer | Technology |
|---|---|
| **AI 모델** | AASIST-L (107K params, 644KB ONNX) |
| **VAD** | Silero VAD v5 (2.3MB ONNX) |
| **STT** | Android SpeechRecognizer (Korean) |
| **피싱 탐지** | 200+ 키워드 사전 + bigram/trigram + 한국어 정규화 |
| **Android** | Kotlin · Jetpack Compose · Material3 · Hilt · Room |
| **Inference** | ONNX Runtime Android 1.17 |
| **Storage** | Room DB v3 + AES-256-GCM EncryptedFile |
| **Server** | FastAPI + PyTorch |
| **학습 환경** | M4 Pro 24GB (MPS) · RTX 4080 Super (CUDA) |

---

## 9. 프로젝트의 차별점

```mermaid
mindmap
  root((Anti-DeepVoice<br/>Guard))
    경량 On-Device
      AASIST-L 644KB
      Pixel 7 기준 100ms 추론
      오프라인 동작
    이중 신호 융합
      딥보이스 탐지
      피싱 키워드 탐지
      5단계 통합 레벨
    알뜰폰 타겟
      통신3사 사각지대
      규제 샌드박스 전제
      SDK 납품 BM
    프라이버시
      기기 내 암호화 저장
      AES-256-GCM
      서버 업로드는 선택적
    검증 프로세스
      Claude agents 리뷰
      Codex GPT-5.4 교차검증
      실 오디오 18000+ 건 검증
```

| 구분 | 기존 솔루션 | Anti-DeepVoice Guard |
|---|---|---|
| 대상 사용자 | 통신3사 가입자 | **알뜰폰 포함 전 Android 사용자** |
| 딥보이스 탐지 | ❌ 없음 (Galaxy S26 포함) | ✅ AASIST on-device |
| 피싱 키워드 탐지 | 부분 지원 | ✅ STT + 200+ 키워드 + n-gram |
| 배포 형태 | OEM 번들 | **SDK 형태, 알뜰폰 사업자 B2B** |
| 추론 위치 | 일부 서버 의존 | **On-device 우선, 서버는 옵션** |

---

## 10. 현재 상태 & 한계

### ✅ 완료
- AASIST v4 fine-tuning (val_acc 92.95%, 전체 97%)
- Android 앱 라이브 마이크 파이프라인 + Demo 시나리오 6종
- 통합 위협 레벨 13행 의사결정 테이블
- Room DB + 암호화 저장, 알림 dedupe/쿨다운
- Narrowband 도메인 매칭 전처리
- FastAPI 서버 v1/v2 API, 테스트 커버리지

### ⚠️ 알려진 한계
| 항목 | 내용 |
|---|---|
| **Android 10+ 통화 캡처 제약** | 상대방 음성 직접 캡처 불가 (VOICE_CALL/VOICE_DOWNLINK 차단) → **스피커폰 모드**에서만 양쪽 음성 수집 가능 |
| **에뮬레이터 마이크 문제** | Apple Silicon Mac + Android 에뮬레이터는 호스트 마이크 라우팅 이슈 → 라이브 테스트는 **실기기 필수** |
| **edge_ko_injoon TTS 오탐** | 특정 TTS 엔진에서 아직 탐지율 낮음 (21%) — 추가 학습 데이터로 개선 중 |

### 🎯 프랙티컴 경진대회 시연 방식
1. **Demo 탭** — 사전 녹음된 6개 시나리오 파일로 에뮬레이터에서 확실한 데모
2. **실기기 라이브** — USB 연결 실 Android 폰에서 스피커폰으로 다른 폰의 피싱 대본 재생 → 실시간 CRITICAL 알림
3. **사업 가정** — 실시간 통화 감지는 **알뜰폰 사업자가 규제 샌드박스를 통해 구현하는 방향**을 전제로 PoC

---

## 11. 검증 프로세스

모든 주요 변경은 **다중 AI 에이전트 교차 검증**을 거침:

```mermaid
graph LR
    CODE[코드 변경] --> R1[Claude code-reviewer<br/>컨벤션·보안]
    CODE --> R2[silent-failure-hunter<br/>에러 핸들링]
    CODE --> R3[pr-test-analyzer<br/>커버리지]
    CODE --> R4[type-design-analyzer<br/>타입 설계]
    R1 --> CODEX[Codex GPT-5.4<br/>교차 독립 검증]
    R2 --> CODEX
    R3 --> CODEX
    R4 --> CODEX
    CODEX --> TEST[실 오디오 샘플<br/>통합 검증]
    TEST --> MERGE[Merge]
```

---

## 부록 — 관련 문서

- **[README.md](../README.md)** — 빠른 시작 가이드
- **[docs/DEMO_TESTING_GUIDE.md](DEMO_TESTING_GUIDE.md)** — Demo 시나리오 시연 방법
- **[docs/REAL_DEVICE_TESTING.md](REAL_DEVICE_TESTING.md)** — 실기기 테스트 가이드
- **[ISSUE_SUMMARY_20260404.md](../ISSUE_SUMMARY_20260404.md)** — 법률/기술 이슈 요약
- **[ISSUE_SUMMARY_20260406.md](../ISSUE_SUMMARY_20260406.md)** — 비즈니스/기술 이슈 요약
- **AASIST 원 논문** — Jung et al., *"AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"*, ICASSP 2022

---

_작성 일자: 2026-04-19 · 프랙티컴 경진대회 출품작_
