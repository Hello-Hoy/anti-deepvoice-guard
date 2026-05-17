## 0. 한 장 요약 (TL;DR)

```mermaid
flowchart LR
    A["💡 BM 가설<br/>알뜰폰용 실시간<br/>딥보이스 탐지"] --> B["🧱 규제·기술 벽<br/>통신3사·삼성만<br/>통화음성 접근"]
    B --> C["📜 우회 전략<br/>규제 샌드박스<br/>가정"]
    C --> D["📱 MVP 개발<br/>AASIST + STT<br/>Android 앱"]
    D --> E1["🎬 시연 1차<br/>Demo 파일 재생<br/>+ STT + 탐지"]
    E1 -.시연 임팩트 부족.-> E2["🎤 시연 2차<br/>스피커폰→마이크<br/>on-device"]
    E2 -.마이크 필터로<br/>탐지율 급락.-> F["🔁 모델 원복<br/>+ 시연 방식<br/>재설계 필요"]
    F --> G{{"❓ 의사결정 필요<br/>1안 vs 2안"}}

    style A fill:#e3f2fd,stroke:#1565c0
    style B fill:#ffebee,stroke:#c62828
    style C fill:#fff8e1,stroke:#f9a825
    style D fill:#e8f5e9,stroke:#2e7d32
    style E1 fill:#f3e5f5,stroke:#6a1b9a
    style E2 fill:#f3e5f5,stroke:#6a1b9a
    style F fill:#fce4ec,stroke:#ad1457
    style G fill:#fff3e0,stroke:#e65100,stroke-width:3px
```

| 핵심 메시지 | 한 줄 |
|---|---|
| **우리가 풀려는 문제** | 알뜰폰 사용자도 보이스피싱 + 딥보이스 동시 보호받게 만들기 |
| **막힌 지점** | "실시간 통화 음성 직접 접근"은 통신3사·OEM만 합법/기술적으로 가능 |
| **우회 가정** | 규제 샌드박스 신청·승인을 전제로 PoC 진행 ([Notion 참조](https://www.notion.so/AIMBA6-31ca85c6667c809baf24daf4a84413a2)) |
| **시연 시도 1** | Demo 음원 재생 + STT 자막 + Fake Voice 점수 → 임팩트 부족 |
| **시연 시도 2** | 스피커폰 + 마이크로 라이브 분석 → **마이크 acoustic filter** 때문에 탐지율 급락 |
| **현재 상태** | AASIST 모델 원복 완료. 시연 시나리오 재설계 필요 |
| **결정 필요** | **1안: 피해 상황 영상** vs **2안: 가상 통화 + 앱 동작 영상** |

---

## 1. 본 시작 — 비즈니스 모델 가설

### 1.1 기술적 차별점 (계획)

- **AASIST-L** on-device 딥보이스 탐지 (107K params, 644KB ONNX)
- **Korean STT + 200+ 키워드 사전**으로 보이스피싱 키워드 동시 탐지
- 두 신호를 **CombinedThreatAggregator**가 결합해 **5단계 통합 위협 레벨** 산출
- 알뜰폰 사업자에 **SDK 형태 B2B 납품**

---

## 2. 부딪힌 벽 — "왜 우리는 익시오처럼 못 하는가"

### 2.1 익시오가 가능한 이유 ≠ 우리가 가능한 이유

```mermaid
flowchart TB
    subgraph LGU_path["✅ LGU+ 익시오 (가능)"]
        L1[LG U+ <br/>= 통신사업자] --> L2[삼성 전화앱과<br/>시스템 레벨 연동]
        L2 --> L3[CAPTURE_AUDIO_OUTPUT<br/>시스템 권한 보유]
        L3 --> L4[수화기 모드에서도<br/>상대 음성 직접 접근 OK]
    end

    subgraph our_path["❌ 알뜰폰 + 일반 앱 (불가)"]
        O1[알뜰폰 사업자<br/>= 별정통신사업자] --> O2[자체 IMS·망 없음<br/>망 임차만 함]
        O2 --> O3[일반 앱 권한만 가능<br/>CAPTURE_AUDIO_OUTPUT ❌]
        O3 --> O4[수화기 모드 음성 접근 불가<br/>스피커폰만 가능]
    end

    style L1 fill:#c8e6c9
    style L4 fill:#a5d6a7
    style O1 fill:#ffcdd2
    style O4 fill:#ef9a9a
```

### 2.2 법·기술 이중 차단

| 차원 | 차단 근거 |
|---|---|
| **법률** | 통신비밀보호법 제3조 — "누구든지 이 법에 의하지 아니하고는 전기통신의 감청을 하지 못한다." 상업적 통화 음성 분석은 원칙적으로 감청에 해당 |
| **기술** | Android OS는 `CAPTURE_AUDIO_OUTPUT`을 시스템 앱(OEM·정부 협력)에만 부여. 일반 앱은 통화 다운링크 직접 캡처 불가 |
| **사업자 지위** | 알뜰폰은 전기통신사업법상 **별정통신사업자** — 자체 통신설비가 없어 네트워크 레벨에서도 음성 접근 불가 |

> 결론: **일반 앱이 일반 단말 위에서 수화기 모드 통화 음성에 접근하는 길은 현재 닫혀 있다.**

---

## 3. 우리의 우회 전략 — 규제 샌드박스 가정

```mermaid
flowchart LR
    A[현재 법·기술 상태<br/>알뜰폰 일반 앱<br/>통화 음성 접근 불가] --> B[ICT 규제 샌드박스<br/>실증특례 신청]
    B --> C[승인 가정<br/>제한 시범사업으로<br/>실시간 통화 음성 분석 허용]
    C --> D[이 가정 위에서<br/>MVP 개발 진행]

    style A fill:#ffcdd2
    style B fill:#fff8e1
    style C fill:#dcedc8
    style D fill:#c5e1a5
```

- **참고**: 위 경로 검토 및 신청 가능성은 [Notion 문서](https://www.notion.so/AIMBA6-31ca85c6667c809baf24daf4a84413a2) 하위 페이지에 정리.
- **본 프랙티컴의 PoC는 이 가정 위에서 진행됨.**
- 시연·발표 시에도 "샌드박스 통과를 전제로 한 가까운 미래 PoC"라는 점을 명시할 필요.

---

## 4. MVP 개발 여정 — 두 번의 시연 피벗

```mermaid
timeline
    title TrueVoice Guard 시연 방식 변천
    Phase 0 : BM·기술 가설 정의 : 알뜰폰 SDK + AASIST + STT
    Phase 1 : Android MVP 구축 : DemoScreen / HomeScreen / 알림 / Room DB
    Phase 2 : 시연 1차(Demo 파일) : 6개 시나리오 사전 녹음 분석
            : "재생 + 결과 표시" 형태
            : 임팩트 부족 판단
    Phase 3 : 시연 2차(스피커폰 라이브) : 다른 폰에서 TTS 재생
            : 우리 폰 마이크가 듣고 on-device 분석
    Phase 4 : 마이크 필터 문제 발견 : 실기기 fake 탐지율 ~20%로 급락
            : 모바일 캡처 데이터로 fine-tune 시도
    Phase 5 : 모델 원복(현재) : AASIST v4 baseline으로 복귀
            : 시연 방식 재설계 필요
```

### 4.1 시연 1차 — Demo 파일 재생 방식

**구성:** `DemoScreen` 에서 사전 녹음된 6개 시나리오를 순차 재생하며, 같은 음원에 AASIST + STT 키워드 탐지를 적용해 결과를 표시.

```mermaid
sequenceDiagram
    participant U as 👤 발표자
    participant D as DemoScreen
    participant P as DemoAnalysisPipeline
    participant A as AASIST ONNX
    participant S as STT 전사
    participant V as ViewModel

    U->>D: 시나리오 선택 (1~6)
    D->>P: WAV 파일 분석 요청
    P->>A: 4초 세그먼트 추론
    A-->>P: fake score
    P->>S: 자막 전사 (사전 준비)
    S-->>P: 텍스트
    P-->>V: 결과 묶음
    V-->>D: 파형/자막/점수 동기 표시
    D-->>U: 재생과 동기화된 결과 노출
```

**검증된 점수 (현재 baseline):**

| 시나리오 | 평균 fake score | 최대 | 판정 |
|---|---:|---:|---|
| demo_01 일상 통화 | 0.240 | 0.645 | 🟢 SAFE |
| demo_02 TTS 일상 대화 | 0.994 | 1.000 | 🔴 DANGER |
| demo_03 실제 사람 피싱 | 0.017 | 0.036 | 🟢 SAFE (딥보이스 기준) |
| demo_04 TTS 피싱 | 0.985 | 1.000 | 🔴 DANGER |
| demo_05 은행 정상 안내 | 0.011 | 0.046 | 🟢 SAFE |
| demo_06 보험 사기 (사람) | 0.042 | 0.217 | 🟢 SAFE (딥보이스 기준) |

**한계 인식:**
- 시연 시 "녹음된 파일을 분석" → 현장감·임팩트 부족
- "실제 통화처럼 보이는" 시연이 필요하다는 판단

---

### 4.2 시연 2차 — 스피커폰 + 마이크 라이브 방식

**구성:** 다른 폰(또는 PC 스피커)에서 클로닝/TTS 음성을 재생하고, 시연 폰의 마이크로 들어와 on-device에서 실시간 분석.

```mermaid
flowchart LR
    subgraph SRC["발신측 시뮬레이션"]
        TTS[TTS / Voice Clone] --> SPK[🔊 스피커 재생]
    end
    SPK -.공기 전파.-> MIC
    subgraph PHONE["📱 시연 폰"]
        MIC[🎤 마이크 입력] --> FILT[음향 필터링<br/>대역 제한·왜곡·노이즈]
        FILT --> AAS[AASIST 추론]
        AAS --> RES[결과 표시]
    end

    style FILT fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    style RES fill:#fff
```

**문제 발견:**

```mermaid
graph LR
    A[원본 TTS 음원<br/>학습 분포 내] --> B[스피커 재생]
    B --> C[공기 + 마이크 통과<br/>acoustic filter]
    C --> D[학습 분포 외 신호<br/>OOD]
    D --> E[AASIST 탐지율 급락<br/>실측 ~20%]

    style A fill:#c8e6c9
    style D fill:#ffcdd2
    style E fill:#ef9a9a,stroke:#b71c1c
```

| 측정 대상 | 결과 |
|---|---|
| Python 직접 파일 분석 (TTS) | **100%** 탐지 |
| 모바일 캡처 fake 445개 파이썬 홀드아웃 | **56%** 탐지 |
| 실기기 시연 환경 (Xcover 5) | **~20%** 탐지 |

**시도한 보정:**
- iPhone 녹음 real 878개 + 모바일 캡처 fake 445개로 fine-tune (v4)
- → 한국어 real 오탐은 33% → 0%로 잡았으나 스피커→마이크 fake 탐지는 여전히 미흡
- → **acoustic augmentation 데이터가 훨씬 더 필요한 것으로 결론**

**조치:**
- AASIST 모델은 **v4 baseline으로 원복** (`a257e59 fix(demo): stabilize TrueVoice model baseline`)
- 시연 방식 자체를 재설계해야 한다는 결론

---

## 5. 현재 난관 — 의사결정이 필요한 지점

```mermaid
flowchart TB
    PROB["📌 우리가 가진 자산<br/>① 동작하는 AASIST v4 모델<br/>② STT + 키워드 탐지<br/>③ TrueVoice Guard 앱 UI<br/>④ Demo 파일 6종 검증된 점수"] --> OPT
    LIM["⚠️ 우리가 못 하는 것<br/>① 실제 통화 음성 라이브 캡처<br/>② 스피커→마이크 라이브 시연<br/>③ 샌드박스 승인 받기 전 양산"] --> OPT
    OPT{{"🎬 시연 어떻게 보여줄까?"}}
    OPT --> P1["1안<br/>피해 상황 연출 영상"]
    OPT --> P2["2안<br/>가상 통화 + 앱 동작 영상"]

    style PROB fill:#e8f5e9,stroke:#2e7d32
    style LIM fill:#ffebee,stroke:#c62828
    style OPT fill:#fff3e0,stroke:#e65100,stroke-width:3px
    style P1 fill:#e3f2fd,stroke:#1565c0
    style P2 fill:#f3e5f5,stroke:#6a1b9a
```

---

## 6. 시연 1안 vs 2안 비교

### 6.1 1안 — "현실의 피해 상황을 보여주는" 사회적 영상

> Clone voice 기반 보이스피싱 피해를 드라마처럼 연출 → 서비스 필요성 강조

```mermaid
flowchart LR
    S1[가족 사칭 voice clone<br/>제작] --> S2[피해자 통화 장면<br/>연기·재현]
    S2 --> S3[금전 송금·낭패<br/>상황 묘사]
    S3 --> S4["💡 만약 우리 앱이 있었다면…<br/>인터스티셜 카피"]
    S4 --> S5[앱 화면 짧게 노출<br/>5단계 위협 알림]

    style S3 fill:#ffcdd2
    style S4 fill:#fff8e1
    style S5 fill:#e3f2fd
```

| 항목 | 평가 |
|---|---|
| **장점** | 감정 임팩트 큼 / 비즈니스 문제(시장 사각지대) 직관적 / 기술 한계 우회 가능 |
| **단점** | 기술 PoC 면에서 약함 / "그래서 너희가 만든 게 뭔데?"에 답이 약함 / 영상 제작·연기 리소스 필요 |
| **신뢰성 리스크** | "탐지 가능했다"는 주장이 영상 안에서만 성립 — 심사위원이 검증하기 어려움 |
| **제작 난이도** | 중상 (스토리보드, 출연자, 편집) |

---

### 6.2 2안 — "가상 통화 + 앱 동작" 시연 영상

> 실제 fake voice는 우리가 만들고, 가상 수신 화면 + TrueVoice Guard 앱 화면을 합성해 "전화가 오면 실제로 이렇게 동작합니다" 시연

```mermaid
flowchart LR
    A[Voice clone 생성<br/>실제 사람 흡사] --> B[가상 수신 화면<br/>Mock incoming call]
    B --> C[TrueVoice Guard 앱<br/>실시간 분석 화면]
    C --> D[5단계 위협 레벨<br/>알림 + STT 자막 + 점수]
    D --> E[CRITICAL 알림<br/>발신자 차단 안내]

    style A fill:#fff8e1
    style B fill:#e3f2fd
    style C fill:#f3e5f5
    style D fill:#fff3e0
    style E fill:#ffcdd2
```

| 항목 | 평가 |
|---|---|
| **장점** | 우리 앱 동작이 직접 노출 / 기술 PoC 메시지 명확 / Demo 파일 분석 결과를 그대로 활용 가능 |
| **단점** | "실제 통화 라이브 아님" 명시 필요 / 영상 합성 연출이 어색하면 신뢰 떨어짐 |
| **신뢰성 리스크** | 라이브 캡처가 아님이 드러나면 PoC로서 약점 — 그러나 "샌드박스 승인 후 가능"이라는 가정 명시로 보완 가능 |
| **제작 난이도** | 중 (앱 화면 녹화 + voice clone + 화면 합성) |
