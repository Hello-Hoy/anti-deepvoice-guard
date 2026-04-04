# Anti-DeepVoice Guard — Issue Summary

> 2026-04-03 ~ 04-04 세션에서 논의된 기술 검토, 구현, 비즈니스 분석 종합 정리

---

## 1. 구현 완료 사항

### 1-1. 모델 v5.1 (현재 배포 중)
- **val_acc: 92.95%** (v3 85.8% → v4 90.5% → v5.1 92.95%)
- 테스트 정확도: 서버 46/47 (97.9%), Android 23/24 (95.8%)
- 학습 데이터: real 9,176 + fake 9,684 = 18,860개
- 개선 이력:
  - v3: 기본 multi-TTS + clone 학습
  - v4: edge_ko_hyunsu/injoon 추가 데이터 500개, 전화품질 augmentation
  - v5: InJoon 전용 500개 추가, 볼륨 정규화, TTS2 오탐 해결
  - v5.1: 코드 리뷰 반영 (FFT rolloff 수정, clamp 추가, gain cap, MPS contiguous)

### 1-2. Android 앱 통화 자동 감지
- PhoneStateReceiver: 전화 상태 변화 감지 (IDLE/RINGING/OFFHOOK)
- BootReceiver: 부팅 시 자동 서비스 시작
- CallSession: 통화 세션 추적
- AudioSource: 통화 중 VOICE_COMMUNICATION, 일반 MIC
- Android 14+ FGS 백그라운드 제한 대응
- 코드 리뷰 11개 이슈 수정 (Claude + Codex 교차 검증)

### 1-3. 서버 버그 수정
- m4a/aac 파일 처리: soundfile → librosa 폴백 추가

### 1-4. 프로젝트 인프라
- CLAUDE.md, AGENTS.md: 에이전트 팀 + Codex 교차 검증 프로세스 정의
- Android instrumented test (OnDeviceDetectionTest)
- Build & Test Guide PDF (한국어, 8페이지)
- GitHub push 완료: github.com/Hello-Hoy/anti-deepvoice-guard

---

## 2. 남은 기술 이슈

### 2-1. edge_ko_injoon 오탐 (fake=41.4%)
- InJoonNeural TTS가 매우 자연스러워 모델이 탐지하지 못함
- 1,000개+ 추가 데이터 or RTX 4080 unfreeze 학습으로 개선 가능
- 현재 수준에서도 CAUTION 경고는 가능 (앱 임계값 조정)

### 2-2. 스피커폰 모드 제약
- **일반 앱(우리 앱)은 수화기 모드에서 상대방 음성 캡처 불가**
- Android OS가 CAPTURE_AUDIO_OUTPUT 권한을 시스템 앱에만 부여
- 스피커폰 모드에서만 마이크가 양쪽 음성을 수집

### 2-3. 코드 리뷰 미반영 사항 (향후 리팩토링)
- WeightedSampler + class_weight 이중 보정 → 하나만 사용
- test-samples가 학습에 포함될 가능성 → 디렉토리 분리 필요
- augmentation이 padding 이후 적용 → 순서 변경 필요
- DataStore/SharedPreferences 이중 쓰기 → 단일화

---

## 3. 비즈니스 분석: 실시간 딥보이스 탐지 서비스

### 3-1. 한국 시장 현황

| 서비스 | 제공자 | 딥보이스 탐지 | 기술 방식 |
|--------|--------|:---:|----------|
| 익시오(ixi-O) | LG유플러스 | O (안티딥보이스) | 온디바이스 AI, 삼성 전화앱 연동 |
| 에이닷 전화 | SKT | X (대화 패턴만) | 온디바이스 AI |
| 후후 | KT | O | 온디바이스 AI |
| 삼성 전화앱 | 삼성전자 | X (대화 패턴만) | One UI 8 기본 내장 |
| KB리브모바일 | KB금융+LGU+ | O (익시오 제공) | LGU+ 익시오 번들 |

### 3-2. 핵심 발견: 통신사들도 네트워크가 아닌 On-Device AI 사용

- 통신 3사 모두 개인정보 보호를 위해 기기 내에서 분석
- **삼성 전화앱과 시스템 레벨 연동**으로 통화 오디오 접근
- 정부(과기정통부) + 삼성 + 통신사 3자 협력 체계

### 3-3. MVNO(알뜰폰) 협업 검토 결과

**결론: MVNO 앱에 SDK를 넣어도 수화기 모드에서 상대방 음성 캡처 불가**

| 검토 항목 | 결과 |
|----------|------|
| MVNO가 네트워크 레벨에서 음성 접근 | **불가** — 전원 단순 재판매(Light MVNO), 자체 IMS 없음 |
| MVNO 앱에 SDK로 음성 캡처 | **불가** — 일반 앱이므로 CAPTURE_AUDIO_OUTPUT 권한 없음 |
| MVNO 전용 단말기에 시스템 앱 탑재 | 이론적 가능, 현실적 불가 (수억 원, MVNO 동기 없음) |
| KB리브모바일 사례 | MVNO 자체 기술 아님 → LGU+ 익시오를 가져다 번들 제공한 것 |

### 3-4. Android 통화 오디오 접근 — 모든 경로 정리

| 경로 | 수화기 모드에서 가능? | 현실성 |
|------|:---:|------|
| 시스템 앱 (삼성/통신사) | ✅ | 삼성+정부 협력 필요 |
| Samsung Knox SDK | ❌ | API가 Android 11에서 폐기됨 |
| Android Enterprise / DPC | ❌ | 해당 API 없음 |
| Accessibility Service | ❌ | 오디오 접근 불가 (UI만) |
| CallScreeningService | ❌ | 번호만, 오디오 없음 |
| MediaProjection | ❌ | 통화 오디오 제외 |
| Google API | ❌ | 서드파티용 API 없음 |
| 스피커폰 모드 | ✅ | **현재 MVP 방식** |
| 루팅 기기 | ✅ | 일반 사용자 불가 |
| iOS | ❌ | 어떤 앱도 불가 |

### 3-5. 현실적 비즈니스 전략 (우선순위)

**1순위: 삼성+정부 협력 체계 진입**
- 과기정통부 "AI 보이스피싱 대응 R&D" 참여
- ICT 규제 샌드박스 실증특례 신청 (LGU+ 익시오 선례)
- 삼성 One UI 보이스피싱 파이프라인에 딥보이스 엔진 통합 제안

**2순위: 통신사에 딥보이스 탐지 SDK 공급**
- SKT 에이닷은 아직 딥보이스(합성음) 탐지 미보유 → 엔진 공급 기회
- 삼성 기본 전화앱도 대화 패턴만 분석, 합성음 탐지 미보유

**3순위: 금융권 B2B**
- 은행/보험사 콜센터에 API 공급
- 정부 ASAP 플랫폼 (금융권 130개사) 생태계 편입
- 서버 환경이므로 Android 제약 없음

**4순위: 현재 MVP 유지 (스피커폰 모드)**
- 데모/시연/PoC 용도
- MVNO 앱에 SDK 공급 시 스피커폰 모드로 제공

---

## 4. 경쟁 환경

### 직접 경쟁: LGU+ 익시오 안티딥보이스
| | 우리 (Anti-DeepVoice Guard) | LGU+ 익시오 |
|--|--------------------------|-------------|
| 딥보이스 탐지 정확도 | 93% (v5.1) | 98% |
| 모델 크기 | 1.8MB (ONNX) | 비공개 |
| 학습 데이터 | ~19K 자체 제작 | 3,000시간/200만건 + 국과수 |
| 분석 위치 | On-Device | On-Device |
| 앱 권한 | 일반 앱 | 삼성 시스템 레벨 연동 |
| 부가 기능 | 없음 (순수 탐지) | STT+NLP+성문매칭+ATM차단 |

### 차별화 포인트
- 경량 모델 (644KB AASIST-L) — 초경량 엣지 디바이스 구동
- 오픈소스 기반 — 커스터마이징 가능, 통신사 종속 X
- 범용 오디오 분석 — 통화뿐 아니라 녹음 파일, 스트리밍 등 적용 가능
- 다국어 확장 가능 — 한국어 외 시장 진출

---

## 5. 기술 스택 요약

```
프로젝트 구조:
├── model-training/     # AASIST 모델 학습, ONNX 변환
│   ├── AASIST (PyTorch) → ONNX 변환
│   ├── 학습: MPS (M4 Pro) / CUDA (RTX 4080)
│   └── Augmentation: gain, noise, phone quality, volume norm
├── model-server/       # FastAPI 추론 서버
│   └── librosa 오디오 로딩 (m4a/mp3/wav 지원)
├── android-app/        # Kotlin + Jetpack Compose
│   ├── ONNX Runtime (On-Device 추론, ~250ms)
│   ├── Silero VAD (음성 구간 검출)
│   ├── Foreground Service (마이크 캡처)
│   ├── PhoneStateReceiver (통화 자동 감지)
│   ├── Room DB + AES-256-GCM 암호화 저장
│   └── Hilt DI, Material 3, Navigation Compose
└── test-samples/       # real/fake 테스트 오디오
```

---

## 6. 커밋 히스토리 (이번 세션)

```
2493dc0 docs: add Build & Test Guide PDF
15e0d4c fix: add missing ONNX models (silero_vad, aasist_l)
c535521 chore: update v5.1 model weights, gitignore
e24e516 fix: .contiguous() after irfft for MPS
9525fd9 feat: model v5 — fix TTS2 false negative, improve augmentation
a2cc9b5 fix: address remaining code review issues
fb5e3ae feat: add phone call auto-detection, retrain model v4
28cd1c6 fix: server m4a audio handling, add project guidelines
```

---

*이 문서는 2026-04-04 세션 기준으로 작성되었습니다.*
