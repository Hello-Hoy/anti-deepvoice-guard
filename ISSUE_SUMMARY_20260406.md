# Anti-DeepVoice Guard — Issue Summary (2026-04-06)

> 2026-04-03 ~ 04-06 세션에서 논의된 기술 검토, 구현, 비즈니스 분석 종합 정리
> 이전 버전: ISSUE_SUMMARY_20260404.md

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

### 3-3. KB리브모바일 사례 분석

KB리브모바일이 "통화 중 보이스피싱 감지"를 제공하지만, **MVNO 자체 기술이 아님**:
- ATM 출금 차단: KB리브모바일 자체 (음성 분석 아님, 통화 상태만 확인)
- AI 통화 분석 + 안티딥보이스: **LGU+ 익시오를 가져다 번들 제공**한 것
- 익시오가 수화기 모드에서 동작하는 이유: **삼성 전화앱과 시스템 레벨 연동** (삼성+정부 협력)

### 3-4. MVNO(알뜰폰) 협업 검토 결과

**결론: MVNO 앱에 SDK를 넣어도 수화기 모드에서 상대방 음성 캡처 불가**

| 검토 항목 | 결과 |
|----------|------|
| MVNO가 네트워크 레벨에서 음성 접근 | **불가** — 전원 단순 재판매(Light MVNO), 자체 IMS 없음 |
| MVNO 앱에 SDK로 음성 캡처 | **불가** — 일반 앱이므로 CAPTURE_AUDIO_OUTPUT 권한 없음 |
| MVNO 전용 단말기에 시스템 앱 탑재 | 이론적 가능, 현실적 불가 (수억 원, MVNO 동기 없음) |
| KB리브모바일 사례 | MVNO 자체 기술 아님 → LGU+ 익시오를 가져다 번들 제공한 것 |

---

## 4. MVNO 실시간 통화 음성 분석 불가능 — 법적/기술적 근거

### 4-1. 법적 근거

#### 헌법 제18조
> "모든 국민은 통신의 비밀을 침해받지 아니한다."

#### 통신비밀보호법

| 조항 | 내용 | 적용 |
|------|------|------|
| **제2조 제7호** | "감청이란 전기통신에 대해 당사자의 동의 없이 **전자장치·기계장치 등을 사용**하여 통신의 음향을 **청취하여 그 내용을 지득**하는 것" | AI 통화 음성 분석 = 전자장치로 음향을 청취하여 내용 지득 = **감청 해당** |
| **제3조 제1항** | "**누구든지** 이 법에 의하지 아니하고는 전기통신의 감청을 하지 못한다" | 예외: 환부우편물, 수출입우편물, 구속/복역 통신, 파산선고자, 혼신제거만. **상업 서비스 불포함** |
| **제3조 제2항** | "통신제한조치는 **범죄수사 또는 국가안전보장** 목적의 보충적 수단" | 상업적 목적 원천 불가 |
| **제16조 제1항** | 제3조 위반 벌칙 | **1년~10년 징역 + 5년 이하 자격정지** |

#### 전기통신사업법

| 조항 | 내용 | 적용 |
|------|------|------|
| **제2조** | 기간통신사업자: "자체 통신설비로 역무 제공" / 별정통신사업자: "기간통신사업자 설비 이용" | MVNO = 별정통신사업자 → 자체 설비 없음 |
| **제83조 제1항** | "**누구든지** 전기통신사업자가 취급 중에 있는 통신의 비밀을 침해하거나 누설하여서는 아니 된다" | MVNO 자신도 자사 가입자 통화 비밀 침해 불가 |

#### 개인정보보호법

| 조항 | 내용 | 적용 |
|------|------|------|
| **제23조** | 민감정보(생체인식정보) 처리 제한 | 음성 = "행동적 특징" → 생체인식정보 → 별도 동의 필요 |
| **시행령 제18조** | "특정 개인을 알아볼 목적으로 기술적 수단을 통해 생성한 정보" | AI 음성 분석 = 기술적 수단 → 민감정보 처리 |
| **제15조** | 개인정보 수집 시 정보주체 동의 필요 | **통화 상대방은 MVNO 가입자 아님 → 동의 획득 구조적 불가** |
| 위반 시 | | **5년 이하 징역 또는 5천만원 이하 벌금** |

### 4-2. 기술적 근거

#### Android OS 권한 체계

**CAPTURE_AUDIO_OUTPUT 권한:**
- Protection Level: `signature|privileged`
- Android SDK 공식: **"Not for use by third-party applications."**
- 획득 조건: (1) OS와 동일 인증서 서명(OEM만) 또는 (2) `/system/priv-app/` + `privapp-permissions.xml` 등록
- **일반 앱은 어떤 계약을 하든 획득 불가**

**AudioSource 접근:**
- `VOICE_CALL`, `VOICE_UPLINK`, `VOICE_DOWNLINK` → `CAPTURE_AUDIO_OUTPUT` 필수
- Android 12에서 서드파티 완전 차단

**Google Play 정책 (2022.05~):**
- Accessibility Service 통화 녹음 완전 금지
- "The Accessibility API is not designed and cannot be requested for remote call audio recording."

#### MVNO 망 구조

```
음성 통화 경로:
발신자 → 기지국 → MNO 코어 네트워크(IMS/VoLTE) → 기지국 → 수신자
                        ↑
                   MNO만 접근 가능, MVNO 접근 불가

MVNO 접근 가능: 과금 정보, 가입회선 관리, 통화 상태(통화 중 여부)
MVNO 접근 불가: 음성 원본(PCM), RTP 스트림, VoLTE 미디어 플레인
```

- 한국 MVNO 100% = 단순 재판매(Light MVNO)
- 풀 MVNO = 국내 0곳
- 도매제공 계약에 음성 원본 데이터 미포함

### 4-3. LGU+ 익시오가 합법적으로 가능한 이유 (비교)

| 항목 | LGU+ 익시오 (가능) | MVNO 앱 (불가능) |
|------|-----------------|---------------|
| 법적 지위 | **기간통신사업자** | 별정통신사업자 |
| 자체 통신설비 | IMS, VoLTE 코어 보유 | 없음 (MNO 임차) |
| 앱 권한 | 삼성 **프리인스톨** → `/system/priv-app/` | Play Store 일반 앱 |
| 삼성 협력 | 갤럭시 S25 사전 탑재, S21까지 확대 | OEM 프리인스톨 불가 |
| 규제 특례 | ICT 규제 샌드박스 승인 (2025.09) | 기술적 전제조건 미충족 |
| 음성 분석 | **온디바이스** (서버 전송 없음) → 감청 아님 | 오디오 접근 자체 불가 |
| 상대방 동의 | 온디바이스 처리로 우회 | 해결 방법 없음 |

### 4-4. 모든 경로 총정리

| 경로 | 수화기 모드 상대방 음성 | 현실성 |
|------|:---:|------|
| 삼성 프리인스톨 (익시오 방식) | ✅ | 기간통신사업자만 가능 |
| 통신사 앱에 SDK 공급 | ✅ | 통신사가 시스템 권한 보유 |
| 정부 R&D / 규제 샌드박스 | ✅ | 6개월~1년, 선례 있음 |
| MVNO 앱에 SDK | ❌ | **Android OS가 차단 + 법적 근거 없음** |
| Samsung Knox SDK | ❌ | API가 Android 11에서 폐기 |
| Accessibility Service | ❌ | 오디오 접근 불가 + Play Store 금지 |
| CallScreeningService | ❌ | 번호만, 오디오 없음 |
| MediaProjection | ❌ | 통화 오디오 제외 |
| 스피커폰 모드 | ✅ | **현재 MVP 방식** |
| 루팅 기기 | ✅ | 일반 사용자 불가 |
| iOS | ❌ | 어떤 앱도 불가 |

---

## 5. 현실적 비즈니스 전략 (우선순위)

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

---

## 6. 경쟁 환경

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

## 7. 법적/기술적 근거 출처

### 법령
- 대한민국헌법 제18조
- 통신비밀보호법 제2조 제7호, 제3조, 제16조
- 전기통신사업법 제2조, 제83조
- 개인정보보호법 제15조, 제23조 / 시행령 제18조
- 정보통신 진흥 및 융합 활성화 등에 관한 특별법 제38조의2 (ICT 규제 샌드박스)

### 기술 문서
- Android SDK: android.permission.CAPTURE_AUDIO_OUTPUT — "Not for use by third-party applications"
- Android AOSP: Privileged Permission Allowlist (source.android.com/docs/core/permissions/perms-allowlist)
- Google Play 정책: Accessibility API 통화 녹음 금지 (2022.05~)

### 보도자료/뉴스
- LG유플러스 익시오 안티딥보이스 세계 최초 상용화 (lg.co.kr/media/release/29093)
- LGU+ ICT 규제 샌드박스 실증특례 승인 (2025.09)
- KB리브모바일 알뜰폰 최초 AI 통화 에이전트 도입 (2025.12)
- KB국민은행+LGU+ MWC26 AI 보이스피싱 실시간 대응 공개
- 과기정통부 AI 보이스피싱 대응 R&D 민관 협의체

---

## 8. 커밋 히스토리

```
5fd1ba1 docs: add issue summary — technical review and business analysis
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

*이 문서는 2026-04-06 세션 기준으로 작성되었습니다.*
