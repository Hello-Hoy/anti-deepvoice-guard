# 실기기 라이브 마이크 테스트 가이드

USB 연결된 Android 폰에서 라이브 마이크로 파이프라인을 검증합니다.
스피커폰 모드로 피싱 대본을 재생 → 실시간 탐지 + 알림 시연.

## 왜 실기기가 필요한가?

**Apple Silicon Mac** (M1/M2/M3/M4) + **Android Studio 에뮬레이터** 조합에서는
호스트 마이크 → 게스트 AudioRecord 라우팅이 **불안정**합니다.
- 토글이 ON으로 보여도 AudioRecord가 실제 샘플을 받지 못함
- VAD probability 0 유지, 파이프라인 동작 안 함

실기기는 **물리 마이크 → AudioRecord**가 직접 동작하므로 문제 없음.

## 사전 준비

### 1. Android 폰
- **Android 8.0(Oreo) 이상**
- 가급적 Pixel, Samsung, LG 등 **GMS(Google Mobile Services) 탑재 기기**
  - SpeechRecognizer 오프라인 한국어 모델이 설치되어 있어야 STT 동작
  - 대부분의 폰은 기본 설치되어 있음
- 배터리 30% 이상 (빌드/설치 시간 고려)

### 2. 개발자 모드 + USB 디버깅 ON

폰에서:
1. **설정 → 휴대전화 정보 → 빌드 번호**를 7번 연속 탭
2. "개발자가 되었습니다" 메시지
3. 뒤로 가서 **설정 → 시스템 → 개발자 옵션**
4. **USB 디버깅** 토글 ON

### 3. USB 케이블 연결

Mac ↔ Android 폰 USB-C 케이블 연결 후 폰에 **"USB 디버깅 허용"** 다이얼로그 뜨면 **허용**.

확인:
```bash
adb devices
# → List of devices attached
# → XXXXXXXX	device    (← 정상)
```

## 앱 빌드 + 설치

### 1. Android Studio

1. `anti-deepvoice-guard/android-app/` 프로젝트 열기
2. 상단 디바이스 선택 드롭다운에서 **실기기 선택** (에뮬레이터 X)
3. **▶ Run 'app'** 클릭
4. 첫 실행 시 5~10분 소요

### 2. 또는 명령행

```bash
cd android-app
./gradlew installDebug
adb shell am start -n com.deepvoiceguard.app/.MainActivity
```

## 권한 설정

앱 최초 실행 시 권한 다이얼로그 수락:

| 권한 | 용도 | 필수 여부 |
|------|------|---------|
| 마이크 (RECORD_AUDIO) | AudioRecord 음성 캡처 | **필수** |
| 알림 (POST_NOTIFICATIONS, Android 13+) | 위협 탐지 알림 | **필수** |
| 전화 상태 (READY_PHONE_STATE) | 통화 수신 자동 시작 | 선택 |

STT 사용 시 추가:
- 앱 내 Settings → "STT 사용 동의" 다이얼로그 → **동의**
  - 주의: 음성이 Google 서버로 전송될 수 있음

## 라이브 시연 방법

### 방법 A: 다른 폰에서 스피커폰 재생

**준비물**: 피싱 대본 오디오가 담긴 다른 휴대폰 or 컴퓨터 스피커.

1. **주 폰**(앱 설치): 앱 실행 → **Home 탭 → START**
2. **Monitoring Active** 상태 확인 + 상단 알림바에 "Anti-DeepVoice Guard — Monitoring..." 표시
3. **재생 폰**: `test-samples/demo/` 폴더의 Demo4 오디오(TTS 피싱) 또는 유사한 피싱 스크립트 재생
4. 두 폰을 **30cm 이내 거리**에 두고 재생 폰 **스피커폰 모드**
5. 주 폰의 화면에서 다음 확인:
   - **Voice Activity bar** 실시간 움직임
   - 수 초 후 **CRITICAL 알림** + 진동
   - Home 탭 **Threat Status**: CRITICAL, 딥보이스 ~90%+, 피싱 100%

### 방법 B: 실제 통화 + 스피커폰

1. 주 폰에서 앱 **START**
2. 다른 폰에서 주 폰으로 **전화 걸기**
3. 주 폰 전화 수신 → **스피커폰** 켜기
4. 다른 폰에서 피싱 대본을 읽거나 재생
5. 주 폰 앱에서 실시간 탐지 확인

이 시나리오가 경진대회 시연의 **"핵심 스토리"**입니다:
> "알뜰폰 사용자가 보이스피싱 전화를 받아 스피커폰으로 받으면,
> 앱이 통화 음성을 분석해 AI 합성 + 피싱 키워드를 실시간 탐지하여 경고합니다."

## 체크리스트

시연 전:
- [ ] 주 폰 배터리 50% 이상 + 스피커 음량 정상
- [ ] 재생 폰 피싱 오디오 파일 준비 (길이 10~30초 권장)
- [ ] 조용한 환경 (배경 소음 적은 방)
- [ ] 주 폰 앱 **Monitoring Stopped** 상태로 시작 (클린 스타트)
- [ ] 알림 수신 소리/진동 테스트 (기타 앱 방해금지 OFF)

시연 도중:
- [ ] START 후 Voice Activity bar 움직임 확인
- [ ] 피싱 오디오 재생 중 ~2초 내 반응 (VAD speechEnd 후 AASIST 추론)
- [ ] CRITICAL 알림 상단 배너로 표시
- [ ] 알림 탭 시 앱으로 이동

시연 후:
- [ ] STOP 버튼으로 모니터링 종료
- [ ] History 탭 → 방금 탐지한 이벤트 기록 확인

## 성능 기대치

| 측정 | 값 |
|------|-----|
| 첫 반응 지연 | ~2.5초 (VAD 256ms + AASIST 100ms + 알림 UI) |
| 배터리 소모 | ~5~10% /시간 (연속 모니터링 중) |
| CPU 사용률 | ~15% (추론 순간), ~3% (idle VAD) |
| 메모리 | ~80MB |

## 문제 해결

### "Voice Activity bar가 전혀 안 움직임"

- 폰 마이크 차단 여부 확인 (Privacy 모드 등)
- Settings 앱에서 **Anti-DeepVoice Guard → 권한 → 마이크 허용**
- 실행 중인 다른 녹음/통화 앱 종료 (마이크 독점 가능)

### "STT가 동작 안 함"

- 폰 Settings → Google → Voice → Offline speech recognition → **Korean (한국어)** 다운로드
- 앱 내 Settings → STT 토글 + 동의
- Google Play가 없는 커스텀 ROM은 SpeechRecognizer 미지원

### "Monitoring Active인데 알림이 안 옴"

- 앱 알림 권한 (Android 13+): 설정 → 알림 → Anti-DeepVoice Guard → 허용
- 방해금지 모드 OFF
- 30초 쿨다운: 같은 위협 레벨은 30초 이내 중복 알림 안 뜸 (의도적)

### "실기기는 인식됐는데 Run이 안 됨"

```bash
adb devices
# unauthorized 표시 → 폰에서 USB 디버깅 팝업 "허용" 재확인
```

### "빌드는 되는데 설치 단계에서 실패"

- 폰 저장공간 확인 (최소 500MB 필요)
- 이전 APK 제거: `adb uninstall com.deepvoiceguard.app`
- 서명 불일치 시: Android Studio → Build → Clean Project

## 경진대회 리허설 시나리오

**총 소요시간: 3~4분**

```
[0:00] 앱 소개 (10초)
  "Anti-DeepVoice Guard는 통화 중 상대방의 음성을 실시간 분석해
   AI 딥보이스와 보이스피싱 키워드를 동시에 감지합니다."

[0:10] Demo 탭 시연 (1분 30초)
  - #1 SAFE → 일반 통화는 조용히 통과
  - #4 CRITICAL → 가장 위험한 시나리오 집중
  - #3 WARNING → 실제 사람이 피싱 스크립트 읽을 때 (피싱만)
  - #5 SAFE → 은행 정상 안내 오탐 방지

[1:40] 실기기 전환 + 라이브 시연 (1분 30초)
  - 주 폰 START → Monitoring Active
  - 재생 폰 스피커폰으로 피싱 대본 재생
  - 주 폰 CRITICAL 알림 + 진동 확인
  - History 탭으로 증거 확인

[3:10] 설계/비즈니스 설명 (50초)
  - 알뜰폰 SDK 라이선스 모델
  - 규제 샌드박스 로드맵
  - 2026 통신사 확산 목표
```

## 라이브 시연 백업 플랜

**라이브 실패 시나리오** (실기기 문제, Wi-Fi 끊김 등):
1. Demo 탭 시연으로 **원활히 전환** (미리 Demo 탭으로 이동해둘 것)
2. "실기기 환경 이슈로 Demo 모드 시연합니다"라고 **자연스럽게 설명**
3. Demo 모드도 동일한 파이프라인 (VAD 제외) 통과하므로 **기능 완전성은 동일**

Demo는 파일 기반이라 **100% 안정적**입니다. 경진대회에서는 Demo 시연이 primary, 실기기가 secondary로 가는 게 안전.
