# Codex 위임 구현 가이드

3개 feature plan 을 Codex (codex-rescue agent) 가 순차 구현하도록 안내. 각 plan 별로 별도 codex 호출 → 구현 → 검증 → 다음 plan 진행.

## 권장 순서

```
1) feature/stt-display     ← 가장 영향 범위 작음
   ↓
2) feature/multi-alert     ← 새 알림 채널 추가 (HomeScreen 충돌 X)
   ↓
3) feature/brand-themes    ← HomeScreen / SettingsScreen / SettingsRepository 충돌 가능 — 마지막
   ↓
4) 데모 직전 → 모든 feature branch 를 master 또는 demo 브랜치로 머지
```

각 단계 사이에 **반드시 실기기 1회 sanity check** 후 다음으로.

## Codex 호출 prompt 템플릿 (각 plan 별)

### Plan 1 — STT Display

```
당신은 Android Kotlin developer 입니다. 다음 implementation plan 을 그대로 따라 구현해주세요.

Plan: docs/superpowers/plans/2026-04-25-feature-stt-display.md

작업 흐름:
1. plan 의 Phase 0 (branch + push) 실행
2. Phase 1 → 1.5 → 2 → 3 → 4 순서로 진행. 각 Task 의 Step 들을 정확히 따르되,
   - 코드 블록은 plan 에 적힌 그대로 사용 (수정 금지)
   - 각 Step 의 "기대" 결과 검증 (테스트 통과, 빌드 성공)
   - 각 Task 마지막의 git commit 명령 그대로 실행
3. Phase 3 (실기기 검증) 은 자동 실행 못 하므로 SKIP — 사용자가 직접 진행한다고 명시
4. Phase 4 (push) 실행

제약:
- Robolectric 의존성이 build.gradle 에 없으면 Phase 1.5 의 신규 테스트는 builtinKeywords() 만
  사용하는 단순 단위 테스트로 변경 후 진행 (plan 의 "알려진 한계" 참조)
- 기존 PhishingKeywordDetector 회귀 테스트가 깨지면 plan 의 Step 1.5.5 지침대로
  테스트 데이터 보강하여 통과시킴
- 어느 step 에서든 빌드 실패 / 테스트 실패가 나면 즉시 멈추고 원인 분석 후 보고

작업 디렉터리: /Users/hyohee/Documents/Claude_project/anti-deepvoice-guard
시작: git checkout android/real-device-verify
```

### Plan 2 — Multi-Alert

```
당신은 Android Kotlin developer 입니다. 다음 implementation plan 을 그대로 따라 구현해주세요.

Plan: docs/superpowers/plans/2026-04-25-feature-multi-alert.md

작업 흐름:
1. plan 의 Phase 0 (branch + push) 실행
2. Phase 1 → 2 → 3 → 4 → 5 → 6 순서로 진행. 각 Task 의 Step 들을 정확히 따르되,
   - 코드 블록은 plan 에 적힌 그대로 사용 (수정 금지)
   - 각 Step 의 "기대" 결과 검증
   - 각 Task 마지막의 git commit 명령 그대로 실행
3. Phase 5 (실기기 검증) 은 SKIP — 사용자가 직접 진행
4. Phase 6 (push) 실행

특별 주의:
- Phase 3 Task 4 의 NotificationChannel.enableVibration(false) 는 3개 채널 모두에 적용해야 함
- Phase 3 Task 5 의 NotificationHelper 생성자 시그니처 변경은 호출 측 (AudioCaptureService)
  까지 같이 수정 — Step 5.4 도 잊지 말 것
- AlertChannelDispatcher 의 Hilt @Inject 가 AudioCaptureService 에서만 쓰이는지 확인
  (다른 곳에서 의존하면 추가 wiring 필요)

작업 디렉터리: /Users/hyohee/Documents/Claude_project/anti-deepvoice-guard
시작: git checkout android/real-device-verify  (또는 feature/stt-display 머지 후)
```

### Plan 3 — Brand Themes

```
당신은 Android Kotlin developer 입니다. 다음 implementation plan 을 그대로 따라 구현해주세요.

Plan: docs/superpowers/plans/2026-04-25-feature-brand-themes.md

작업 흐름:
1. plan 의 Phase 0 (branch + push) 실행
2. Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 순서. 각 Task 의 Step 정확히 따름.
3. Phase 7 (실기기 검증) 은 SKIP — 사용자 직접 진행.
4. Phase 8 (push) 실행

특별 주의:
- 가상 브랜드명 (Nova 모바일, Lumi 모바일) 변경 금지 — 법적 안전 위해 정해진 이름
- 로고 vector drawable 의 SVG path 데이터는 plan 에 정확히 적힘 — 그대로 사용
- Theme.kt 전체 교체 (기존 dynamic color 코드 제거됨) — Phase 3 Step 3.1 참조
- MainActivity 는 Hilt 변환 X. Architecture 명시대로 `remember { SettingsRepository(context) }`
  fallback 패턴 그대로 사용. 이는 검증됨.
- SettingsScreen 의 다른 카드와 충돌 없는지 확인 (Plan 2 와 동시 구현 시 주의)

작업 디렉터리: /Users/hyohee/Documents/Claude_project/anti-deepvoice-guard
시작: git checkout android/real-device-verify  (또는 feature/multi-alert 머지 후)
```

## 각 plan 완료 후 사용자가 할 일

### Plan 1 완료 후
1. Android Studio ▶ Run
2. STT 권한 ON 확인
3. START → "검찰청에서 계좌이체 부탁드립니다" 말하기
4. 화면에 "검찰청", "계좌이체" 또는 "계좌 이체" 가 빨간 굵은 글씨로 강조되는지 확인
5. STOP → 카드 사라지는지 확인
6. 통과하면 다음 plan 진행 / 실패하면 codex 에 로그 + 화면 캡처 전달

### Plan 2 완료 후
1. ▶ Run, Settings 에서 Haptic ON / TTS ON 확인
2. 스피커로 fake 재생 (`afplay test-samples/fake/TTS1.wav`)
3. 폰 진동 + 한국어 TTS 알림 + 푸시 알림 모두 발생 확인
4. Haptic OFF 후 재시도 → 진동 없는지 (NotificationChannel 진동 OFF 가 효과가 있는지)
5. TTS OFF 후 재시도 → 음성 없는지

### Plan 3 완료 후
1. ▶ Run
2. HomeScreen 상단에 기본 브랜드 헤더 (방패+Anti-DeepVoice Guard)
3. Settings → 브랜드 → "Nova 모바일" 선택 → 보라색 별 + 보라 톤
4. "Lumi 모바일" → 주황 원 + 주황 톤
5. 앱 재시작 후 마지막 선택 유지

## 트러블슈팅

| 증상 | 원인 가능성 | 해결 |
|---|---|---|
| 빌드 실패 — unresolved reference | import 누락 | plan 의 import 블록 다시 확인 |
| Robolectric 테스트 실패 | dependency 미적용 | builtinKeywords() 만 쓰는 형태로 다운그레이드 |
| Plan 2 — TTS 첫 알림 누락 | pendingMessage flush 미동작 | `onInit` 콜백에서 `pendingMessage.getAndSet(null)` 호출 확인 |
| Plan 3 — 브랜드 변경 후 색상 안 바뀜 | recomposition 안 일어남 | `remember(settings.carrierBrand) { ... }` key 명시 확인 |
| HomeScreen.kt 에서 `transcription` 변수 없음 | line 변경됨 | `grep -n "transcription"` 으로 새 위치 찾아 통합 |

## 머지 전 체크리스트

각 feature branch 의 PR 또는 직접 머지 전:

- [ ] `./gradlew :app:testDebugUnitTest` 통과
- [ ] `./gradlew :app:assembleDebug` 통과
- [ ] 실기기에서 해당 feature 의 핵심 시나리오 동작 확인
- [ ] 다른 feature branch 와 충돌 가능 파일 (SettingsRepository, SettingsScreen, HomeScreen) 변경 부분 확인 — 머지 시 conflict 예상이면 미리 정렬
- [ ] DIAG 로그 (`Log.d` with throttling) 그대로 둠 — 데모 직전 정리는 별도 task
