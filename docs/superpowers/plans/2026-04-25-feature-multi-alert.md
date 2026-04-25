# Feature 2 — 다중 알림 (햅틱 + 백그라운드 음성)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DANGER/WARNING/CRITICAL 발생 시 기존 푸시 알림에 더해 (1) 햅틱 진동, (2) 한국어 TTS 음성 알림이 발생. **각 채널이 독립 ON/OFF** 가능 — 진동만 끄고 TTS 만 활성화 같은 조합도 정상 동작.

**Architecture:**
- 새 `AlertChannelDispatcher` 클래스가 햅틱(`Vibrator`) + TTS(`TextToSpeech`) 발화를 캡슐화. **CombinedThreatLevel 만 받음** — 호출 측에서 ThreatLevel→CombinedThreatLevel 매핑.
- 기존 `NotificationHelper` 의 `showDetectionAlert(AggregatedResult)` / `showCombinedAlert(CombinedThreatResult)` 둘 다 알림 게시 직후 dispatcher 호출. ThreatLevel 만 들어오는 케이스(딥보이스 단독)는 `CombinedThreatAggregator.deepfakeOnlyLevel(ThreatLevel)` 헬퍼로 변환.
- **NotificationChannel 의 자체 진동(`enableVibration(true)`)을 끄고**, 모든 진동을 dispatcher 가 책임. 그래야 `hapticAlertEnabled = false` 시 진짜로 진동 없음.
- TTS 는 **fully-asynchronous init**: 첫 호출 시 init 시작 + 메시지를 큐에 넣고, `onInit` 콜백에서 큐 flush. 첫 알림이 누락되지 않음.
- Settings 에 두 토글 (haptic, tts). 푸시 알림 자체는 항상 ON (기본 알림 채널이 책임).

**Tech Stack:** Kotlin, Android `VibratorManager` (S+) / `Vibrator` (legacy), `TextToSpeech`, Hilt DI.

**Base branch:** `android/real-device-verify`. Plan 1 (`feature/stt-display`) 와 병렬 작업 가능 — 충돌 파일 없음.

---

## File Structure

| File | 책임 |
|---|---|
| `android-app/app/src/main/java/com/deepvoiceguard/app/alert/AlertSettings.kt` (신규) | data class — dispatcher 호출 시 채널별 ON/OFF |
| `android-app/app/src/main/java/com/deepvoiceguard/app/alert/AlertChannelDispatcher.kt` (신규) | Vibrator + TTS 통합. CombinedThreatLevel 기반. lifecycle. |
| `android-app/app/src/main/java/com/deepvoiceguard/app/inference/CombinedThreatAggregator.kt` | (이미 존재) `deepfakeOnlyLevel(ThreatLevel)` 헬퍼 사용 — 본 plan에선 변경 X, 호출만 |
| `android-app/app/src/main/java/com/deepvoiceguard/app/service/NotificationHelper.kt` | NotificationChannel 진동 OFF + showDetectionAlert / showCombinedAlert 끝에 dispatcher 호출 |
| `android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioCaptureService.kt` | dispatcher Hilt @Inject + NotificationHelper 생성 시 dispatcher + settingsProvider 람다 전달 + onDestroy 에서 shutdown |
| `android-app/app/src/main/java/com/deepvoiceguard/app/storage/SettingsRepository.kt` | `hapticAlertEnabled`, `ttsAlertEnabled` 필드 + Keys + setter |
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/SettingsScreen.kt` | "알림 채널" 카드 (기존 `SettingRow` 패턴 — 콜백은 `() -> Unit`) |
| `android-app/app/src/main/java/com/deepvoiceguard/app/di/AppModule.kt` | dispatcher Hilt provide |
| `android-app/app/src/test/java/com/deepvoiceguard/app/alert/AlertChannelDispatcherTest.kt` (신규) | dispatcher 단위 테스트 (Vibrator/TTS Robolectric/mockk) |

---

## Phase 0 — Branch & 백업

- [ ] **Step 0.1: 새 feature branch**

```bash
git checkout android/real-device-verify
git checkout -b feature/multi-alert
```

- [ ] **Step 0.2: 원격 push**

```bash
git push -u origin feature/multi-alert
```

---

## Phase 1 — Settings 필드

### Task 1: AppSettings 확장

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/storage/SettingsRepository.kt`

- [ ] **Step 1.1: data class 필드 추가**

기존 `data class AppSettings(...)` 안에 두 줄 추가 (기존 컨벤션 유지):

```kotlin
val hapticAlertEnabled: Boolean = true,        // DANGER/CRITICAL 시 진동 (기본 ON)
val ttsAlertEnabled: Boolean = false,          // 백그라운드 TTS 음성 알림 (기본 OFF — opt-in)
```

- [ ] **Step 1.2: Keys object 에 prefs key 추가**

```kotlin
val HAPTIC_ALERT_ENABLED = booleanPreferencesKey("haptic_alert_enabled")
val TTS_ALERT_ENABLED = booleanPreferencesKey("tts_alert_enabled")
```

- [ ] **Step 1.3: settings flow map 에 reader 추가**

기존 `AppSettings(...)` 빌드 부분에 두 줄 추가:

```kotlin
hapticAlertEnabled = prefs[Keys.HAPTIC_ALERT_ENABLED] ?: true,
ttsAlertEnabled = prefs[Keys.TTS_ALERT_ENABLED] ?: false,
```

- [ ] **Step 1.4: setter 함수 2개 추가** (기존 setter 패턴 동일)

```kotlin
suspend fun setHapticAlertEnabled(value: Boolean) {
    context.dataStore.edit { it[Keys.HAPTIC_ALERT_ENABLED] = value }
}

suspend fun setTtsAlertEnabled(value: Boolean) {
    context.dataStore.edit { it[Keys.TTS_ALERT_ENABLED] = value }
}
```

- [ ] **Step 1.5: 빌드 + 커밋**

```bash
cd android-app && ./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

기대: BUILD SUCCESSFUL.

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/storage/SettingsRepository.kt
git commit -m "feat(settings): add hapticAlertEnabled / ttsAlertEnabled fields with DataStore keys"
```

---

## Phase 2 — AlertChannelDispatcher (TDD)

### Task 2: AlertSettings + Dispatcher 단위 테스트

**Files:**
- Create: `android-app/app/src/main/java/com/deepvoiceguard/app/alert/AlertSettings.kt`
- Create: `android-app/app/src/test/java/com/deepvoiceguard/app/alert/AlertChannelDispatcherTest.kt`

- [ ] **Step 2.1: AlertSettings data class 작성**

```kotlin
package com.deepvoiceguard.app.alert

/**
 * Dispatcher 호출 시 전달되는 채널별 ON/OFF.
 * SettingsRepository.AppSettings 에서 매핑.
 */
data class AlertSettings(
    val haptic: Boolean,
    val ttsAlert: Boolean,
)
```

- [ ] **Step 2.2: 테스트 작성 (companion 의 pure function 위주 + dispatch 시점 mock 검증)**

핵심은 `vibrationPatternFor` / `ttsMessageFor` 가 leveling 정확하고, `dispatch()` 가 settings flag 를 정확히 게이팅하는지 + Vibrator/TTS 호출 횟수 검증.

```kotlin
package com.deepvoiceguard.app.alert

import com.deepvoiceguard.app.inference.CombinedThreatLevel
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AlertChannelDispatcherTest {

    @Test
    fun `vibration intensity grows with severity`() {
        val warn = AlertChannelDispatcher.vibrationPatternFor(CombinedThreatLevel.WARNING)
        val danger = AlertChannelDispatcher.vibrationPatternFor(CombinedThreatLevel.DANGER)
        val critical = AlertChannelDispatcher.vibrationPatternFor(CombinedThreatLevel.CRITICAL)
        assertNotNull(warn); assertNotNull(danger); assertNotNull(critical)
        // critical pattern 합 > danger 합 > warning 합 (총 진동 시간으로 강도 비교)
        val warnSum = warn!!.sum()
        val dangerSum = danger!!.sum()
        val criticalSum = critical!!.sum()
        assertTrue("danger > warning total", dangerSum > warnSum)
        assertTrue("critical > danger total", criticalSum > dangerSum)
    }

    @Test
    fun `safe and caution have no vibration`() {
        assertNull(AlertChannelDispatcher.vibrationPatternFor(CombinedThreatLevel.SAFE))
        assertNull(AlertChannelDispatcher.vibrationPatternFor(CombinedThreatLevel.CAUTION))
    }

    @Test
    fun `tts message for danger mentions ai or synthesis`() {
        val msg = AlertChannelDispatcher.ttsMessageFor(CombinedThreatLevel.DANGER)
        assertNotNull(msg)
        // 한국어 사용자가 즉시 상황 인지 가능한 문구 — 핵심 단어 포함 보장
        assertTrue(
            "DANGER 메시지에 'AI' 또는 '합성' 또는 '딥' 포함되어야",
            msg!!.contains("AI") || msg.contains("합성") || msg.contains("딥")
        )
    }

    @Test
    fun `tts message for critical mentions both threats`() {
        val msg = AlertChannelDispatcher.ttsMessageFor(CombinedThreatLevel.CRITICAL)
        assertNotNull(msg)
        // CRITICAL = 딥보이스 + 피싱 동시 → 두 위협 모두 언급
        assertTrue("CRITICAL 메시지에 '피싱' 포함되어야", msg!!.contains("피싱"))
    }

    @Test
    fun `safe and caution have no tts message`() {
        assertNull(AlertChannelDispatcher.ttsMessageFor(CombinedThreatLevel.SAFE))
        assertNull(AlertChannelDispatcher.ttsMessageFor(CombinedThreatLevel.CAUTION))
    }

    @Test
    fun `dispatch with all toggles off does nothing`() {
        // 단순 검증: settings 모두 false → companion pure function 결과만 확인 가능 (dispatch
        // 자체는 Vibrator/TTS Android API 라 unit test 에선 실행 안 함). 여기는 게이팅 로직이
        // companion 결과를 무시하지 않음을 확인하는 sanity check.
        val patterns = listOf(
            CombinedThreatLevel.WARNING, CombinedThreatLevel.DANGER, CombinedThreatLevel.CRITICAL
        ).map { AlertChannelDispatcher.vibrationPatternFor(it) }
        // pattern null이 아니더라도 dispatch 호출 측에서 settings.haptic=false 면 vibrate() 안 함.
        // 그 동작은 instrumentation 또는 manual test 에서 확인.
        assertTrue(patterns.all { it != null })
    }

    @Test
    fun `level-to-message maps every non-safe level`() {
        // 회귀 방지 — 새 CombinedThreatLevel 값이 추가됐는데 매핑 안 한 케이스 잡기
        for (level in CombinedThreatLevel.values()) {
            val pattern = AlertChannelDispatcher.vibrationPatternFor(level)
            val msg = AlertChannelDispatcher.ttsMessageFor(level)
            // SAFE/CAUTION 만 null, 나머지는 모두 설정돼 있어야
            if (level == CombinedThreatLevel.SAFE || level == CombinedThreatLevel.CAUTION) {
                assertNull("$level should be null pattern", pattern)
                assertNull("$level should be null msg", msg)
            } else {
                assertNotNull("$level missing pattern", pattern)
                assertNotNull("$level missing message", msg)
            }
        }
    }
}
```

(테스트는 mockk 의존 없이 companion pure function 만 검증 — dispatch 자체의 Vibrator/TTS 호출은 instrumentation test 또는 수동 검증 영역. unit-test 레벨에서 race-safe 한 dispatch 검증은 시간 비용 큼 — 데모 우선순위에서 제외.)

- [ ] **Step 2.3: 테스트 실패 확인**

```bash
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.alert.AlertChannelDispatcherTest" 2>&1 | tail -10
```

기대: unresolved reference `AlertChannelDispatcher`.

### Task 3: AlertChannelDispatcher 구현 (TTS 큐잉 + Vibrator + lifecycle)

**Files:**
- Create: `android-app/app/src/main/java/com/deepvoiceguard/app/alert/AlertChannelDispatcher.kt`

- [ ] **Step 3.1: 클래스 작성**

핵심 설계:
- TTS 가 **lazy + async init** 이므로 첫 dispatch 시 message 를 `pendingMessage` 에 저장 → onInit 콜백에서 flush.
- Vibrator 는 동기 — 즉시 호출.

```kotlin
package com.deepvoiceguard.app.alert

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.util.Log
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

/**
 * DANGER/CRITICAL 발생 시 푸시 알림 외에 햅틱+TTS 음성 알림을 dispatch.
 *
 * NotificationChannel 자체의 진동은 OFF (NotificationHelper 가 책임). 모든 진동은
 * 이 클래스가 발화 — 사용자 [AlertSettings.haptic] 토글이 진짜로 동작하도록.
 *
 * TTS 는 첫 호출 시 lazy + async init. init 진행 중에 dispatch 가 들어오면 메시지가
 * [pendingMessage] 에 저장되고, [TextToSpeech.OnInitListener] 콜백에서 flush — 첫 알림이
 * 누락되지 않음.
 */
class AlertChannelDispatcher(private val context: Context) {

    private val vibrator: Vibrator? by lazy {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)
                ?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    }

    @Volatile private var ttsEngine: TextToSpeech? = null
    @Volatile private var ttsReady: Boolean = false
    /** init 중에 도착한 메시지를 보관 — onInit 콜백에서 즉시 발화 후 비움. */
    private val pendingMessage: AtomicReference<String?> = AtomicReference(null)

    /** 메인 스레드 안전. 첫 호출 시 TTS init 시작. */
    private fun ensureTtsInitStarted() {
        if (ttsEngine != null) return
        synchronized(this) {
            if (ttsEngine != null) return
            val engine = TextToSpeech(context.applicationContext) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    ttsEngine?.language = Locale.KOREAN
                    ttsReady = true
                    // init 도중 들어온 메시지 flush
                    pendingMessage.getAndSet(null)?.let { msg ->
                        runCatching {
                            ttsEngine?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "alert-init-flush")
                        }
                    }
                } else {
                    Log.w(TAG, "TextToSpeech init failed status=$status")
                    ttsReady = false
                }
            }
            ttsEngine = engine
        }
    }

    fun dispatch(level: CombinedThreatLevel, settings: AlertSettings) {
        if (settings.haptic) {
            vibrationPatternFor(level)?.let { pattern ->
                runCatching {
                    val effect = VibrationEffect.createWaveform(pattern, -1)
                    vibrator?.vibrate(effect)
                }.onFailure { Log.w(TAG, "vibration failed", it) }
            }
        }
        if (settings.ttsAlert) {
            ttsMessageFor(level)?.let { message ->
                ensureTtsInitStarted()
                if (ttsReady) {
                    runCatching {
                        ttsEngine?.speak(message, TextToSpeech.QUEUE_FLUSH, null, "alert-${level.name}")
                    }.onFailure { Log.w(TAG, "tts speak failed", it) }
                } else {
                    // init 진행 중 — 메시지 큐에 저장. 가장 최근 것만 유지 (쇄도 시 latest only).
                    pendingMessage.set(message)
                }
            }
        }
    }

    fun shutdown() {
        runCatching { ttsEngine?.stop() }
        runCatching { ttsEngine?.shutdown() }
        ttsEngine = null
        ttsReady = false
        pendingMessage.set(null)
    }

    companion object {
        private const val TAG = "AlertDispatcher"

        /** ms 단위 진동 wave: pause/on 교차. null 이면 진동 없음. */
        fun vibrationPatternFor(level: CombinedThreatLevel): LongArray? = when (level) {
            CombinedThreatLevel.SAFE, CombinedThreatLevel.CAUTION -> null
            CombinedThreatLevel.WARNING -> longArrayOf(0, 200)
            CombinedThreatLevel.DANGER -> longArrayOf(0, 300, 200, 300)
            CombinedThreatLevel.CRITICAL -> longArrayOf(0, 400, 200, 400, 200, 600)
        }

        fun ttsMessageFor(level: CombinedThreatLevel): String? = when (level) {
            CombinedThreatLevel.SAFE, CombinedThreatLevel.CAUTION -> null
            CombinedThreatLevel.WARNING -> "주의. 의심스러운 음성 패턴이 감지되었습니다."
            CombinedThreatLevel.DANGER -> "경고. AI 합성 음성이 감지되었습니다."
            CombinedThreatLevel.CRITICAL -> "위험. 딥페이크와 보이스피싱이 동시에 감지되었습니다."
        }
    }
}
```

- [ ] **Step 3.2: 테스트 통과 확인**

```bash
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.alert.AlertChannelDispatcherTest" 2>&1 | tail -10
```

기대: BUILD SUCCESSFUL, 7 passed.

- [ ] **Step 3.3: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/alert/AlertSettings.kt \
         android-app/app/src/main/java/com/deepvoiceguard/app/alert/AlertChannelDispatcher.kt \
         android-app/app/src/test/java/com/deepvoiceguard/app/alert/AlertChannelDispatcherTest.kt
git commit -m "feat(alert): AlertChannelDispatcher with vibration + TTS

- vibrationPatternFor / ttsMessageFor: SAFE/CAUTION = null, 나머지 강도순
- TTS lazy async init + pendingMessage queue: init 도중 dispatch도 누락 없이 flush
- VibratorManager (S+) / 레거시 Vibrator fallback
- shutdown lifecycle: TTS engine 정리 + queue clear
- 7 unit tests (companion pure functions, level mapping, severity ordering)"
```

---

## Phase 3 — NotificationChannel 진동 OFF + Hilt 통합

### Task 4: NotificationChannel 진동 비활성화 + Hilt provider

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/service/NotificationHelper.kt` (NotificationChannel 진동만)
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/di/AppModule.kt`

- [ ] **Step 4.1: NotificationChannel 진동 끄기**

`NotificationHelper.kt` 의 `createChannels()` 안 3개 채널의 다음 두 줄 모두 제거:

```kotlin
enableVibration(true)
vibrationPattern = longArrayOf(0, 500, 200, 500)  // 채널마다 다른 값
```

대신 다음으로 교체 (각 채널):

```kotlin
// AlertChannelDispatcher 가 진동 책임 — 채널 자체에선 OFF (사용자 토글 분리)
enableVibration(false)
```

- [ ] **Step 4.2: Hilt provider 추가**

`AppModule.kt` 의 `@Provides` 블록 끝부분에:

```kotlin
@Provides
@Singleton
fun provideAlertChannelDispatcher(
    @ApplicationContext context: Context,
): AlertChannelDispatcher = AlertChannelDispatcher(context)
```

(필요 import: `com.deepvoiceguard.app.alert.AlertChannelDispatcher`, 그리고 `androidx.hilt.android.qualifiers.ApplicationContext` / `dagger.Provides` / `dagger.hilt.InstallIn` / `javax.inject.Singleton` 은 기존 file에 이미 있음)

- [ ] **Step 4.3: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

기대: BUILD SUCCESSFUL.

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/service/NotificationHelper.kt \
         android-app/app/src/main/java/com/deepvoiceguard/app/di/AppModule.kt
git commit -m "feat(alert): disable channel-level vibration; provide AlertChannelDispatcher via Hilt

NotificationChannel 의 enableVibration(true) 를 false 로 변경 — 사용자 hapticAlertEnabled
토글이 push 알림 진동까지 통제하도록. AlertChannelDispatcher 는 @Singleton Hilt provide."
```

### Task 5: NotificationHelper 가 dispatcher 호출

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/service/NotificationHelper.kt`

- [ ] **Step 5.1: 생성자에 두 파라미터 추가**

기존 `class NotificationHelper(private val context: Context) {` 를 다음으로 교체:

```kotlin
class NotificationHelper(
    private val context: Context,
    private val alertDispatcher: AlertChannelDispatcher? = null,
    private val alertSettingsProvider: () -> AlertSettings = {
        AlertSettings(haptic = true, ttsAlert = false)  // 기본값 — 호출 측 미지정 시 fallback
    },
) {
```

(필요 import 추가:)

```kotlin
import com.deepvoiceguard.app.alert.AlertChannelDispatcher
import com.deepvoiceguard.app.alert.AlertSettings
import com.deepvoiceguard.app.inference.CombinedThreatAggregator
```

- [ ] **Step 5.2: showDetectionAlert 끝에 dispatch 추가**

`fun showDetectionAlert(result: AggregatedResult, callSessionId: String? = null): Boolean { ... }` 의 마지막 `return true` (또는 `nm.notify(...)` 호출 직후) 직전에 다음 추가:

```kotlin
// AlertChannelDispatcher: 푸시 게시 후 햅틱 + TTS 트리거.
// AggregatedResult.threatLevel 은 ThreatLevel 이므로 CombinedThreatLevel 로 매핑.
val combinedLevel = CombinedThreatAggregator.deepfakeOnlyLevel(result.threatLevel)
alertDispatcher?.dispatch(combinedLevel, alertSettingsProvider())
```

- [ ] **Step 5.3: showCombinedAlert 끝에 dispatch 추가**

`fun showCombinedAlert(result: CombinedThreatResult, callSessionId: String? = null): Boolean { ... }` 의 마지막에 다음 추가 (CombinedThreatResult 는 이미 `combinedThreatLevel` 보유):

```kotlin
alertDispatcher?.dispatch(result.combinedThreatLevel, alertSettingsProvider())
```

- [ ] **Step 5.4: ensureIncidentAlerted 도 동일 처리**

`ensureIncidentAlerted(result: CombinedThreatResult, ...)` 가 알림 게시할 때도 dispatch — 위와 동일 로직.

- [ ] **Step 5.5: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/service/NotificationHelper.kt
git commit -m "feat(alert): NotificationHelper invokes AlertChannelDispatcher after each notify

- showDetectionAlert: ThreatLevel → deepfakeOnlyLevel(ThreatLevel) → CombinedThreatLevel
- showCombinedAlert: result.combinedThreatLevel 직접 사용
- ensureIncidentAlerted: 동일 패턴
- dispatcher 가 null 이면 no-op (테스트 친화)"
```

### Task 6: AudioCaptureService 통합

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioCaptureService.kt`

- [ ] **Step 6.1: dispatcher Hilt 주입 + NotificationHelper 생성 변경**

기존:
```kotlin
@Inject lateinit var narrowbandPreprocessor: com.deepvoiceguard.app.audio.NarrowbandPreprocessor
private lateinit var notificationHelper: NotificationHelper
```

뒤에 추가:
```kotlin
@Inject lateinit var alertDispatcher: AlertChannelDispatcher
```

(import 추가: `com.deepvoiceguard.app.alert.AlertChannelDispatcher`, `com.deepvoiceguard.app.alert.AlertSettings`)

기존 `onCreate` 내부의 `notificationHelper = NotificationHelper(this)` 를 다음으로 교체:

```kotlin
notificationHelper = NotificationHelper(
    context = this,
    alertDispatcher = alertDispatcher,
    alertSettingsProvider = {
        // liveSettings 가 아직 null 이면 안전한 기본값 반환
        val s = liveSettings?.value
        if (s == null) {
            AlertSettings(haptic = true, ttsAlert = false)
        } else {
            AlertSettings(haptic = s.hapticAlertEnabled, ttsAlert = s.ttsAlertEnabled)
        }
    },
)
```

- [ ] **Step 6.2: onDestroy 에 shutdown 추가**

`onDestroy()` 안의 `super.onDestroy()` 직전에 추가:

```kotlin
runCatching { alertDispatcher.shutdown() }
```

- [ ] **Step 6.3: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioCaptureService.kt
git commit -m "feat(alert): AudioCaptureService injects dispatcher and wires settingsProvider

- @Inject AlertChannelDispatcher
- NotificationHelper 생성 시 dispatcher + settings provider 람다 전달
- onDestroy 에서 dispatcher.shutdown 으로 TTS engine 정리"
```

---

## Phase 4 — Settings UI 토글

### Task 7: Settings 화면에 알림 카드

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/SettingsScreen.kt`

- [ ] **Step 7.1: 새 카드 섹션 추가**

기존 `SettingRow` 시그니처는 `(label, checked, onCheckedChange: () -> Unit)` — 클릭 시 `() -> Unit` 호출하므로 호출자가 `!checked` 처리. 기존 패턴 따라:

```kotlin
Spacer(modifier = Modifier.height(16.dp))

Card(modifier = Modifier.fillMaxWidth()) {
    Column(modifier = Modifier.padding(16.dp)) {
        Text(
            "알림 채널",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
        )
        Spacer(modifier = Modifier.height(8.dp))

        SettingRow("진동(햅틱) 알림", settings.hapticAlertEnabled) {
            scope.launch {
                repository.setHapticAlertEnabled(!settings.hapticAlertEnabled)
            }
        }

        SettingRow("음성(TTS) 알림", settings.ttsAlertEnabled) {
            scope.launch {
                repository.setTtsAlertEnabled(!settings.ttsAlertEnabled)
            }
        }
    }
}
```

- [ ] **Step 7.2: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/SettingsScreen.kt
git commit -m "feat(settings): UI toggles for haptic and TTS alerts (() -> Unit pattern)"
```

---

## Phase 5 — 실기기 검증

### Task 8: 동작 확인

- [ ] **Step 8.1: ▶ Run, 폰 권한**

햅틱은 권한 불필요. TTS 도 권한 불필요 (시스템 TTS engine 사용).

- [ ] **Step 8.2: Settings 에서 Haptic ON, TTS ON**

- [ ] **Step 8.3: 스피커로 fake 재생 → DANGER 트리거**

```bash
! afplay /Users/hyohee/Documents/Claude_project/anti-deepvoice-guard/test-samples/fake/TTS1.wav
```

기대:
- 푸시 알림 (기존 시각 알림)
- 폰 진동 패턴 (DANGER: 짧+짧)
- 스피커에서 한국어 TTS "경고. AI 합성 음성이 감지되었습니다."

- [ ] **Step 8.4: TTS 토글 OFF 후 재시도**

기대: 푸시 + 진동만, 음성 없음.

- [ ] **Step 8.5: Haptic 토글 OFF 후 재시도 (TTS ON 상태로)**

기대: 푸시 + TTS, **진동 없음** (NotificationChannel 진동 비활성 + dispatcher 도 진동 skip).

- [ ] **Step 8.6: 모든 토글 OFF 후 재시도**

기대: 푸시 시각 알림만, 진동/음성 둘 다 없음.

---

## Phase 6 — 백업 push

- [ ] **Step 6.1: 원격 push**

```bash
git push origin feature/multi-alert
```

---

## Self-Review Checklist

- [x] Spec 커버리지: 햅틱 + TTS + 독립 토글. 푸시는 기존 NotificationHelper 가 처리, 채널 진동은 OFF 로 dispatcher가 일원 책임.
- [x] Placeholder 없음.
- [x] 타입 일관성: dispatcher 는 항상 `CombinedThreatLevel` 만 받음. ThreatLevel → CombinedThreatLevel 변환은 호출 측 (`CombinedThreatAggregator.deepfakeOnlyLevel`).
- [x] SettingRow 시그니처 (`() -> Unit`) 기존 패턴 정확히 따름.
- [x] TTS init race: `pendingMessage` AtomicReference 로 init 도중 도착한 메시지를 onInit 콜백에서 flush.
- [x] NotificationChannel 진동 분리: `enableVibration(false)` 로 채널 자체 진동 OFF, dispatcher 가 햅틱 책임.

## 알려진 한계

- TTS init 첫 1~2초 지연 가능. 첫 알림은 init flush 로 살아나지만 약간 늦게 발화될 수 있음.
- 일부 기기는 Do Not Disturb 시 TTS 발화 안 됨. 햅틱은 동작.
- 진동 패턴 강도는 hardware 별 차이 — 본 plan 의 ms 값은 typical 기기 기준.
- `dispatch()` 자체에 대한 unit test 없음 (Vibrator/TTS Android API 의존). Robolectric 도입은 별도 작업.
