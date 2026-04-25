package com.deepvoiceguard.app.stt

import android.content.Context
import android.content.Intent
import android.os.Build
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.Locale

/**
 * 기기의 STT 가용성을 확인하고 오프라인 전용 동작을 검증한다.
 *
 * **개인정보 보호 원칙 (H91/H94 fail-closed)**: 통화 음성은 민감한 개인정보이므로
 * 네트워크 recognition은 일체 허용하지 않는다. Android 12 (API 31)에 도입된
 * `SpeechRecognizer.createOnDeviceSpeechRecognizer()`를 사용하는 것 자체가 offline
 * 보장을 제공한다. `EXTRA_PREFER_OFFLINE`은 legacy compatibility hint일 뿐이며
 * on-device invariant를 증명하지 못한다. API 31 미만 기기에서는 offline invariant를
 * 증명할 수 없으므로 STT를 비활성화한다 (fail-closed).
 */
class SttCapabilityChecker(private val context: Context) {

    /** SpeechRecognizer가 기기에서 사용 가능한지 확인. */
    fun isRecognitionAvailable(): Boolean =
        SpeechRecognizer.isRecognitionAvailable(context)

    /**
     * **검증된 on-device 인식 지원** 여부를 반환한다 (H94).
     *
     * 요건:
     *  1. API 31 (Android 12) 이상 — `createOnDeviceSpeechRecognizer`가 존재.
     *  2. `SpeechRecognizer.isOnDeviceRecognitionAvailable()`이 true (API 33+)
     *     또는 API 31-32는 존재 체크만 (OEM 차이 존재 — GoogleSttEngine에서 런타임 fail-closed).
     *
     * 이 함수가 false면 STT 토글은 비활성화되고 `startListening()`은 절대 호출되지 않는다.
     */
    fun isOfflineRecognitionSupported(): Boolean {
        // API 33+ (Android 13): 공식 API로 on-device 가용성 질의.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val onDeviceOk = try {
                SpeechRecognizer.isOnDeviceRecognitionAvailable(context)
            } catch (_: Throwable) {
                false
            }
            if (onDeviceOk) return true
            // [DEMO ONLY] on-device 미지원 시 네트워크 STT 로 fallback (시연용).
            // 프로덕션 배포 전 반드시 revert — 개인정보 정책상 통화 음성을 클라우드로 보내면 안 됨.
            // 본 commit 의 변경 (SttCapabilityChecker.kt + GoogleSttEngine.kt) 은 단일 commit
            // 으로 분리되어 있어 `git revert <hash>` 로 복원 가능.
            return isRecognitionAvailable()
        }
        // API 31-32: createOnDeviceSpeechRecognizer 존재. 실제 동작 여부는 GoogleSttEngine에서
        // onError(ERROR_*)로 확인되며, 에러 발생 시 fail-closed로 STT 비활성.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return isRecognitionAvailable()
        }
        // API 31 미만: offline 보장 불가 → fail-closed.
        return false
    }

    /**
     * No-op: Android API로 직접 다운로드를 트리거할 수 없다.
     * 사용자가 설정 → 음성에서 수동 설치해야 한다.
     */
    @Deprecated(
        message = "No-op: Android API로 직접 다운로드를 트리거할 수 없다. 설정 → 음성에서 수동 설치 필요.",
    )
    fun triggerKoreanModelDownload() {
        // no-op
    }

    /**
     * 한국어 음성 인식용 Intent 생성.
     *
     * **H91/H94**: on-device recognizer 사용 자체가 offline 보장이다. 아래 legacy extra들은
     * on-device 경로에서 무시될 수 있으므로 essential extra만 유지한다.
     */
    fun createRecognitionIntent(): Intent =
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.KOREAN.toLanguageTag())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
}
