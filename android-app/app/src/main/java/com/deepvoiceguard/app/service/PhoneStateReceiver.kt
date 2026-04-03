package com.deepvoiceguard.app.service

import android.app.ActivityManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager

/**
 * 전화 상태 변화를 감지하여 AudioCaptureService를 자동 시작/중지한다.
 *
 * Android 14+ 백그라운드 FGS 제한 대응:
 * - 서비스가 이미 실행 중이면: 통화 모드로 전환 (ACTION_START_CALL)
 * - 서비스가 미실행이면: 사용자가 직접 앱에서 시작해야 함
 *
 * 상태 전이:
 *  IDLE → RINGING : 수신 전화 (아직 시작 안 함)
 *  RINGING → OFFHOOK : 수신 전화 응답 → 모니터링 시작/전환
 *  IDLE → OFFHOOK : 발신 전화 → 모니터링 시작/전환
 *  OFFHOOK → IDLE : 통화 종료 → 모니터링 중지
 *  RINGING → IDLE : 부재중/거절 (무시)
 */
class PhoneStateReceiver : BroadcastReceiver() {

    companion object {
        private const val PREF_NAME = "phone_state_prefs"
        private const val KEY_LAST_STATE = "last_telephony_state"
        private const val KEY_AUTO_MONITOR = "auto_start_on_call"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return

        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val autoMonitor = prefs.getBoolean(KEY_AUTO_MONITOR, true)
        if (!autoMonitor) return

        val newState = intent.getStringExtra(TelephonyManager.EXTRA_STATE) ?: return
        val lastState = prefs.getString(KEY_LAST_STATE, TelephonyManager.EXTRA_STATE_IDLE)
            ?: TelephonyManager.EXTRA_STATE_IDLE

        @Suppress("DEPRECATION")
        val phoneNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)

        val transition = resolveTransition(lastState, newState)

        when (transition) {
            is CallTransition.StartCall -> {
                // Android 14+ 에서는 백그라운드에서 microphone FGS 시작 불가
                // 서비스가 이미 실행 중일 때만 통화 모드로 전환
                if (isServiceRunning(context)) {
                    val serviceIntent = Intent(context, AudioCaptureService::class.java).apply {
                        action = AudioCaptureService.ACTION_START_CALL
                        putExtra(AudioCaptureService.EXTRA_CALL_DIRECTION, transition.direction.name)
                        putExtra(AudioCaptureService.EXTRA_PHONE_NUMBER, phoneNumber)
                    }
                    context.startService(serviceIntent)
                } else {
                    // 서비스 미실행 — 앱이 포그라운드일 때 startForegroundService 시도
                    try {
                        val serviceIntent = Intent(context, AudioCaptureService::class.java).apply {
                            action = AudioCaptureService.ACTION_START_CALL
                            putExtra(AudioCaptureService.EXTRA_CALL_DIRECTION, transition.direction.name)
                            putExtra(AudioCaptureService.EXTRA_PHONE_NUMBER, phoneNumber)
                        }
                        context.startForegroundService(serviceIntent)
                    } catch (_: Exception) {
                        // 백그라운드 제한으로 시작 실패 — 무시 (사용자가 앱에서 직접 시작)
                    }
                }
            }
            is CallTransition.StopCall -> {
                val serviceIntent = Intent(context, AudioCaptureService::class.java).apply {
                    action = AudioCaptureService.ACTION_STOP_CALL
                }
                runCatching { context.startService(serviceIntent) }
            }
            CallTransition.Ignore -> { /* 무시 */ }
        }

        prefs.edit().putString(KEY_LAST_STATE, newState).commit()
    }

    @Suppress("DEPRECATION")
    private fun isServiceRunning(context: Context): Boolean {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return am.getRunningServices(Integer.MAX_VALUE)
            .any { it.service.className == AudioCaptureService::class.java.name }
    }

    private fun resolveTransition(lastState: String, newState: String): CallTransition {
        return when {
            // 수신 전화 응답
            lastState == TelephonyManager.EXTRA_STATE_RINGING
                && newState == TelephonyManager.EXTRA_STATE_OFFHOOK ->
                CallTransition.StartCall(CallDirection.INCOMING)

            // 발신 전화
            lastState == TelephonyManager.EXTRA_STATE_IDLE
                && newState == TelephonyManager.EXTRA_STATE_OFFHOOK ->
                CallTransition.StartCall(CallDirection.OUTGOING)

            // 통화 종료
            lastState == TelephonyManager.EXTRA_STATE_OFFHOOK
                && newState == TelephonyManager.EXTRA_STATE_IDLE ->
                CallTransition.StopCall

            // 부재중/거절, RINGING→IDLE 등
            else -> CallTransition.Ignore
        }
    }

    private sealed class CallTransition {
        data class StartCall(val direction: CallDirection) : CallTransition()
        data object StopCall : CallTransition()
        data object Ignore : CallTransition()
    }
}
