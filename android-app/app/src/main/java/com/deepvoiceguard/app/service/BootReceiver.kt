package com.deepvoiceguard.app.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/**
 * 부팅 완료 시 AudioCaptureService를 자동 시작한다.
 * SharedPreferences의 auto_start_on_boot 설정에 따라 동작.
 */
class BootReceiver : BroadcastReceiver() {

    companion object {
        private const val PREF_NAME = "phone_state_prefs"
        private const val KEY_AUTO_BOOT = "auto_start_on_boot"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return

        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val autoStart = prefs.getBoolean(KEY_AUTO_BOOT, false)

        if (autoStart) {
            val serviceIntent = Intent(context, AudioCaptureService::class.java).apply {
                action = AudioCaptureService.ACTION_START
            }
            context.startForegroundService(serviceIntent)
        }
    }
}
