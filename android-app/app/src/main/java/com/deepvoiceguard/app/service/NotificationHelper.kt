package com.deepvoiceguard.app.service

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.deepvoiceguard.app.R
import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.ThreatLevel
import com.deepvoiceguard.app.ui.MainActivity

/**
 * Deepfake 탐지 알림을 생성하고 표시한다.
 */
class NotificationHelper(private val context: Context) {

    companion object {
        const val ALERT_CHANNEL_ID = "deepvoice_alert_channel"
        private var notificationId = 1000
    }

    init {
        createAlertChannel()
    }

    private fun createAlertChannel() {
        val channel = NotificationChannel(
            ALERT_CHANNEL_ID,
            "Deepfake Alerts",
            NotificationManager.IMPORTANCE_HIGH,
        ).apply {
            description = "Alerts when deepfake voice is detected"
            enableVibration(true)
            vibrationPattern = longArrayOf(0, 500, 200, 500)
        }
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(channel)
    }

    fun showDetectionAlert(result: AggregatedResult) {
        if (result.threatLevel == ThreatLevel.SAFE) return

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) return
        }

        val (title, text) = when (result.threatLevel) {
            ThreatLevel.DANGER -> "Deepfake Detected!" to
                    "High confidence AI voice detected (${(result.averageFakeScore * 100).toInt()}%)"
            ThreatLevel.WARNING -> "Suspicious Voice" to
                    "Multiple suspicious segments detected (${(result.averageFakeScore * 100).toInt()}%)"
            ThreatLevel.CAUTION -> "Elevated Risk" to
                    "Slightly elevated deepfake score (${(result.averageFakeScore * 100).toInt()}%)"
            else -> return
        }

        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            context, 0, intent, PendingIntent.FLAG_IMMUTABLE,
        )

        val priority = when (result.threatLevel) {
            ThreatLevel.DANGER -> NotificationCompat.PRIORITY_MAX
            ThreatLevel.WARNING -> NotificationCompat.PRIORITY_HIGH
            else -> NotificationCompat.PRIORITY_DEFAULT
        }

        val notification = NotificationCompat.Builder(context, ALERT_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentTitle(title)
            .setContentText(text)
            .setPriority(priority)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()

        NotificationManagerCompat.from(context).notify(notificationId++, notification)
    }
}
