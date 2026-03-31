package com.deepvoiceguard.app.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.deepvoiceguard.app.R
import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.InferenceEngine
import com.deepvoiceguard.app.inference.OnDeviceEngine
import com.deepvoiceguard.app.ui.MainActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Foreground Service로 마이크 오디오를 캡처하고 AudioPipeline에 공급한다.
 */
class AudioCaptureService : Service() {

    companion object {
        const val CHANNEL_ID = "deepvoice_guard_channel"
        const val NOTIFICATION_ID = 1
        const val SAMPLE_RATE = 16000
        const val BUFFER_SIZE_SAMPLES = 512
        const val ACTION_START = "com.deepvoiceguard.ACTION_START"
        const val ACTION_STOP = "com.deepvoiceguard.ACTION_STOP"
    }

    inner class LocalBinder : Binder() {
        val pipeline: AudioPipeline? get() = this@AudioCaptureService.pipeline
    }

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private var audioRecord: AudioRecord? = null
    private var pipeline: AudioPipeline? = null
    private var vadEngine: VadEngine? = null
    private var inferenceEngine: InferenceEngine? = null
    private var isRecording = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startCapture()
            ACTION_STOP -> stopCapture()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder = binder

    private fun startCapture() {
        if (isRecording) return

        startForeground(NOTIFICATION_ID, createNotification("Monitoring..."))

        // 엔진 초기화
        vadEngine = VadEngine(this)
        inferenceEngine = OnDeviceEngine(this)
        pipeline = AudioPipeline(vadEngine!!, inferenceEngine!!, serviceScope)

        // AudioRecord 설정
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(BUFFER_SIZE_SAMPLES * 2)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        )

        isRecording = true
        audioRecord?.startRecording()

        // 오디오 읽기 루프
        serviceScope.launch(Dispatchers.IO) {
            val buffer = ShortArray(BUFFER_SIZE_SAMPLES)
            while (isActive && isRecording) {
                val readSize = audioRecord?.read(buffer, 0, BUFFER_SIZE_SAMPLES) ?: -1
                if (readSize > 0) {
                    pipeline?.processAudioChunk(buffer, readSize)
                }
            }
        }
    }

    private fun stopCapture() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        vadEngine?.close()
        inferenceEngine?.close()
        pipeline = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        stopCapture()
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "DeepVoice Guard",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Audio monitoring service"
        }
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(channel)
    }

    private fun createNotification(text: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent, PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Anti-DeepVoice Guard")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
