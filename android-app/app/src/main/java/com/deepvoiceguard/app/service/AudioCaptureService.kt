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
import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.InferenceEngine
import com.deepvoiceguard.app.inference.OnDeviceEngine
import com.deepvoiceguard.app.inference.ServerEngine
import com.deepvoiceguard.app.storage.AppSettings
import com.deepvoiceguard.app.storage.DetectionDao
import com.deepvoiceguard.app.storage.DetectionEntity
import com.deepvoiceguard.app.storage.EncryptedStorage
import com.deepvoiceguard.app.storage.SettingsRepository
import com.deepvoiceguard.app.ui.MainActivity
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Foreground Service로 마이크 오디오를 캡처하고 AudioPipeline에 공급한다.
 */
@AndroidEntryPoint
class AudioCaptureService : Service() {

    @Inject lateinit var detectionDao: DetectionDao
    @Inject lateinit var encryptedStorage: EncryptedStorage
    @Inject lateinit var settingsRepository: SettingsRepository
    private lateinit var notificationHelper: NotificationHelper

    companion object {
        const val CHANNEL_ID = "deepvoice_guard_channel"
        const val NOTIFICATION_ID = 1
        const val SAMPLE_RATE = 16000
        const val BUFFER_SIZE_SAMPLES = 512
        const val ACTION_START = "com.deepvoiceguard.ACTION_START"
        const val ACTION_STOP = "com.deepvoiceguard.ACTION_STOP"
        const val ACTION_START_CALL = "com.deepvoiceguard.ACTION_START_CALL"
        const val ACTION_STOP_CALL = "com.deepvoiceguard.ACTION_STOP_CALL"
        const val EXTRA_CALL_DIRECTION = "call_direction"
        const val EXTRA_PHONE_NUMBER = "phone_number"
    }

    // 서비스 상태를 live observable로 노출
    private val _isMonitoring = MutableStateFlow(false)
    val isMonitoring: StateFlow<Boolean> = _isMonitoring

    private val _latestResult = MutableStateFlow<AggregatedResult?>(null)
    val latestResult: StateFlow<AggregatedResult?> = _latestResult

    private val _vadProbability = MutableStateFlow(0f)
    val vadProbability: StateFlow<Float> = _vadProbability

    private val _stats = MutableStateFlow(PipelineStats(0, 0))
    val stats: StateFlow<PipelineStats> = _stats

    private val _callSession = MutableStateFlow<CallSession?>(null)
    val callSession: StateFlow<CallSession?> = _callSession

    inner class LocalBinder : Binder() {
        val service: AudioCaptureService get() = this@AudioCaptureService
    }

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private var audioRecord: AudioRecord? = null
    private var pipeline: AudioPipeline? = null
    private var vadEngine: VadEngine? = null
    private var inferenceEngine: InferenceEngine? = null
    @Volatile
    private var isRecording = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        notificationHelper = NotificationHelper(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> serviceScope.launch { startCapture() }
            ACTION_STOP -> stopCapture()
            ACTION_START_CALL -> {
                val direction = intent.getStringExtra(EXTRA_CALL_DIRECTION)
                    ?.let { CallDirection.valueOf(it) } ?: CallDirection.UNKNOWN
                val phoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
                val session = CallSession(
                    direction = direction,
                    phoneNumber = phoneNumber,
                )
                serviceScope.launch { startCapture(session) }
            }
            ACTION_STOP_CALL -> {
                _callSession.value = _callSession.value?.end()
                stopCapture()
            }
            null -> {
                // START_STICKY에 의한 재시작 — intent가 null이면 안전하게 종료
                stopSelf()
            }
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder = binder

    private suspend fun startCapture(callSession: CallSession? = null) {
        if (isRecording) return

        _callSession.value = callSession
        val notifText = if (callSession != null) {
            val dir = if (callSession.direction == CallDirection.INCOMING) "수신" else "발신"
            "통화 모니터링 중 ($dir)"
        } else "Monitoring..."
        startForeground(NOTIFICATION_ID, createNotification(notifText))

        // 설정값 로드
        val settings: AppSettings = settingsRepository.settings.first()

        // 설정에 따른 엔진 초기화
        vadEngine = VadEngine(this)
        val onDeviceEngine = OnDeviceEngine(this)
        inferenceEngine = if (settings.useOnDevice) {
            onDeviceEngine
        } else {
            ServerEngine(settings.serverUrl, fallback = onDeviceEngine)
        }
        val newPipeline = AudioPipeline(
            vadEngine!!, inferenceEngine!!, serviceScope, dangerThreshold = settings.threshold,
        )
        pipeline = newPipeline

        // 파이프라인 상태를 서비스 레벨 StateFlow로 포워딩
        serviceScope.launch {
            newPipeline.latestResult.collect { result ->
                _latestResult.value = result
                _stats.value = newPipeline.getStats()
            }
        }
        serviceScope.launch {
            newPipeline.vadProbability.collect { prob ->
                _vadProbability.value = prob
            }
        }

        // 감지 이벤트 → DB 저장 + 암호화 오디오 저장 + 알림
        serviceScope.launch(Dispatchers.IO) {
            newPipeline.detectionEvents.collect { aggregated ->
                // callSessionId를 즉시 캡처 (stopCapture()에 의한 null 방지)
                val currentCallSessionId = _callSession.value?.sessionId

                val audioSegment = newPipeline.ringBuffer.getLast(80_000)
                val audioFilePath = if (audioSegment.isNotEmpty()) {
                    try { encryptedStorage.saveSegment(audioSegment) } catch (_: Exception) { null }
                } else null

                val entity = DetectionEntity(
                    timestamp = System.currentTimeMillis(),
                    fakeScore = aggregated.averageFakeScore,
                    realScore = 1f - aggregated.averageFakeScore,
                    confidence = aggregated.latestResult.confidence,
                    durationMs = aggregated.latestResult.latencyMs,
                    audioFilePath = audioFilePath,
                    inferenceMode = inferenceEngine?.getModelInfo()?.type ?: "on_device",
                    threatLevel = aggregated.threatLevel.name,
                    latencyMs = aggregated.latestResult.latencyMs,
                    callSessionId = currentCallSessionId,
                )
                try { detectionDao.insert(entity) } catch (_: Exception) { }

                notificationHelper.showDetectionAlert(aggregated)
            }
        }

        // AudioRecord 설정
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(BUFFER_SIZE_SAMPLES * 2)

        // 통화 중: VOICE_COMMUNICATION (AEC 적용), 일반: MIC
        val audioSource = if (callSession != null) {
            MediaRecorder.AudioSource.VOICE_COMMUNICATION
        } else {
            MediaRecorder.AudioSource.MIC
        }

        audioRecord = AudioRecord(
            audioSource,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        )

        // AudioRecord 초기화 검증
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            stopCapture()
            return
        }

        isRecording = true
        _isMonitoring.value = true
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
        _isMonitoring.value = false
        _callSession.value = null
        runCatching { audioRecord?.stop() }
        runCatching { audioRecord?.release() }
        audioRecord = null
        vadEngine?.close()
        inferenceEngine?.close()
        pipeline = null
        _latestResult.value = null
        _vadProbability.value = 0f
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
