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
import android.util.Log
import androidx.core.app.NotificationCompat
import com.deepvoiceguard.app.R
import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.CombinedThreatAggregator
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.CombinedThreatResult
import com.deepvoiceguard.app.inference.InferenceEngine
import com.deepvoiceguard.app.inference.OnDeviceEngine
import com.deepvoiceguard.app.inference.ServerEngine
import com.deepvoiceguard.app.phishing.PhishingKeywordDetector
import com.deepvoiceguard.app.stt.GoogleSttEngine
import com.deepvoiceguard.app.stt.SttEngine
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
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import javax.inject.Inject

/**
 * Foreground Service로 마이크 오디오를 캡처하고 AudioPipeline에 공급한다.
 * STT/피싱 탐지는 설정 + 동의 게이트를 통과해야만 활성화된다.
 */
@AndroidEntryPoint
class AudioCaptureService : Service() {

    @Inject lateinit var detectionDao: DetectionDao
    @Inject lateinit var encryptedStorage: EncryptedStorage
    @Inject lateinit var settingsRepository: SettingsRepository
    @Inject lateinit var phishingDetector: PhishingKeywordDetector
    @Inject lateinit var combinedAggregator: CombinedThreatAggregator
    @Inject lateinit var narrowbandPreprocessor: com.deepvoiceguard.app.audio.NarrowbandPreprocessor
    private lateinit var notificationHelper: NotificationHelper

    companion object {
        private const val TAG = "AudioCaptureService"
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

    // 신규: STT + 통합 위협 state.
    private val _combinedResult = MutableStateFlow<CombinedThreatResult?>(null)
    val combinedResult: StateFlow<CombinedThreatResult?> = _combinedResult

    private val _transcription = MutableStateFlow("")
    val transcription: StateFlow<String> = _transcription

    private val _sttActive = MutableStateFlow(false)
    val sttActive: StateFlow<Boolean> = _sttActive

    inner class LocalBinder : Binder() {
        val service: AudioCaptureService get() = this@AudioCaptureService
    }

    private val binder = LocalBinder()

    /**
     * 서비스 전체 lifecycle scope.
     * **CoroutineExceptionHandler**: 어떤 child coroutine이 throw하더라도 서비스가
     * 죽지 않도록 swallow + 로그만 남김. Lifecycle은 stop을 통해서만 종료.
     */
    private val serviceExceptionHandler: kotlinx.coroutines.CoroutineExceptionHandler =
        kotlinx.coroutines.CoroutineExceptionHandler { _, throwable ->
            Log.e(TAG, "Uncaught coroutine exception in serviceScope", throwable)
            _isMonitoring.value = false
            val throwableMessage = throwable.message?.take(120).orEmpty().ifBlank { "알 수 없는 오류" }
            notificationHelper.showServiceError(throwableMessage)
            serviceScope.launch {
                lifecycleMutex.withLock {
                    runCatching { stopCaptureInternal() }
                        .onFailure { stopError ->
                            Log.w(TAG, "stopCaptureInternal failed after service exception", stopError)
                        }
                }
            }
        }
    private val serviceScope: CoroutineScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default + serviceExceptionHandler
    )

    /**
     * **Per-capture scope** — startCapture()마다 새로 생성, stopCapture()에서 cancel.
     * 바인딩된 서비스라 stopSelf()가 destroy를 보장하지 않으므로 collector 누수를 막기 위함.
     */
    private var captureScope: CoroutineScope? = null
    // 동일 startCapture와 stopCapture 사이의 STT attach/detach race 방지.
    private val sttAttachMutex = kotlinx.coroutines.sync.Mutex()

    /** **H152**: per-capture STT recovery 시도 카운터. 3회 초과 시 retry 중단. */
    private val sttRecoveryAttempts = java.util.concurrent.atomic.AtomicInteger(0)

    private var audioRecord: AudioRecord? = null
    private var pipeline: AudioPipeline? = null
    private var vadEngine: VadEngine? = null
    private var inferenceEngine: InferenceEngine? = null
    private var sttEngine: SttEngine? = null
    @Volatile
    private var isRecording = false

    /**
     * Live settings StateFlow — startCapture()에서 resolved snapshot으로 초기화한다.
     * 기본 [AppSettings()] seed를 사용하지 않아 cold-start race가 발생하지 않는다.
     */
    private var liveSettings: StateFlow<AppSettings>? = null

    /** 현재 attach된 STT의 전사 forwarder Job — detach/재부착 시 명시적 cancel. */
    @Volatile private var sttTranscriptionForwarderJob: kotlinx.coroutines.Job? = null

    /**
     * 현재 capture 세션의 notification identity.
     * call session이면 CallSession.sessionId, 그렇지 않으면 capture마다 고유 UUID.
     * 세션 종료 시 NotificationHelper.clearIncidentState로 정리.
     */
    @Volatile private var currentNotificationSessionId: String? = null

    /**
     * Capture epoch counter — startCapture마다 증가. late collector가 자신의 epoch와
     * 비교하여 새 capture이 시작된 경우 alert/persistence emission을 drop할 수 있다.
     */
    private val captureEpoch = java.util.concurrent.atomic.AtomicLong(0)
    private val persistenceWarningPosted = java.util.concurrent.atomic.AtomicBoolean(false)

    /**
     * 현재 capture 세션의 shutdown 시작 시각 홀더.
     * startCaptureBody마다 새 인스턴스 생성 → collector closure에서 capture.
     * stopCaptureInternal이 이 인스턴스에 시각 기록. 다음 capture가 시작되어도
     * 이전 collector는 자신의 local watermark를 참조하므로 global 리셋의 영향 없음.
     */
    @Volatile
    private var currentShutdownWatermark: java.util.concurrent.atomic.AtomicLong? = null

    /**
     * 서비스 shutdown 진행 중 플래그.
     * applyGateTransition()이 늦은 attach로 STT를 부활시키지 않도록 차단.
     */
    private val isShuttingDown = java.util.concurrent.atomic.AtomicBoolean(false)

    // pendingPersistenceJobs는 AudioPipeline 측에서 emit 시점에 incr,
    // 여기 collector가 처리 후 decr한다 (H53에서 emit-time으로 이동).

    /**
     * 전체 start/stop lifecycle을 직렬화하는 mutex.
     * 동시 START/STOP intent가 두 세션을 동시에 초기화하거나 cleanup을 race하지 않도록.
     */
    private val lifecycleMutex = kotlinx.coroutines.sync.Mutex()

    /**
     * STT 부착 gate (H95): STT 활성화 + 동의 동의만 검사한다.
     * `phishingDetectionEnabled`는 여기서 검사하지 않는다 — STT/transcription은 피싱 분석과
     * 독립된 user-facing 기능이며, Settings UI가 두 토글을 분리 제공하므로 runtime 의미가
     * UI와 일치해야 한다. phishingDetectionEnabled는 phishing detector 호출 시점에만 검사한다.
     */
    private fun sttConsentCurrentlyValid(): Boolean {
        val s = liveSettings?.value ?: return false
        return s.sttEnabled && s.sttConsentGiven
    }

    /** 피싱 분석/저장 활성 여부 — STT gate와 독립. */
    private fun phishingAnalysisEnabled(): Boolean =
        liveSettings?.value?.phishingDetectionEnabled == true

    private fun incrementPersistenceFailure(currentPipeline: AudioPipeline) {
        val failures = currentPipeline.persistenceFailures.incrementAndGet()
        if (failures >= 5 && persistenceWarningPosted.compareAndSet(false, true)) {
            notificationHelper.showServiceError("저장 공간 문제로 기록이 저장되지 않습니다")
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        notificationHelper = NotificationHelper(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "onStartCommand action=${intent?.action}")
        when (intent?.action) {
            ACTION_START -> serviceScope.launch {
                lifecycleMutex.withLock { startCaptureInternal(null) }
            }
            ACTION_STOP -> serviceScope.launch {
                lifecycleMutex.withLock { stopCaptureInternal() }
            }
            ACTION_START_CALL -> {
                val direction = intent.getStringExtra(EXTRA_CALL_DIRECTION)
                    ?.let { CallDirection.valueOf(it) } ?: CallDirection.UNKNOWN
                val phoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
                val session = CallSession(
                    direction = direction,
                    phoneNumber = phoneNumber,
                )
                serviceScope.launch {
                    lifecycleMutex.withLock { startCaptureInternal(session) }
                }
            }
            ACTION_STOP_CALL -> {
                serviceScope.launch {
                    lifecycleMutex.withLock {
                        _callSession.value = _callSession.value?.end()
                        stopCaptureInternal()
                    }
                }
            }
            null -> {
                // START_STICKY에 의한 재시작 — intent가 null이면 안전하게 종료
                stopSelf()
            }
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder = binder

    /**
     * **lifecycleMutex 보유 상태에서만 호출 가능.**
     * 진입 즉시 isRecording=true로 atomic 예약하여 동시 START intent를 차단한다.
     * 모든 init은 try/catch로 감싸 예외 시 reservation을 무조건 해제한다.
     */
    private suspend fun startCaptureInternal(callSession: CallSession? = null) {
        if (isRecording) return

        // 진입 즉시 예약 (atomic) — 동시 START가 두 번째에 도달하면 위 가드에서 reject.
        isRecording = true
        // **isShuttingDown.set(false)는 recording state 검증 성공 후로 이동 (H87)**.
        // 현재 상태 유지 → late collector가 !isShuttingDown을 잘못 통과하지 않음.

        try {
            startCaptureBody(callSession)
        } catch (e: kotlinx.coroutines.CancellationException) {
            // 정상 cancel — 그대로 전파.
            throw e
        } catch (e: Throwable) {
            // **Startup exception graceful degradation**:
            // reservation 해제 + 부분 init 정리. 예외는 swallow하여 root coroutine을
            // 통한 service crash를 방지. 후속 START는 정상 진행 가능.
            Log.e(TAG, "startCapture failed, recovering", e)
            val throwableMessage = e.message?.take(120).orEmpty().ifBlank { "알 수 없는 오류" }
            runCatching { stopCaptureInternal() }
                .onFailure { stopError ->
                    Log.w(TAG, "stopCaptureInternal failed during startup recovery", stopError)
                }
            notificationHelper.showServiceError("시작 실패: $throwableMessage")
            isRecording = false
        }
    }

    private suspend fun startCaptureBody(callSession: CallSession?) {
        persistenceWarningPosted.set(false)
        // **Call session 초기화** — collector/persistence/UI가 _callSession.value를 참조하므로
        // 모든 collector 등록 전에 반드시 set. ACTION_STOP_CALL이 이 값을 .end()로 갱신.
        _callSession.value = callSession
        // **Notification session ID** — call이면 callSession.sessionId, 아니면 capture UUID.
        currentNotificationSessionId = callSession?.sessionId
            ?: "capture-${java.util.UUID.randomUUID()}"
        // **Epoch/watermark 선언 — 실제 bump/설치는 recorder 시작 성공 후 (H83)**.
        // 실패한 start가 이전 capture의 persistence를 drop시키지 않도록, epoch 증가는
        // audioRecord?.startRecording() 성공 시점까지 지연.
        var myEpoch: Long = captureEpoch.get()  // placeholder, 실제 bump는 아래에서.
        val myShutdownWatermark = java.util.concurrent.atomic.AtomicLong(0L)

        val notifText = if (callSession != null) {
            val dir = if (callSession.direction == CallDirection.INCOMING) "수신" else "발신"
            "통화 모니터링 중 ($dir)"
        } else "Monitoring..."
        startForeground(NOTIFICATION_ID, createNotification(notifText))

        // **Per-capture scope**: 이 캡처 세션 동안만 살아있는 child scope.
        // stopCapture()에서 cancel하여 collector 누수 방지.
        // ExceptionHandler를 명시 포함 — capture 안의 모든 collector/inference 예외가
        // service crash로 비화되지 않고 swallow + 로그.
        captureScope?.cancel()
        val capScope = CoroutineScope(
            SupervisorJob(parent = serviceScope.coroutineContext[kotlinx.coroutines.Job])
                + Dispatchers.Default
                + serviceExceptionHandler
        )
        captureScope = capScope

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

        // liveSettings를 resolved snapshot으로 초기화 — 기본 AppSettings() seed race 제거.
        // 이후 모든 sttConsentCurrentlyValid() 호출은 실제 DataStore 값을 본다.
        val resolvedLive = settingsRepository.settings
            .stateIn(capScope, SharingStarted.Eagerly, settings)
        liveSettings = resolvedLive

        // **초기 STT 시작 미루기** — snapshot이 아니라 audioRecord 시작 이후
        // applyGateTransition에서 latest gate를 평가한 뒤에만 attach.
        // 여기서는 STT를 만들지 않는다 (모든 attach는 reconcile 경로로 통일).
        sttEngine = null
        _sttActive.value = false

        // 피싱 탐지는 STT가 활성화된 경우에만 의미 있음.
        // 파이프라인은 일단 STT 없이 생성. attach는 reconcile 단계에서 발생.
        val newPipeline = AudioPipeline(
            vadEngine = vadEngine!!,
            inferenceEngine = inferenceEngine!!,
            scope = capScope,
            dangerThreshold = settings.threshold,
            sttEngine = null,
            phishingDetector = null,
            combinedAggregator = combinedAggregator,
            // **v4+ 모델은 raw PCM 입력으로 학습** — narrowband 전처리 적용 시 감지력 56%→7%로 하락.
            // training/inference mismatch 확정. 항상 null 고정 (코드 정리 시 파이프라인에서 완전 제거 예정).
            narrowbandPreprocessor = null,
            // [DIAG] 첫 5개 세그먼트 PCM 덤프 디렉터리 — /sdcard/Android/data/<pkg>/files/segments_debug/
            debugDumpDir = java.io.File(getExternalFilesDir(null), "segments_debug"),
        )
        pipeline = newPipeline

        // 파이프라인 상태를 서비스 레벨 StateFlow로 포워딩.
        // **H128**: 모든 forwarder는 capture-epoch/isShuttingDown 가드를 *쓰기 전*에 적용 —
        // stale capture의 late emission이 service-level UI state를 덮어쓰는 것 차단.
        capScope.launch {
            newPipeline.latestResult.collect { result ->
                if (captureEpoch.get() != myEpoch || isShuttingDown.get()) return@collect
                _latestResult.value = result
                _stats.value = newPipeline.getStats()
            }
        }
        capScope.launch {
            newPipeline.vadProbability.collect { prob ->
                if (captureEpoch.get() != myEpoch || isShuttingDown.get()) return@collect
                _vadProbability.value = prob
            }
        }
        // **H157/H159**: pipeline의 phishing expiry signal을 받아 NotificationHelper 정리.
        // **H159**: 단, expiry 후 combined level이 실제 SAFE일 때만 incident 통째 reset. 그렇지
        // 않으면 (예: CRITICAL→DANGER 강등) 기존 incident 연장이므로 NotificationHelper 안 건드림 —
        // monotonic rank로 follow-up DANGER alert가 자연 suppress된다.
        newPipeline.setOnPhishingExpiredListener {
            if (captureEpoch.get() != myEpoch || isShuttingDown.get()) return@setOnPhishingExpiredListener
            val postLevel = newPipeline.combinedResult.value?.combinedThreatLevel
            if (postLevel == CombinedThreatLevel.SAFE) {
                val sid = _callSession.value?.sessionId ?: currentNotificationSessionId
                notificationHelper.onIncidentSafe(sid)
            }
        }
        capScope.launch {
            var prevLevel: CombinedThreatLevel = CombinedThreatLevel.SAFE
            newPipeline.combinedResult.collect { combined ->
                if (captureEpoch.get() != myEpoch || isShuttingDown.get()) return@collect
                _combinedResult.value = combined
                val level = combined?.combinedThreatLevel ?: CombinedThreatLevel.SAFE
                val sid = _callSession.value?.sessionId ?: currentNotificationSessionId
                if (prevLevel != CombinedThreatLevel.SAFE && level == CombinedThreatLevel.SAFE) {
                    notificationHelper.onIncidentSafe(sid)
                } else if (prevLevel.ordinal > level.ordinal &&
                    level != CombinedThreatLevel.SAFE
                ) {
                    // **H163/H165**: severity downgrade — stale 트레이 알림 cancel +
                    // replacement alert post (현재 elevated state 표시).
                    notificationHelper.onIncidentDowngrade(sid, combined)
                }
                prevLevel = level
            }
        }

        // 초기 STT 전사 forwarder는 applyGateTransition에서 attach 시점에 launch.

        // [기존] 딥보이스 단독 감지 이벤트 → DB 기본 필드 저장.
        // **AudioPipeline의 single-stream routing**으로 인해 detectionEvents는
        // STT가 비활성일 때만 emit되므로 별도 가드가 필요 없다. 중복 처리 방지 끝.
        // **Immutable session/consent snapshots** — stopCaptureInternal 타임아웃 이후 late
        // collector가 새 capture의 service-global state를 읽지 못하도록 capture-time에 동결.
        val capturedSessionId = currentNotificationSessionId
        val capturedCallSessionId = _callSession.value?.sessionId
        // H95: STT 동의 snapshot에서 phishingDetectionEnabled 분리 — phishing은 collector에서 판단.
        val capturedSttConsent = settings.sttEnabled && settings.sttConsentGiven
        val capturedTranscriptStorage = settings.transcriptStorageEnabled
        val capturedInferenceMode = inferenceEngine?.getModelInfo()?.type ?: "on_device"
        var lastDetectionSeqProcessed = 0L
        capScope.launch(Dispatchers.IO) {
            newPipeline.detectionEvents.collect { envelope ->
                val aggregated = envelope.result
                kotlinx.coroutines.withContext(kotlinx.coroutines.NonCancellable) {
                try {
                if (captureEpoch.get() != myEpoch) return@withContext
                // **H183**: sequence-based stale check — older or duplicate envelope drop.
                if (envelope.seq <= lastDetectionSeqProcessed) return@withContext
                lastDetectionSeqProcessed = envelope.seq
                val currentCallSessionId = capturedCallSessionId
                // **H184**: emit 시점에 capture된 audio 사용 (envelope.audio).
                val audioSegment = envelope.audio
                // dead code removal: H84 이후 .pending path가 곧 final path다.
                val pendingPath = if (audioSegment.isNotEmpty()) {
                    try {
                        encryptedStorage.saveSegment(audioSegment)
                    } catch (e: Exception) {
                        Log.w(TAG, "encryptedStorage.saveSegment failed", e)
                        incrementPersistenceFailure(newPipeline)
                        null
                    }
                } else null
                val finalPath = pendingPath
                val entity = DetectionEntity(
                    timestamp = System.currentTimeMillis(),
                    fakeScore = aggregated.averageFakeScore,
                    realScore = 1f - aggregated.averageFakeScore,
                    confidence = aggregated.latestResult.confidence,
                    durationMs = aggregated.latestResult.latencyMs,
                    audioFilePath = finalPath,
                    inferenceMode = capturedInferenceMode,
                    threatLevel = aggregated.threatLevel.name,
                    latencyMs = aggregated.latestResult.latencyMs,
                    callSessionId = currentCallSessionId,
                    combinedThreatLevel = aggregated.threatLevel.name,
                    sttAvailable = false,
                )
                try {
                    detectionDao.insert(entity)
                } catch (e: Exception) {
                    Log.w(TAG, "detectionDao.insert (deepfake) failed", e)
                    incrementPersistenceFailure(newPipeline)
                    // Insert 실패 시 committed final file 삭제 (orphan 방지).
                    finalPath?.let { path ->
                        val deleted = runCatching { encryptedStorage.deleteSegment(path) }
                            .getOrDefault(false)
                        if (!deleted) {
                            Log.w(TAG, "orphan encrypted file not deleted: $path")
                            incrementPersistenceFailure(newPipeline)
                        }
                    }
                }
                // **Epoch + shutdown guard (M15/H73)**: 새 capture 시작 혹은 shutdown 중이면 drop.
                if (captureEpoch.get() == myEpoch && !isShuttingDown.get()) {
                    notificationHelper.showDetectionAlert(
                        aggregated,
                        currentCallSessionId ?: capturedSessionId,
                    )
                }
                } finally { newPipeline.pendingPersistenceJobs.decrementAndGet() }
                } // NonCancellable end
            }
        }

        // [신규] 통합 위협 이벤트 → DB + 통합 알림 (피싱 활성 경로).
        capScope.launch(Dispatchers.IO) {
            newPipeline.combinedEvents.collect { combined ->
                // **Counter는 emit 시점에 pipeline에서 incr됨** (H53). 여기서는 처리 후 decr만.
                kotlinx.coroutines.withContext(kotlinx.coroutines.NonCancellable) {
                try {
                // **Epoch guard (H81)**: 새 capture 시작됐으면 drop. 다음 capture의 live consent가
                // old capture의 이벤트에 잘못 적용되는 것 차단.
                if (captureEpoch.get() != myEpoch) return@withContext
                // **H167/H171/H173**: stale event 판별 — DB insert 전.
                // (a) generation 비교: SAFE 전환 후 incidentGeneration advance → drop.
                // (b) level mismatch: live state로 sanitize. 단순 drop은 fresh event 손실 위험.
                val pipelineGen = newPipeline.incidentGenerationValue()
                if (combined.emitGeneration < pipelineGen) {
                    return@withContext
                }
                // **H185+**: pipeline이 _combinedResult.value를 audioSnapshot과 함께 원자적으로
                // 갱신하므로, 단일 read로 live metadata-audio 쌍을 얻는다.
                val liveSnapshot = newPipeline.combinedResult.value
                val liveLevel = liveSnapshot?.combinedThreatLevel ?: CombinedThreatLevel.SAFE
                if (liveLevel == CombinedThreatLevel.SAFE) {
                    return@withContext
                }
                // **H186 (adversarial #1 fix)**: level mismatch 처리.
                // - UPGRADE (emitted level < live level): 더 최신 replacement가 오므로 A를 drop.
                // - DOWNGRADE (emitted level > live level): phishing detach / STT detach /
                //   phishing expiry 경로가 _combinedResult만 내려가고 replacement event는 보내지
                //   않으므로, 이 elevated event는 history에 남기되 **알림은 억제**한다.
                //   (NotificationHelper가 onIncidentDowngrade로 이미 낮은 레벨을 반영했는데
                //   뒤늦게 CRITICAL 알림을 띄우면 tray가 재-에스컬레이션되어 UX가 꼬인다.)
                val isStaleDowngrade: Boolean
                @Suppress("NAME_SHADOWING")
                val combined = when {
                    combined.combinedThreatLevel.ordinal < liveLevel.ordinal -> {
                        // UPGRADE: drop.
                        return@withContext
                    }
                    combined.combinedThreatLevel.ordinal > liveLevel.ordinal -> {
                        // DOWNGRADE: history에 남기되 모든 alert 경로(primary + ensureIncidentAlerted)
                        // 를 건너뛴다. suppressAlert flag는 기존 H122 의미(중복 방지)로 남겨두고,
                        // 이 특수 케이스는 별도 local flag로만 분기.
                        isStaleDowngrade = true
                        combined
                    }
                    else -> {
                        isStaleDowngrade = false
                        combined
                    }
                }
                // **Three-layer consent gate**:
                //  (1) capturedSttConsent — capture 시점 snapshot (cross-capture isolation, H72)
                //  (2) sttConsentCurrentlyValid() — write 시점 live check (mid-capture revocation, H74)
                //  (3) emit-time shutdown check (H77/H79): per-capture watermark로 late
                //      collector가 새 capture의 global 리셋 영향을 받지 않음.
                // **H80**: capturedSttConsent 제거 — mid-session STT opt-in 차단 방지.
                // cross-capture isolation은 epoch와 callSession snapshot으로 충분.
                val shutdownAt = myShutdownWatermark.get()
                val emittedAfterShutdown = shutdownAt > 0 && combined.timestampMs >= shutdownAt
                // H95: STT data gate (transcript) = STT 활성. phishing 필드는 별도 gate.
                val sttDataAllowed = combined.sttAvailable &&
                    sttConsentCurrentlyValid() &&
                    !emittedAfterShutdown
                // H95: 피싱 score/키워드/combined escalation 저장은 phishingDetectionEnabled에도 연동.
                val phishingAllowed = sttDataAllowed && phishingAnalysisEnabled()

                // **H71: immutable snapshot 사용** — _callSession.value 직접 읽지 않음.
                val currentCallSessionId = capturedCallSessionId
                // **H185**: 이벤트 emit 시점에 캡처된 audio snapshot을 사용 — ringBuffer 재읽기는
                // STT/피싱 분석 지연 동안 buffer가 전진해 증거 불일치를 유발. snapshot이 비어 있거나
                // sanitize 경로로 교체됐을 때만 fallback으로 ringBuffer 현재 내용을 사용한다.
                val audioSegment = combined.audioSnapshot.takeIf { it.isNotEmpty() }
                    ?: newPipeline.ringBuffer.getLast(80_000)
                // dead code removal: H84 이후 .pending path가 곧 final path다.
                val pendingPath = if (audioSegment.isNotEmpty()) {
                    try {
                        encryptedStorage.saveSegment(audioSegment)
                    } catch (e: Exception) {
                        Log.w(TAG, "encryptedStorage.saveSegment failed", e)
                        incrementPersistenceFailure(newPipeline)
                        null
                    }
                } else null
                val deepfakeThreat = combined.deepfakeThreatLevel
                val fakeScore = combined.deepfakeResult?.averageFakeScore ?: 0f
                val latency = combined.deepfakeResult?.latestResult?.latencyMs ?: 0L

                // 전사/키워드는 sttDataAllowed가 true일 때만 저장.
                val transcriptToStore = if (
                    sttDataAllowed && capturedTranscriptStorage &&
                        liveSettings?.value?.transcriptStorageEnabled == true
                ) {
                    combined.transcription.take(2000)
                } else null

                // H95: keywords/phishingScore는 phishingAllowed gate에 종속.
                val keywordsToStore = if (phishingAllowed) {
                    combined.matchedKeywords
                        .joinToString(",") { it.keyword }
                        .take(500)
                        .ifBlank { null }
                } else null

                // 동의 철회 or 피싱 분석 OFF 시 점수 마스킹.
                val phishingScoreToStore = if (phishingAllowed) combined.phishingScore else 0f
                val combinedLevelToStore = if (phishingAllowed) {
                    combined.combinedThreatLevel.name
                } else {
                    // 피싱 분석이 차단되면 통합 레벨도 딥보이스 단독 결과로 강등.
                    deepfakeThreat.name
                }

                val finalPath = pendingPath
                val entity = DetectionEntity(
                    timestamp = System.currentTimeMillis(),
                    fakeScore = fakeScore,
                    realScore = 1f - fakeScore,
                    confidence = combined.deepfakeResult?.latestResult?.confidence ?: 0f,
                    durationMs = latency,
                    audioFilePath = finalPath,
                    inferenceMode = capturedInferenceMode,
                    threatLevel = deepfakeThreat.name,
                    latencyMs = latency,
                    callSessionId = currentCallSessionId,
                    phishingScore = phishingScoreToStore,
                    phishingKeywords = keywordsToStore,
                    transcription = transcriptToStore,
                    combinedThreatLevel = combinedLevelToStore,
                    sttAvailable = sttDataAllowed,
                )
                try {
                    detectionDao.insert(entity)
                } catch (e: Exception) {
                    Log.w(TAG, "detectionDao.insert (combined) failed", e)
                    incrementPersistenceFailure(newPipeline)
                    // Orphan cleanup: insert 실패 시 committed final file 삭제.
                    finalPath?.let { path ->
                        val deleted = runCatching { encryptedStorage.deleteSegment(path) }
                            .getOrDefault(false)
                        if (!deleted) {
                            Log.w(TAG, "orphan encrypted file not deleted: $path")
                            incrementPersistenceFailure(newPipeline)
                        }
                    }
                }

                // 알림: STT-파생 텍스트 + 피싱 증거가 들어가지 않도록 sanitized result로 발송.
                // **H97**: alert 경로는 phishingAllowed를 기준으로도 downgrade — 피싱 OFF인데
                // phishing/CRITICAL 알림이 발송되는 false-positive 차단.
                val alertResult = if (phishingAllowed) {
                    combined
                } else if (sttDataAllowed) {
                    // STT는 동작하지만 phishing 분석은 OFF — 피싱 근거만 제거, 전사는 유지 불필요.
                    combined.copy(
                        matchedKeywords = emptyList(),
                        matchedPhrases = emptyList(),
                        phishingScore = 0f,
                        combinedThreatLevel = CombinedThreatAggregator
                            .deepfakeOnlyLevel(deepfakeThreat),
                    )
                } else {
                    // STT 자체가 비활성 — 모든 STT-파생 필드 제거.
                    combined.copy(
                        transcription = "",
                        matchedKeywords = emptyList(),
                        matchedPhrases = emptyList(),
                        phishingScore = 0f,
                        sttAvailable = false,
                        combinedThreatLevel = CombinedThreatAggregator
                            .deepfakeOnlyLevel(deepfakeThreat),
                    )
                }
                // **H103**: suppressAlert 플래그가 있으면 alert 우회 (persistence-only).
                // **H186**: isStaleDowngrade(DOWNGRADE 경로)는 primary alert도 건너뛴다 —
                // notification은 이미 downgrade handler가 반영했고 여기서 재-에스컬레이션 금지.
                if (!isStaleDowngrade && !combined.suppressAlert &&
                    alertResult.combinedThreatLevel != CombinedThreatLevel.SAFE
                ) {
                    if (captureEpoch.get() == myEpoch && !isShuttingDown.get()) {
                        notificationHelper.showCombinedAlert(
                            alertResult,
                            currentCallSessionId ?: capturedSessionId,
                        )
                    }
                } else if (isStaleDowngrade) {
                    // **DOWNGRADE stale event**: alert 경로 전면 생략.
                    // 근거: 이 분기에 도달한 시점에는 downgrade 주체(phishing detach / STT detach /
                    //   phishing expiry)가 이미 onIncidentDowngrade() 경로로 replacement 알림을
                    //   게시한 상태이거나, 그 자체가 incident 종료를 의미한다. stale event로 alert
                    //   retry를 시도하면 (1) 라우팅/필드 오염, (2) 재-에스컬레이션, (3) consent/settings
                    //   차이에 따른 오탐 등 어느 방향으로도 손해가 크다. 따라서 persistence(아래 DB
                    //   insert)만 남기고 알림은 downgrade handler의 책임으로 위임한다.
                } else if (combined.suppressAlert &&
                    alertResult.combinedThreatLevel != CombinedThreatLevel.SAFE
                ) {
                    if (captureEpoch.get() == myEpoch && !isShuttingDown.get()) {
                        notificationHelper.ensureIncidentAlerted(
                            alertResult,
                            currentCallSessionId ?: capturedSessionId,
                        )
                    }
                }
                // SAFE 분기는 별도 combinedResult StateFlow watcher에서 처리 (H124).
                // combinedEvents 는 SAFE를 emit하지 않으므로 여기에 도달하지 않음.
                } finally { newPipeline.pendingPersistenceJobs.decrementAndGet() }
                } // NonCancellable end
            }
        }

        // AudioRecord 설정
        run {
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(BUFFER_SIZE_SAMPLES * 2)

        // 통화 중: VOICE_COMMUNICATION (AEC 적용), 일반: MIC.
        // VOICE_RECOGNITION 실험 결과 오히려 RMS가 10배 감소하여 AASIST 훈련 분포 이탈. 원복.
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
        Log.i(
            TAG,
            "AudioRecord init: source=$audioSource sampleRate=$SAMPLE_RATE bufferSize=$bufferSize state=${audioRecord?.state}"
        )

        // AudioRecord 초기화 검증 — 실패 시 동기 cleanup으로 안전하게 롤백.
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord init FAILED — state != INITIALIZED")
            stopCaptureInternal()
            return
        }

        // 이미 startCaptureInternal 진입 시 isRecording=true로 예약함.
        _isMonitoring.value = true
        audioRecord?.startRecording()
        Log.i(
            TAG,
            "AudioRecord.startRecording() → recordingState=${audioRecord?.recordingState} (target=${AudioRecord.RECORDSTATE_RECORDING})"
        )
        // **Recording state 검증 (H85)** — startRecording()이 return해도 실제로 RECORDING
        // 상태가 아닐 수 있음(audio policy contention 등). RECORDSTATE_RECORDING 확인 후에만
        // epoch/watermark를 bump하여 실패한 start가 이전 epoch를 invalidate하지 않게.
        if (audioRecord?.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
            android.util.Log.w(
                "AudioCaptureService",
                "AudioRecord startRecording did not enter RECORDING state; aborting capture. Prior epoch preserved."
            )
            stopCaptureInternal()
            return
        }
        myEpoch = captureEpoch.incrementAndGet()
        currentShutdownWatermark = myShutdownWatermark
        sttRecoveryAttempts.set(0)  // H152: per-capture retry budget reset.
        // **Now** clear isShuttingDown — new capture가 known-good 확인된 이후.
        // 이전 단계에서 fail했으면 prior isShuttingDown=true 유지되어 late collector guard 유효.
        isShuttingDown.set(false)

        // 오디오 읽기 루프
        capScope.launch(Dispatchers.IO) {
            val buffer = ShortArray(BUFFER_SIZE_SAMPLES)
            while (isActive && isRecording) {
                val readSize = audioRecord?.read(buffer, 0, BUFFER_SIZE_SAMPLES) ?: -1
                if (readSize > 0) {
                    pipeline?.processAudioChunk(buffer, readSize)
                }
            }
        }
        }

        // **Live consent watcher** — pipeline + isRecording 활성화 후 시작.
        // 초기 STT 부착도 이 reconcile 단계에서 수행 (스냅샷 기반 즉시 시작 X).
        // applyGateTransition은 sttAttachMutex로 직렬화되며 isShuttingDown을 체크.
        capScope.launch {
            // 시작 시 latest gate로 바로 reconcile — 첫 emission 스킵 없이 사용.
            val latestGate = sttConsentCurrentlyValid()
            // **H130**: 호출 시점의 captureEpoch가 아니라 startCaptureBody에서 캡처한 myEpoch를
            // 명시 전달 — stale watcher가 새 capture의 epoch로 통과하지 못하도록.
            applyGateTransition(latestGate, myEpoch)
            var firstEmissionSkipped = false
            resolvedLive
                .map { it.sttEnabled && it.sttConsentGiven }
                .distinctUntilChanged()
                .collect { gateOpen ->
                    if (!firstEmissionSkipped) {
                        firstEmissionSkipped = true
                        return@collect
                    }
                    applyGateTransition(gateOpen, myEpoch)
                }
        }

        // **H98: Phishing-enabled watcher** — STT gate와 독립. 피싱 토글 OFF 전환 시
        // pipeline에서 phishingDetector만 detach하고 combinedResult를 deepfake-only로 강등.
        // ON 전환 시 새 detector를 re-attach.
        capScope.launch {
            var firstEmission = true
            resolvedLive
                .map { it.phishingDetectionEnabled }
                .distinctUntilChanged()
                .collect { enabled ->
                    if (firstEmission) {
                        // 초기 상태는 capture 시작 시의 phishingDetector로 이미 세팅됨.
                        firstEmission = false
                        return@collect
                    }
                    // Capture-epoch guard — 이 watcher가 old capture의 pipeline에 쓰는 것 방지.
                    if (captureEpoch.get() != myEpoch) return@collect
                    // **H132**: STT attach와 직렬화 — applyGateTransition이 attach 중 suspend된
                    // 동안 phishing toggle이 들어와도 attach 완료 후에 적용된다. 이로써 attach가
                    // 끝난 뒤 watcher가 실제 enabled 상태를 그대로 반영하여 OFF 후 detector
                    // 재주입되는 race를 차단.
                    sttAttachMutex.withLock {
                        if (captureEpoch.get() != myEpoch) return@withLock
                        val pipelineRef = pipeline ?: return@withLock
                        if (!enabled) {
                            runCatching { pipelineRef.setPhishingEnabled(false) }
                        } else {
                            if (sttEngine != null) {
                                runCatching {
                                    pipelineRef.setPhishingEnabled(true, phishingDetector)
                                }
                            }
                        }
                    }
                }
        }
    }

    /**
     * STT 엔진을 안전하게 drain → destroy → detach.
     * stop()으로 신규 result 차단 → onResults worker idle await → pipeline fullyIdle
     * → destroy + 서비스 ref 정리 → pipeline detach 순.
     * sttAttachMutex를 보유한 상태에서 호출해야 한다.
     */
    private suspend fun drainAndDetachStt() {
        val engineToTeardown = sttEngine ?: run {
            // 엔진이 없어도 forwarder만 정리하고 종료.
            sttTranscriptionForwarderJob?.cancel()
            sttTranscriptionForwarderJob = null
            return
        }
        // 1. stop()만 호출 — 새 partial/final result 차단.
        runCatching { engineToTeardown.stop() }
        // 2. onResults worker drain (최대 1초). timeout 시 명시 로깅.
        val resultDrain = runCatching {
            kotlinx.coroutines.withTimeoutOrNull(1000L) {
                engineToTeardown.awaitResultJobsIdle()
                true
            }
        }.getOrNull()
        if (resultDrain == null) {
            android.util.Log.w(
                "AudioCaptureService",
                "STT result jobs drain TIMEOUT — late publish may occur after detach"
            )
        }
        // 3. pipeline의 in-flight 분석 await (최대 1.5초). timeout 시 명시 로깅.
        val pipelineRef = pipeline
        val pipelineDrain = runCatching {
            kotlinx.coroutines.withTimeoutOrNull(1500L) {
                pipelineRef?.awaitFullyIdle()
                true
            }
        }.getOrNull()
        if (pipelineDrain == null) {
            android.util.Log.w(
                "AudioCaptureService",
                "Pipeline drain TIMEOUT during gate-off — final phishing/combined result may be lost"
            )
        }
        // 4. 서비스 ref 정리 + destroy.
        sttEngine = null
        sttTranscriptionForwarderJob?.cancel()
        sttTranscriptionForwarderJob = null
        runCatching { engineToTeardown.destroy() }
        // 5. pipeline detach.
        runCatching { pipelineRef?.detachSttAndPhishing() }
        // 6. UI state 정리.
        _sttActive.value = false
        _transcription.value = ""
    }

    /**
     * Gate 변경에 대한 STT attach/detach 작업을 직렬화 수행.
     * Watcher와 startup 보정에서 모두 호출 — 항상 같은 mutex로 가드.
     * Shutdown 중이면 attach를 거부하여 STT 부활을 차단.
     */
    /**
     * **H130**: default 값 제거 — 호출자가 반드시 자신의 originating capture epoch를 명시한다.
     * default = captureEpoch.get()는 stale watcher가 호출 시점의 새 epoch를 보고 통과하여
     * 잘못된 capture의 STT/pipeline을 mutate하는 race를 유발했다.
     */
    private suspend fun applyGateTransition(gateOpen: Boolean, ownerEpoch: Long) {
        sttAttachMutex.withLock {
            if (isShuttingDown.get()) return
            // **Epoch guard (H90)**: 이 watcher의 owner epoch가 현재 capture와 다르면 무시.
            // 이전 capture의 stale watcher가 새 capture의 global state를 mutate하지 않게.
            if (captureEpoch.get() != ownerEpoch) return
            val capScope = captureScope ?: return
            if (!gateOpen) {
                // **Drained teardown** — full shutdown과 동일한 순서로 진행하여
                // gate-off 도중 in-flight 분석이 손실되거나 stale state로 강등되지 않게.
                drainAndDetachStt()
            } else if (sttEngine == null && pipeline != null && isRecording) {
                val newStt = GoogleSttEngine(
                    context = this@AudioCaptureService,
                    scope = capScope,
                    consentProvider = { sttConsentCurrentlyValid() },
                )
                sttEngine = newStt
                // H99: 초기 attach에서 phishingDetectionEnabled 반영.
                val initialPhishingEnabled = phishingAnalysisEnabled()
                val initialDetector = if (initialPhishingEnabled) phishingDetector else null
                pipeline?.attachSttAndPhishing(newStt, initialDetector)
                // **H132**: attach 중 suspend 동안 phishing toggle이 변경됐을 수 있다 — sttAttachMutex
                // 보유 중이므로 watcher의 setPhishingEnabled는 직렬화되어 우리 뒤에 실행되지만,
                // attach 시작 전 변경된 상태가 attach에 반영되지 못한 케이스를 위해 한 번 더
                // 보정한다. 현재 phishingDetectionEnabled가 attach 시점 snapshot과 다르면 정합화.
                val finalPhishingEnabled = phishingAnalysisEnabled()
                if (finalPhishingEnabled != initialPhishingEnabled) {
                    val finalDetector = if (finalPhishingEnabled) phishingDetector else null
                    runCatching { pipeline?.setPhishingEnabled(finalPhishingEnabled, finalDetector) }
                }
                newStt.start()
                _sttActive.value = true
                sttTranscriptionForwarderJob?.cancel()
                sttTranscriptionForwarderJob = capScope.launch {
                    newStt.transcription.collect { text ->
                        if (sttEngine !== newStt) return@collect
                        _transcription.value = text
                    }
                }
                // **H148**: STT status를 관찰해 ERROR/UNAVAILABLE 도착 시 dead engine 정리.
                // platform/OEM throw로 startup 실패한 경우에도 다음 gate 변화나 동의 토글 시
                // 새 instance로 attach 가능하도록.
                capScope.launch {
                    var dwellJob: kotlinx.coroutines.Job? = null
                    newStt.status.collect { s ->
                        if (sttEngine !== newStt) {
                            dwellJob?.cancel()
                            return@collect
                        }
                        // **H152**: UNAVAILABLE은 terminal — capability 부재로 retry 무의미.
                        // ERROR만 retry 후보로 처리하며 capability 재확인 + bounded retry.
                        when (s) {
                            com.deepvoiceguard.app.stt.SttStatus.LISTENING -> {
                                // **H158/H160**: LISTENING 즉시 reset 대신 30s dwell time 후 reset.
                                // LISTENING→ERROR 즉시 사이클에서 budget이 매번 reset되어 무한
                                // retry 발생하는 것 차단.
                                dwellJob?.cancel()
                                dwellJob = capScope.launch {
                                    kotlinx.coroutines.delay(30_000L)
                                    if (sttEngine === newStt &&
                                        newStt.status.value == com.deepvoiceguard.app.stt.SttStatus.LISTENING
                                    ) {
                                        sttRecoveryAttempts.set(0)
                                    }
                                }
                            }
                            com.deepvoiceguard.app.stt.SttStatus.UNAVAILABLE -> {
                                dwellJob?.cancel()  // H160
                                if (captureEpoch.get() != ownerEpoch || isShuttingDown.get()) return@collect
                                sttAttachMutex.withLock {
                                    if (sttEngine === newStt) drainAndDetachStt()
                                }
                                // **H156**: UNAVAILABLE도 runtime recognizer init failure일 수 있다.
                                // 1회만 capability 재확인 후 retry — 영구 미지원이면 capability 가
                                // false 반환하여 자연 종료.
                                if (sttRecoveryAttempts.incrementAndGet() > 1) return@collect
                                val capable = runCatching {
                                    com.deepvoiceguard.app.stt.SttCapabilityChecker(
                                        this@AudioCaptureService
                                    ).isOfflineRecognitionSupported()
                                }.getOrDefault(false)
                                if (!capable) return@collect
                                kotlinx.coroutines.delay(5_000L)
                                if (captureEpoch.get() != ownerEpoch || isShuttingDown.get()) return@collect
                                if (!isRecording || !sttConsentCurrentlyValid()) return@collect
                                applyGateTransition(true, ownerEpoch)
                            }
                            // H161: PAUSED/READY/CONSENT_NEEDED 등에서도 dwell job cancel —
                            // 어떠한 LISTENING 이탈도 budget reset을 무효화해야 함.
                            com.deepvoiceguard.app.stt.SttStatus.PAUSED,
                            com.deepvoiceguard.app.stt.SttStatus.READY,
                            com.deepvoiceguard.app.stt.SttStatus.CONSENT_NEEDED -> {
                                dwellJob?.cancel()
                            }
                            com.deepvoiceguard.app.stt.SttStatus.ERROR -> {
                                dwellJob?.cancel()  // H160
                                if (captureEpoch.get() != ownerEpoch || isShuttingDown.get()) return@collect
                                sttAttachMutex.withLock {
                                    if (sttEngine === newStt) drainAndDetachStt()
                                }
                                if (sttRecoveryAttempts.incrementAndGet() > 3) {
                                    android.util.Log.w(
                                        "AudioCaptureService",
                                        "STT recovery budget exhausted — giving up for this capture",
                                    )
                                    return@collect
                                }
                                // capability 재확인.
                                val capable = runCatching {
                                    com.deepvoiceguard.app.stt.SttCapabilityChecker(
                                        this@AudioCaptureService
                                    ).isOfflineRecognitionSupported()
                                }.getOrDefault(false)
                                if (!capable) return@collect
                                kotlinx.coroutines.delay(5_000L)
                                if (captureEpoch.get() != ownerEpoch || isShuttingDown.get()) return@collect
                                if (!isRecording || !sttConsentCurrentlyValid()) return@collect
                                applyGateTransition(true, ownerEpoch)
                            }
                            else -> { /* 정상 상태 — no-op */ }
                        }
                    }
                }
            }
        }
    }

    /**
     * **Authoritative shutdown with drain.** lifecycleMutex 보유 상태에서 호출.
     *
     * 종료 순서:
     *  1) shutdown flag, isRecording flag로 신규 작업 차단
     *  2) AudioRecord 정지 — 오디오 읽기 루프가 자연 종료
     *  3) STT teardown (mutex 안에서 await)
     *  4) **Drain 윈도우 (최대 2초)** — in-flight analyzeSegments / DB insert /
     *     알림 발신이 자연 완료되기를 기다림. 마지막 detection이 손실되지 않도록.
     *     inferenceMutex가 풀리고 SharedFlow buffer가 drain될 시간을 줌.
     *  5) captureScope cancelAndJoin — 남은 watcher/collector 정리
     *  6) **그 다음에야** vadEngine/inferenceEngine.close() — OrtSession use-after-close 차단
     */
    private suspend fun stopCaptureInternal() {
        // 1. 즉시 shutdown flag + recording flag.
        isShuttingDown.set(true)
        // 현재 capture의 per-session shutdown watermark에 시각 기록. 다음 capture는
        // 별도 인스턴스를 쓰므로 late collector는 자신의 인스턴스를 계속 참조.
        currentShutdownWatermark?.set(System.currentTimeMillis())
        isRecording = false
        // **Epoch는 여기서 증가시키지 않음 (H82)** — stop만으로 same-capture persistence가
        // drop되지 않도록. epoch invalidation은 startCaptureBody에서 new capture가 실제로
        // 시작될 때만 발생. stop 이후 alert 억제는 isShuttingDown으로 처리.
        _isMonitoring.value = false
        _sttActive.value = false
        runCatching { audioRecord?.stop() }
        runCatching { audioRecord?.release() }
        audioRecord = null

        // 2. **STT drained teardown** — stop → onResults drain → pipeline idle → destroy → detach.
        // applyGateTransition(false)와 동일한 경로로 통일. STT가 없으면 즉시 return하므로
        // deepfake-only 경로에서는 별도 awaitFullyIdle이 필요 (다음 step).
        sttAttachMutex.withLock {
            drainAndDetachStt()
        }

        // 3. **Unconditional pipeline drain** — STT 없는 deepfake-only 세션도 마지막
        // analyzeSegments/DB write가 완료되도록 보장. timeout이면 명시 로깅.
        val drainResult = runCatching {
            kotlinx.coroutines.withTimeoutOrNull(1500L) {
                pipeline?.awaitFullyIdle()
                true
            }
        }.getOrNull()
        if (drainResult == null) {
            android.util.Log.w(
                "AudioCaptureService",
                "shutdown drain TIMEOUT — last detection/persistence may be lost"
            )
        }

        // 4. **Persistence drain (explicit)** — pipeline의 emit-time 카운터를 await.
        // SharedFlow buffer에 들어 있는 이벤트 + 처리 중인 collector 작업까지 모두 fence.
        val pipelineForDrain = pipeline
        val persistDrain = runCatching {
            kotlinx.coroutines.withTimeoutOrNull(2000L) {
                while ((pipelineForDrain?.pendingPersistenceJobs?.get() ?: 0) > 0) {
                    kotlinx.coroutines.delay(20L)
                }
                true
            }
        }.getOrNull()
        if (persistDrain == null) {
            android.util.Log.w(
                "AudioCaptureService",
                "Persistence drain TIMEOUT — ${pipelineForDrain?.pendingPersistenceJobs?.get()} jobs pending; final write may be incomplete"
            )
        }
        val failures = pipelineForDrain?.persistenceFailures?.get() ?: 0
        if (failures > 0) {
            android.util.Log.w(
                "AudioCaptureService",
                "Persistence failures during session: $failures (encrypted save or Room insert errors)"
            )
        }

        // 5. capture scope cancel + bounded join — 남은 watcher/collector 정리.
        // 무한 join 회피를 위해 2초 cap. timeout 시 강제 cancel + 로깅.
        val scopeJob = captureScope?.coroutineContext?.get(Job)
        captureScope = null
        val joinResult = runCatching {
            kotlinx.coroutines.withTimeoutOrNull(2000L) {
                scopeJob?.cancelAndJoin()
                true
            }
        }.getOrNull()
        if (joinResult == null) {
            android.util.Log.w(
                "AudioCaptureService",
                "capture scope join TIMEOUT — pending I/O work may be cancelled mid-write"
            )
            scopeJob?.cancel()
        }

        // 6. 이제 안전하게 engine close.
        runCatching { vadEngine?.close() }
        runCatching { inferenceEngine?.close() }
        vadEngine = null
        inferenceEngine = null
        pipeline = null

        // 7. UI state 정리.
        _latestResult.value = null
        _combinedResult.value = null
        _transcription.value = ""
        _vadProbability.value = 0f
        _callSession.value = null
        // **NotificationHelper 세션 state 정리 (M13)** — 다음 세션이 이전 쿨다운/rank에
        // 의해 영향받지 않도록.
        runCatching { notificationHelper.clearIncidentState(currentNotificationSessionId) }
        persistenceWarningPosted.set(false)
        currentNotificationSessionId = null
        liveSettings = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        // **ANR 방지**: onDestroy는 메인 스레드에서 호출됨 (Service ANR 한도 ~10초).
        // hard cap 6초 (phishing await 1.5s + collector drain 0.5s + scope cancel 2s + buffer 2s).
        // 시간 초과 시 critical resources(audioRecord, sttEngine)는 동기 정리.
        runCatching {
            runBlocking {
                kotlinx.coroutines.withTimeoutOrNull(6000L) {
                    lifecycleMutex.withLock {
                        runCatching { stopCaptureInternal() }
                    }
                } ?: run {
                    // Timeout 시 최소한의 동기 cleanup만.
                    runCatching { audioRecord?.stop() }
                    runCatching { audioRecord?.release() }
                    audioRecord = null
                    runCatching { sttEngine?.destroy() }
                    sttEngine = null
                }
            }
        }
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
