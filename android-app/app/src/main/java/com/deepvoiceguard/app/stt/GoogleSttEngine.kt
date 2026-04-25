package com.deepvoiceguard.app.stt

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Google SpeechRecognizer 기반 STT 엔진.
 *
 * Capability-gated: 기기에서 사용 불가 또는 동의 미획득 시 동작하지 않음.
 * 연속 인식: 인식 종료/에러 시 exponential backoff로 자동 재시작 (최대 3회).
 * 메인 스레드 필수: SpeechRecognizer는 메인 스레드에서만 생성/호출 가능.
 *
 * Authoritative shutdown: [stop]/[destroy] 호출 후에는 콜백이나 pending handler가
 * 절대 재시작하지 않도록 [isRunning] 플래그로 모든 재진입 경로를 차단한다.
 */
class GoogleSttEngine(
    private val context: Context,
    private val scope: CoroutineScope,
    private val consentProvider: () -> Boolean,
    private val capabilityChecker: SttCapabilityChecker = SttCapabilityChecker(context),
    private val mainHandler: Handler = Handler(Looper.getMainLooper()),
    private val recognizerFactory: () -> SpeechRecognizer? = {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // [DEMO ONLY] on-device 우선, 미지원 시 네트워크 STT fallback. 시연용 — 프로덕션
            // 배포 전 revert 필수. SttCapabilityChecker.isOfflineRecognitionSupported 와 짝.
            val onDeviceOk = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                runCatching { SpeechRecognizer.isOnDeviceRecognitionAvailable(context) }.getOrDefault(false)
            } else {
                true  // API 31-32: createOnDeviceSpeechRecognizer 존재만 가정
            }
            if (onDeviceOk) {
                SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
            } else {
                Log.w(TAG, "[DEMO ONLY] on-device STT 미지원 — 네트워크 SpeechRecognizer fallback")
                SpeechRecognizer.createSpeechRecognizer(context)
            }
        } else {
            null
        }
    },
    private val isMainThread: () -> Boolean = { Looper.myLooper() == Looper.getMainLooper() },
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val uptimeMillisProvider: () -> Long = { android.os.SystemClock.uptimeMillis() },
) : SttEngine {

    companion object {
        private const val TAG = "GoogleSttEngine"
        private const val MAX_RESTART_COUNT = 3
        private const val INITIAL_BACKOFF_MS = 500L
        private val RESTART_TOKEN = Any()
    }

    private val transcriptionBuffer = TranscriptionBuffer()

    private var recognizer: SpeechRecognizer? = null
    private var restartCount = 0
    private var currentBackoffMs = INITIAL_BACKOFF_MS

    // 재진입/재시작을 권위적으로 차단하는 게이트.
    private val isRunning = AtomicBoolean(false)
    private val isDestroyed = AtomicBoolean(false)

    /** In-flight onResults worker 카운터 — drain용. */
    private val pendingResultJobs = AtomicInteger(0)

    /**
     * **H140**: reset generation token. resetTranscriptionWindow가 increment하며,
     * onResults worker는 진입 시 myResetGen을 캡처하고 append/listener 호출 직전
     * 일치 여부를 확인한다. 불일치(reset 발생) 시 stale 결과를 폐기하여 main looper에
     * 큐된 callback이 reset 후에 실행되어도 transcript window를 오염시키지 않는다.
     */
    private val resetGeneration = AtomicLong(0L)

    /**
     * **H143**: 진행 중인 recognizer session id. startRecognition마다 increment하고
     * 새 listener instance에 closure-capture된다. Stale session callback (큐된 onResults 등)은
     * 자신의 mySessionId와 currentSessionId.get() 비교로 즉시 폐기된다.
     */
    private val sessionGenerator = AtomicLong(0L)
    @Volatile private var currentSessionId: Long = 0L

    /** Final transcription listener — publish worker 안에서 직접 호출 (StateFlow 우회). */
    @Volatile private var onFinalTranscription: (suspend (String) -> Unit)? = null

    override fun setOnFinalTranscriptionListener(listener: (suspend (String) -> Unit)?) {
        onFinalTranscription = listener
    }

    // H94: offline invariant를 제공할 수 없는 기기에서는 STT를 UNAVAILABLE로 고정.
    private val _status = MutableStateFlow(
        if (capabilityChecker.isOfflineRecognitionSupported()) SttStatus.READY
        else SttStatus.UNAVAILABLE
    )
    override val status: StateFlow<SttStatus> = _status.asStateFlow()

    private val _transcription = MutableStateFlow("")
    override val transcription: StateFlow<String> = _transcription.asStateFlow()

    private val _latestEvent = MutableStateFlow<TranscriptionEvent?>(null)
    override val latestEvent: StateFlow<TranscriptionEvent?> = _latestEvent.asStateFlow()

    // H94: STT 가용성은 verified offline support에 한한다.
    override fun isAvailable(): Boolean = capabilityChecker.isOfflineRecognitionSupported()

    override suspend fun awaitResultJobsIdle() {
        while (pendingResultJobs.get() > 0) {
            kotlinx.coroutines.delay(10L)
        }
    }

    override fun start() {
        if (isDestroyed.get()) {
            Log.w(TAG, "start() ignored — engine destroyed")
            return
        }
        if (!isAvailable()) {
            _status.value = SttStatus.UNAVAILABLE
            return
        }
        if (!consentProvider()) {
            _status.value = SttStatus.CONSENT_NEEDED
            return
        }
        if (!isRunning.compareAndSet(false, true)) {
            Log.d(TAG, "start() ignored — already running")
            return
        }
        // **H146**: isRunning=true 와 main thread post 사이 gap에서 옛 callback이 통과 못 하도록
        // sessionId를 즉시 invalidate. main thread의 startRecognition()이 새 sessionId 발급.
        currentSessionId = sessionGenerator.incrementAndGet()
        restartCount = 0
        currentBackoffMs = INITIAL_BACKOFF_MS
        runOnMain { startRecognition() }
    }

    /**
     * 메인 스레드에서 호출되었으면 직접 실행, 아니면 mainHandler.post 후 await.
     * suspend로 main-thread cleanup이 끝날 때까지 caller를 block하여
     * off-main caller도 return 후 동기적으로 처리된 상태를 본다.
     */
    private suspend fun runOnMainAwait(block: () -> Unit) {
        if (isMainThread()) {
            block()
            return
        }
        val done = kotlinx.coroutines.CompletableDeferred<Unit>()
        mainHandler.post {
            try {
                block()
                done.complete(Unit)
            } catch (t: Throwable) {
                // MEDIUM 9: main-thread block 예외를 caller로 전파한다.
                done.completeExceptionally(t)
            }
        }
        done.await()
    }

    /** 비동기 fire-and-forget (start의 startRecognition 같은 곳에서 사용). */
    private fun runOnMain(block: () -> Unit) {
        if (isMainThread()) {
            block()
        } else {
            mainHandler.post(block)
        }
    }

    override suspend fun stop() {
        // 1. 즉시 플래그 차단 — 후속 콜백/pending runnable이 재시작을 막음.
        if (!isRunning.compareAndSet(true, false)) return
        // **H146**: stop 진입 시점에 sessionId invalidate — 큐된 callback이 통과 못 하도록.
        currentSessionId = sessionGenerator.incrementAndGet()
        // 2. main thread cleanup을 await — off-main caller도 동기 완료 보장.
        runOnMainAwait {
            mainHandler.removeCallbacksAndMessages(RESTART_TOKEN)
            recognizer?.cancel()
            _status.value = SttStatus.READY
            Log.d(TAG, "stop() — shutdown authoritative")
        }
    }

    override suspend fun destroy() {
        isRunning.set(false)
        isDestroyed.set(true)
        // **H146**: destroy 진입 시점에 sessionId invalidate.
        currentSessionId = sessionGenerator.incrementAndGet()
        // **Main thread cleanup await** — off-main caller(serviceScope.launch on Default)도
        // recognizer가 실제 destroy된 후에 다음 step 진행. 메인 caller는 데드락 방지를 위해
        // runOnMainAwait가 직접 실행 분기.
        runOnMainAwait {
            mainHandler.removeCallbacksAndMessages(RESTART_TOKEN)
            recognizer?.cancel()
            recognizer?.destroy()
            recognizer = null
            _status.value = SttStatus.UNAVAILABLE
        }
        _transcription.value = ""
        _latestEvent.value = null
        scope.launch(workerDispatcher) {
            transcriptionBuffer.clear()
        }
    }

    /** 재시작 게이트: stop/destroy 이후에는 모든 경로가 이 체크에서 중단된다. */
    private fun canRestart(): Boolean {
        if (!isRunning.get() || isDestroyed.get()) return false
        if (!consentProvider()) {
            _status.value = SttStatus.CONSENT_NEEDED
            isRunning.set(false)
            return false
        }
        return true
    }

    private fun startRecognition() {
        if (!canRestart()) {
            Log.d(TAG, "startRecognition() blocked by shutdown/consent gate")
            return
        }
        // **H147**: createOnDeviceSpeechRecognizer / setRecognitionListener / startListening 호출은
        // OEM 구현 따라 synchronous throw 가능. 모든 경로 try/catch로 graceful degrade.
        try {
            if (recognizer == null) {
                recognizer = recognizerFactory()
                if (recognizer == null) {
                    Log.w(TAG, "createOnDeviceSpeechRecognizer unavailable — STT disabled")
                    _status.value = SttStatus.UNAVAILABLE
                    isRunning.set(false)
                    return
                }
            }
            // **H143**: 새 session id 발급 + 새 listener instance.
            val newSessionId = sessionGenerator.incrementAndGet()
            currentSessionId = newSessionId
            recognizer?.setRecognitionListener(createSessionListener(newSessionId))
            val intent = capabilityChecker.createRecognitionIntent()
            recognizer?.startListening(intent)
            _status.value = SttStatus.LISTENING
            Log.d(TAG, "Recognition started (session=$newSessionId restart=$restartCount)")
        } catch (t: Throwable) {
            Log.w(TAG, "startRecognition() platform error", t)
            // 안전한 down state — service crash 방지.
            _status.value = SttStatus.ERROR
            isRunning.set(false)
            runCatching { recognizer?.cancel() }
            runCatching { recognizer?.destroy() }
            recognizer = null
        }
    }

    private fun scheduleRestart() {
        if (!canRestart()) return
        if (restartCount >= MAX_RESTART_COUNT) {
            Log.w(TAG, "Max restart count reached, stopping STT")
            _status.value = SttStatus.ERROR
            isRunning.set(false)
            return
        }
        _status.value = SttStatus.PAUSED
        restartCount++
        val delay = currentBackoffMs
        currentBackoffMs = (currentBackoffMs * 2).coerceAtMost(4000L)
        Log.d(TAG, "Scheduling restart in ${delay}ms (attempt $restartCount)")
        val runnable = Runnable { startRecognition() }
        mainHandler.postAtTime(
            runnable,
            RESTART_TOKEN,
            uptimeMillisProvider() + delay,
        )
    }

    /** 게이트 통과 시에만 즉시 재시작 요청. */
    private fun postImmediateRestart() {
        if (!canRestart()) return
        val runnable = Runnable { startRecognition() }
        mainHandler.postAtTime(
            runnable,
            RESTART_TOKEN,
            uptimeMillisProvider(),
        )
    }

    private fun resetBackoff() {
        restartCount = 0
        currentBackoffMs = INITIAL_BACKOFF_MS
    }

    /**
     * **H143**: per-session listener factory — sessionId를 closure로 capture하여
     * stale recognizer session의 callback(reset/restart 후 큐된 message)을 폐기.
     */
    private fun createSessionListener(mySessionId: Long): RecognitionListener = object : RecognitionListener {
        private fun stale(): Boolean {
            if (mySessionId != currentSessionId) {
                Log.d(TAG, "stale session callback (my=$mySessionId vs cur=$currentSessionId) — drop")
                return true
            }
            return false
        }

        override fun onReadyForSpeech(params: Bundle?) {
            if (stale() || !isRunning.get()) return
            _status.value = SttStatus.LISTENING
        }

        override fun onBeginningOfSpeech() {}

        override fun onRmsChanged(rmsdB: Float) {}

        override fun onBufferReceived(buffer: ByteArray?) {}

        override fun onEndOfSpeech() {
            // 인식 종료 — 연속 모드를 위해 재시작 (게이트 통과 시에만)
        }

        override fun onError(error: Int) {
            if (stale()) return
            if (!isRunning.get()) {
                Log.d(TAG, "onError($error) suppressed — engine stopped")
                return
            }
            val errorName = when (error) {
                SpeechRecognizer.ERROR_NO_MATCH -> "NO_MATCH"
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "SPEECH_TIMEOUT"
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "RECOGNIZER_BUSY"
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "INSUFFICIENT_PERMISSIONS"
                SpeechRecognizer.ERROR_NETWORK -> "NETWORK"
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "NETWORK_TIMEOUT"
                SpeechRecognizer.ERROR_SERVER -> "SERVER"
                SpeechRecognizer.ERROR_CLIENT -> "CLIENT"
                SpeechRecognizer.ERROR_AUDIO -> "AUDIO"
                else -> "UNKNOWN($error)"
            }
            Log.d(TAG, "Recognition error: $errorName")

            // **H145**: restart scheduling 직전 session ownership 재검 — stale() 통과 후
            // reset이 발생했을 수 있다. 옛 session callback이 새 session을 invalidate하지 않도록.
            fun safeRestart(immediate: Boolean) {
                if (mySessionId != currentSessionId) {
                    Log.d(TAG, "restart suppressed — session=$mySessionId now stale ($currentSessionId)")
                    return
                }
                if (immediate) postImmediateRestart() else scheduleRestart()
            }
            when (error) {
                SpeechRecognizer.ERROR_NO_MATCH,
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> {
                    resetBackoff()
                    safeRestart(immediate = true)
                }
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                    safeRestart(immediate = false)
                }
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
                    _status.value = SttStatus.CONSENT_NEEDED
                    isRunning.set(false)
                }
                else -> {
                    safeRestart(immediate = false)
                }
            }
        }

        override fun onResults(results: Bundle?) {
            // **H143**: session id mismatch면 진입 즉시 폐기 — 큐된 stale callback 차단.
            if (stale()) return
            val myResetGen = resetGeneration.get()
            pendingResultJobs.incrementAndGet()
            try {
                if (!isRunning.get()) {
                    Log.d(TAG, "onResults suppressed — engine stopped")
                    return
                }
                if (resetGeneration.get() != myResetGen) {
                    Log.d(TAG, "onResults discarded — reset generation bumped")
                    return
                }
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = matches?.firstOrNull()
                if (!text.isNullOrBlank()) {
                    val event = TranscriptionEvent(text = text, isFinal = true)
                    _latestEvent.value = event
                    pendingResultJobs.incrementAndGet()
                    scope.launch(workerDispatcher) {
                        try {
                            // **H140**: append 직전에 다시 generation 검증 — async hop으로 reset이
                            // 발생했을 수 있음.
                            if (resetGeneration.get() != myResetGen) {
                                Log.d(TAG, "onResults worker discarded — reset generation bumped")
                                return@launch
                            }
                            transcriptionBuffer.append(text)
                            val windowText = transcriptionBuffer.getWindowText()
                            // listener 호출 직전 한 번 더 검증.
                            if (resetGeneration.get() != myResetGen) return@launch
                            runCatching { onFinalTranscription?.invoke(windowText) }
                            // **H142**: listener는 suspend — 그 사이 reset이 발생했을 수 있다.
                            // _transcription 쓰기 직전 재검증으로 stale text 복원 차단.
                            if (resetGeneration.get() != myResetGen) return@launch
                            _transcription.value = windowText
                        } finally {
                            pendingResultJobs.decrementAndGet()
                        }
                    }
                }
                resetBackoff()
                // **H145**: restart 직전 session 재검.
                if (mySessionId == currentSessionId) {
                    postImmediateRestart()
                } else {
                    Log.d(TAG, "onResults restart suppressed — session=$mySessionId stale")
                }
            } finally {
                pendingResultJobs.decrementAndGet()
            }
        }

        override fun onPartialResults(partialResults: Bundle?) {
            if (stale() || !isRunning.get()) return
            val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val text = matches?.firstOrNull()
            if (!text.isNullOrBlank()) {
                _latestEvent.value = TranscriptionEvent(text = text, isFinal = false)
            }
        }

        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    /** 테스트용: 현재 TranscriptionBuffer 접근. */
    internal fun getBuffer(): TranscriptionBuffer = transcriptionBuffer

    /**
     * **H137/H138/H140/H141**: 누적 transcript와 sliding window buffer를 즉시 클리어.
     * (1) **resetGeneration 증가** + onResults entry/append/listener 직전 generation 검증 (H140).
     * (2) **recognizer.cancel()**: main looper에 이미 큐된 callback도 무효화 — session-level
     *     fence (H141). 이어서 isRunning이면 새 session 시작.
     * (3) **awaitResultJobsIdle drain**: best-effort, 500ms timeout 시 명시 로깅 (H138).
     */
    override suspend fun resetTranscriptionWindow() {
        resetGeneration.incrementAndGet()
        // **H144**: cancel 전에 currentSessionId를 즉시 갱신 — cancel~새 startRecognition 사이
        // window 동안 큐된 callback이 자신의 (pre-reset) mySessionId와 비교해 stale 판단되도록.
        currentSessionId = sessionGenerator.incrementAndGet()
        // H141: main thread에서 recognizer cancel — 큐된 onResults callback 무효화.
        runOnMainAwait {
            mainHandler.removeCallbacksAndMessages(RESTART_TOKEN)
            runCatching { recognizer?.cancel() }
        }
        val drained = kotlinx.coroutines.withTimeoutOrNull(500L) {
            awaitResultJobsIdle()
            true
        }
        if (drained == null) {
            Log.w(TAG, "resetTranscriptionWindow drain TIMEOUT — generation guard에 fence 의존")
        }
        transcriptionBuffer.clear()
        _transcription.value = ""
        _latestEvent.value = null
        // 정리 후 새 recognition session 재시작 (running일 때만).
        if (isRunning.get() && !isDestroyed.get()) {
            runOnMainAwait { startRecognition() }
        }
    }
}
