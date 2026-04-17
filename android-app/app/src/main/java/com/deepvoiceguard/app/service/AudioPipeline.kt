package com.deepvoiceguard.app.service

import android.util.Log
import com.deepvoiceguard.app.audio.RingBuffer
import com.deepvoiceguard.app.audio.SegmentExtractor
import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.CombinedThreatAggregator
import com.deepvoiceguard.app.inference.CombinedThreatResult
import com.deepvoiceguard.app.inference.DetectionAggregator
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.InferenceEngine
import com.deepvoiceguard.app.phishing.PhishingKeywordDetector
import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.stt.SttEngine
import com.deepvoiceguard.app.stt.SttStatus
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * 오디오 파이프라인: RingBuffer → VAD → SegmentExtractor → InferenceEngine → Aggregator.
 *
 * AudioCaptureService가 이 파이프라인에 오디오 프레임을 공급하고,
 * UI는 StateFlow를 구독하여 실시간 결과를 표시한다.
 */
class AudioPipeline(
    private val vadEngine: VadEngine,
    private val inferenceEngine: InferenceEngine,
    private val scope: CoroutineScope,
    dangerThreshold: Float = 0.9f,
    sttEngine: SttEngine? = null,
    phishingDetector: PhishingKeywordDetector? = null,
    private val combinedAggregator: CombinedThreatAggregator = CombinedThreatAggregator(),
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val phishingExpiryMs: Long = 60_000L,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
    // **도메인 mismatch 보정**: 라이브 마이크 입력을 AASIST 학습 분포(narrowband ~3kHz)로 맞추는
    // 전처리. null이면 원본 segment를 AASIST에 그대로 전달 (테스트/레거시 호환).
    // 주의: VAD와 ring buffer/audio snapshot은 **원본 PCM**을 그대로 쓴다 — 필터링된 신호는
    // 오직 inferenceEngine.detect 입력으로만 사용.
    private val narrowbandPreprocessor: com.deepvoiceguard.app.audio.NarrowbandPreprocessor? = null,
) {
    companion object {
        private const val TAG = "AudioPipeline"
    }

    // STT/피싱 참조는 런타임에 detach 가능하도록 private var로 보관.
    @Volatile private var sttEngine: SttEngine? = sttEngine
    @Volatile private var phishingDetector: PhishingKeywordDetector? = phishingDetector

    // 현재 attach된 STT의 transcription collector Job — detach 시 명시적으로 cancel.
    @Volatile private var sttTranscriptionJob: Job? = null

    /**
     * 현재 in-flight 중인 runPhishingAnalysis 호출 수.
     * shutdown drain은 이 카운터가 0이 될 때까지 대기.
     */
    private val pendingPhishingAnalyses = AtomicInteger(0)

    /** Phishing analysis 시퀀스 — 호출마다 증가, 마지막 적용 시퀀스를 추적. */
    private val phishingSeq = AtomicLong(0)

    /**
     * **H150**: _phishingResult가 마지막으로 갱신된 시각.
     * **H153**: 추가로 expiry job을 schedule해 60s 후 자동 clear + combinedResult downgrade.
     */
    @Volatile private var lastPhishingResultTimeMs: Long = 0L
    @Volatile private var phishingExpiryJob: kotlinx.coroutines.Job? = null

    /**
     * **H157**: phishing expiry로 incident가 강등됐음을 외부 (NotificationHelper)에 알린다.
     * AudioCaptureService가 listener 등록하여 notification 측 incident state까지 리셋.
     */
    @Volatile private var phishingExpiredListener: (() -> Unit)? = null
    fun setOnPhishingExpiredListener(listener: (() -> Unit)?) {
        phishingExpiredListener = listener
    }

    /**
     * **H129**: incident generation. SAFE 전환마다 increment되어 pre-SAFE 시작된 STT 분석이
     * SAFE 이후 결과를 적용하지 못하게 한다. runPhishingAnalysis가 시작 시 캡처한 generation과
     * 결과 적용 시점의 generation이 다르면 stale로 간주하고 폐기.
     */
    private val incidentGeneration = AtomicLong(0)

    /** **H167**: 외부에서 현재 incident generation을 비교할 수 있도록 노출. */
    fun incidentGenerationValue(): Long = incidentGeneration.get()
    private val lastAppliedPhishingSeq = AtomicLong(0)

    /**
     * STT/phishing attach generation. attach/detach마다 increment.
     * pre-detach에 시작된 분석이 post-detach (또는 다음 attach) 후 결과 적용하는 것을 차단.
     */
    private val attachGeneration = AtomicLong(0)

    // pendingTranscriptionDispatch 카운터는 H48에서 제거됨 (StateFlow conflation 회피).
    // 모든 phishing 트리거는 이제 GoogleSttEngine.pendingResultJobs로 fence된다.
    val ringBuffer = RingBuffer()
    private val segmentExtractor = SegmentExtractor(ringBuffer)
    private val aggregator = DetectionAggregator(
        dangerThreshold = dangerThreshold,
        warningThreshold = (dangerThreshold - 0.2f).coerceAtLeast(0.5f),
        cautionThreshold = (dangerThreshold - 0.3f).coerceAtLeast(0.4f),
    )

    private val _latestResult = MutableStateFlow<AggregatedResult?>(null)
    val latestResult: StateFlow<AggregatedResult?> = _latestResult

    /**
     * **H183/H184**: detection event envelope — emit 시점에 sequence와 audio snapshot을 첨부.
     * collector가 envelope.seq를 lastProcessedSeq와 비교하여 stale (older sequence) 폐기,
     * envelope.audio로 emit 시점의 audio segment를 정확히 사용.
     */
    data class DetectionEnvelope(
        val result: AggregatedResult,
        val seq: Long,
        val audio: FloatArray,
    )
    private val _detectionEvents = MutableSharedFlow<DetectionEnvelope>(extraBufferCapacity = 16)
    val detectionEvents: SharedFlow<DetectionEnvelope> = _detectionEvents
    private val detectionEmitSeq = AtomicLong(0L)

    /**
     * Pending persistence work counter — emit 시점에 incr, collector가 처리 후 decr.
     * SharedFlow buffer에 들어 있는 이벤트도 fence하여 shutdown drain이 collector
     * 완료를 정확히 await할 수 있게 한다.
     */
    val pendingPersistenceJobs = AtomicInteger(0)

    /** Persistence 실패 누적 — collector가 encrypted save/Room insert 실패 시 incr. */
    val persistenceFailures = AtomicInteger(0)

    // Branch B: STT + 피싱 탐지 결과
    private val _phishingResult = MutableStateFlow<PhishingResult?>(null)
    val phishingResult: StateFlow<PhishingResult?> = _phishingResult

    private val _combinedResult = MutableStateFlow<CombinedThreatResult?>(null)
    val combinedResult: StateFlow<CombinedThreatResult?> = _combinedResult

    private val _combinedEvents = MutableSharedFlow<CombinedThreatResult>(extraBufferCapacity = 16)
    val combinedEvents: SharedFlow<CombinedThreatResult> = _combinedEvents

    private val _vadProbability = MutableStateFlow(0f)
    val vadProbability: StateFlow<Float> = _vadProbability

    private val inferenceMutex = Mutex()
    private val combinedMutex = Mutex()
    private var segmentsAnalyzed = 0L
    private var detectionsCount = 0
    private var phishingDetectionsCount = 0

    // Dedupe 상태 — 마지막 emit된 통합 결과의 signature.
    private var lastEmittedThreatLevel: String? = null
    private var lastEmittedFakeScore: Float = -1f
    private var lastEmittedPhishingScore: Float = -1f
    private var lastEmittedTimeMs: Long = 0L
    private val emitScoreDelta: Float = 0.15f
    private val emitMinIntervalMs: Long = 10_000L

    // H105: suppressAlert fallback persistence의 별도 coalesce — history/quota 낭비 방지.
    // alert path는 10s interval이지만 persistence는 더 긴 incident-window (30s)로 게이트.
    private var lastFallbackPersistTimeMs: Long = 0L
    private val fallbackPersistIntervalMs: Long = 30_000L

    // VAD에 512 단위로 공급 시 남는 tail samples 버퍼
    private var vadRemainder = FloatArray(0)

    init {
        vadEngine.listener = object : VadEngine.Listener {
            override fun onSpeechStart(timestampMs: Long) {
                // Branch B: STT가 활성 상태이면 이미 독립적으로 동작 중
            }

            override fun onSpeechEnd(timestampMs: Long, durationMs: Long) {
                // Branch A: 음성 종료 → 세그먼트 추출 → AASIST 추론
                scope.launch(workerDispatcher) {
                    analyzeSegments(durationMs)
                }
            }
        }

        // Branch B: 초기 STT가 있으면 transcription 구독 시작.
        // 추적 가능한 Job으로 launch하여 detach 시 cancel.
        startSttTranscriptionCollector()
    }

    /**
     * 현재 sttEngine의 transcription을 구독하는 collector를 시작한다.
     * 이전 collector가 있으면 먼저 cancel하여 중복 collector를 막는다.
     *
     * **H100**: phishingDetector 유무와 무관하게 collector를 시작한다 (H101 분리 정책).
     * detector가 null이면 listener는 no-op으로 설치되며, 피싱 분석이 나중에 활성화될 때
     * [setPhishingEnabled]가 listener만 교체하여 즉시 분석이 시작되도록 한다.
     */
    private fun startSttTranscriptionCollector() {
        sttTranscriptionJob?.cancel()
        sttTranscriptionJob = null
        val engine = sttEngine ?: return
        installPhishingListener(engine)
        // _transcription StateFlow collector는 더 이상 phishing 트리거 안 함 (UI용 deprecated 경로).
        // 호환성 유지를 위해 빈 collector만 launch (Job 추적용).
        sttTranscriptionJob = scope.launch(workerDispatcher) {
            engine.transcription.collect { /* no-op — analysis는 listener에서 처리 */ }
        }
    }

    /**
     * 현재 phishingDetector 상태에 맞춰 STT engine의 final transcription listener를 설치.
     * detector==null이면 listener를 clear, detector != null이면 runPhishingAnalysis 호출 listener 설치.
     */
    private fun installPhishingListener(engine: SttEngine) {
        if (phishingDetector == null) {
            engine.setOnFinalTranscriptionListener(null)
            return
        }
        engine.setOnFinalTranscriptionListener { transcription ->
            // Listener 발화 시점에 STT 재부착이 있었으면 stale listener → drop.
            if (sttEngine !== engine) return@setOnFinalTranscriptionListener
            if (phishingDetector == null) return@setOnFinalTranscriptionListener
            if (transcription.isNotBlank()) {
                runPhishingAnalysis(transcription)
            }
        }
    }

    /**
     * AudioRecord에서 읽은 PCM short 배열을 처리한다.
     * Short → Float 변환 → RingBuffer 쓰기 → VAD 프레임 처리.
     */
    fun processAudioChunk(shorts: ShortArray, readSize: Int) {
        val floats = FloatArray(readSize) { shorts[it] / 32768f }

        // RingBuffer에 쓰기
        ringBuffer.write(floats)

        // 이전 remainder와 합치기
        val combined = if (vadRemainder.isNotEmpty()) {
            vadRemainder + floats
        } else {
            floats
        }

        // VAD에 512 samples 단위로 공급
        var offset = 0
        while (offset + 512 <= combined.size) {
            val frame = combined.copyOfRange(offset, offset + 512)
            val prob = vadEngine.process(frame)
            _vadProbability.value = prob
            offset += 512
        }

        // 남은 tail samples 저장
        vadRemainder = if (offset < combined.size) {
            combined.copyOfRange(offset, combined.size)
        } else {
            FloatArray(0)
        }
    }

    /**
     * STT가 실제로 사용 가능(인식 중)인 상태인지 확인.
     * **`LISTENING`만 인정** — `READY`는 "준비됐지만 아직 미시작" 의미라
     * 실제로는 어떤 음성 인식도 진행되지 않는 윈도우. 이 상태에서 combined 경로로
     * 보내면 sttAvailable=true로 잘못 기록되고 dedupe가 deepfake-only alert를 묻어버린다.
     * `ERROR`/`UNAVAILABLE`/`CONSENT_NEEDED`/`PAUSED`/`READY`는 모두 fallback 경로.
     *
     * **H101**: phishingDetector 유무는 STT 가용성과 무관하다. STT는 detector 없이도 전사가
     * 동작하므로 sttAvailable/sttStatus 판정에 detector 존재 여부를 섞지 않는다.
     */
    private fun isSttUsable(): Boolean {
        val engine = sttEngine ?: return false
        return engine.status.value == SttStatus.LISTENING
    }

    private suspend fun analyzeSegments(durationMs: Long) = inferenceMutex.withLock {
        val segments = segmentExtractor.extract(durationMs)
        for (segment in segments) {
            // **도메인 mismatch 보정**: AASIST에는 narrowband 필터링된 segment를 전달하되,
            // ring buffer / audio snapshot / VAD는 원본 PCM을 그대로 유지.
            val inferenceInput = narrowbandPreprocessor?.process(segment) ?: segment
            val result: DetectionResult = inferenceEngine.detect(inferenceInput)
            segmentsAnalyzed++

            val aggregated = aggregator.add(result)
            _latestResult.value = aggregated

            // **H102**: deepfake 발생은 STT 상태와 무관하게 counter를 증가시켜 기본 메트릭
            // 누락을 방지한다. Persistence/alert는 아래 두 경로로 분기되며, combined
            // dedupe에 걸려 emit이 suppress되면 detection path가 fallback으로 persistence를
            // 유지한다 (updateCombinedResult 내부 처리).
            if (aggregated.threatLevel.ordinal >= 1) {
                detectionsCount++
            }
            if (isSttUsable()) {
                updateCombinedResult(aggregated)
            } else {
                // **H185+**: _combinedResult.value에도 audioSnapshot을 탑재 — collector가
                // liveSnapshot 하나로 metadata-audio 쌍을 일관 read할 수 있도록.
                val snapshotAudio = ringBuffer.getLast(80_000)
                _combinedResult.value = combinedAggregator.combine(
                    deepfakeResult = aggregated,
                    phishingResult = null,
                    sttStatus = sttEngine?.status?.value ?: SttStatus.UNAVAILABLE,
                    transcription = "",
                ).copy(audioSnapshot = snapshotAudio)
                if (aggregated.threatLevel.ordinal >= 1) {
                    val seq = detectionEmitSeq.incrementAndGet()
                    pendingPersistenceJobs.incrementAndGet()
                    try {
                        _detectionEvents.emit(DetectionEnvelope(aggregated, seq, snapshotAudio))
                    } catch (t: Throwable) {
                        // emit 취소/실패 시 counter leak 방지 — drain이 영원히 기다리지 않도록.
                        pendingPersistenceJobs.decrementAndGet()
                        throw t
                    }
                }
            }
        }
    }

    /**
     * Branch B: 피싱 키워드 분석 (conflated — 최신 전사만 처리).
     * **In-flight detach 방어**: 분석 시작 시점 detector를 캡처하고, 결과 적용 직전에
     * 현재 detector와 동일성 확인. detach가 발생했으면 결과를 폐기.
     */
    private suspend fun runPhishingAnalysis(transcription: String) {
        val detector = phishingDetector ?: return
        val myGeneration = attachGeneration.get()
        val myIncidentGen = incidentGeneration.get()
        val mySeq = phishingSeq.incrementAndGet()
        pendingPhishingAnalyses.incrementAndGet()
        try {
            val result = runCatching {
                detector.analyze(transcription)
            }.getOrElse { t ->
                Log.w(TAG, "phishing analyze failed", t)
                return
            }
            // Pre-lock 빠른 가드 — 분명히 stale이면 mutex 안 잡고 폐기.
            if (phishingDetector !== detector || sttEngine == null) return
            if (attachGeneration.get() != myGeneration) return
            if (incidentGeneration.get() != myIncidentGen) return
            // **H131**: commit은 combinedMutex 안에서 generation을 *재검사*. SAFE-producing
            // updateCombinedResult가 동시에 incidentGeneration/attachGeneration을 올렸을 수 있으므로,
            // _phishingResult 갱신과 combined 재계산이 stale state 위에서 일어나지 않도록.
            combinedMutex.withLock {
                if (phishingDetector !== detector || sttEngine == null) return@withLock
                if (attachGeneration.get() != myGeneration) return@withLock
                if (incidentGeneration.get() != myIncidentGen) return@withLock
                while (true) {
                    val cur = lastAppliedPhishingSeq.get()
                    if (mySeq <= cur) return@withLock
                    if (lastAppliedPhishingSeq.compareAndSet(cur, mySeq)) break
                }
                _phishingResult.value = result
                lastPhishingResultTimeMs = nowMs()  // H150
                schedulePhishingExpiry()  // H153
                if (result.threatLevel.ordinal >= 1) {
                    phishingDetectionsCount++
                }
                val deepfakeResult = _latestResult.value
                if (deepfakeResult != null) {
                    // updateCombinedResult는 자체적으로 combinedMutex.withLock — 이미 보유 중이므로
                    // re-entry 발생. mutex가 reentrant인 Kotlin Mutex에서는 deadlock. 풀어서 호출.
                }
            }
            // mutex 외부에서 combined 재계산 — 위 commit은 _phishingResult 갱신만 처리.
            val deepfakeResult = _latestResult.value
            if (deepfakeResult != null) {
                updateCombinedResult(deepfakeResult)
            }
        } finally {
            pendingPhishingAnalyses.decrementAndGet()
        }
    }

    /**
     * 딥보이스 + 피싱 결과를 결합하여 통합 위협 레벨 갱신.
     *
     * [combinedMutex]로 직렬화하여 Branch A/B 동시 호출 시 StateFlow가
     * 혼합된 (stale deepfake × fresh phishing) 값을 내지 않도록 한다.
     *
     * Dedupe: 상태 전이(레벨 변경), 점수 변화(± [emitScoreDelta] 초과),
     * 또는 시간 경과([emitMinIntervalMs] 이상) 시에만 _combinedEvents를 emit.
     * UI 표시용 _combinedResult는 항상 업데이트.
     */
    @androidx.annotation.VisibleForTesting
    internal suspend fun updateCombinedResult(deepfakeResult: AggregatedResult) = combinedMutex.withLock {
        // **H101**: STT 가용성은 sttEngine 기반, phishing 가용성은 phishingDetector 기반으로 분리.
        // - sttDetached: STT 엔진이 붙어있지 않은 상태 → transcript/sttStatus 강제 무효화.
        // - phishingDisabled: detector가 없으면 피싱 점수/결과만 제외 (STT는 유지).
        val sttDetached = sttEngine == null
        val phishingDisabled = phishingDetector == null
        val sttStatus = if (sttDetached) SttStatus.UNAVAILABLE
            else sttEngine?.status?.value ?: SttStatus.UNAVAILABLE
        val transcription = if (sttDetached) "" else sttEngine?.transcription?.value ?: ""
        // **H150**: phishingResult age expiry — transcription 60s window를 벗어난 stale phishing이
        // 새 deepfake에 latch되어 escalate되는 것 차단.
        val nowForExpiry = nowMs()
        val phishingMaxAgeMs = phishingExpiryMs
        val phishingFresh = (_phishingResult.value)?.let { result ->
            if (nowForExpiry - lastPhishingResultTimeMs > phishingMaxAgeMs) null else result
        }
        if (phishingFresh == null && _phishingResult.value != null) {
            _phishingResult.value = null
        }
        val phishing = when {
            sttDetached || phishingDisabled -> {
                _phishingResult.value = null
                null
            }
            else -> phishingFresh
        }

        val combined = combinedAggregator.combine(
            deepfakeResult = deepfakeResult,
            phishingResult = phishing,
            sttStatus = sttStatus,
            transcription = transcription,
        )
        // **H185+**: updateCombinedResult 한 번의 실행 동안 ringBuffer를 단 한 번만 샘플.
        // 아래의 _combinedResult.value 갱신과 emit/fallback 경로가 모두 이 동일 snapshot을 재사용해
        // capture 스레드 쓰기가 사이에 끼어드는 skew를 차단.
        val snapshot = ringBuffer.getLast(80_000)
        _combinedResult.value = combined.copy(audioSnapshot = snapshot)

        if (combined.combinedThreatLevel.ordinal < 1) {
            resetAlertDedupeOnlyLocked()
            lastFallbackPersistTimeMs = 0L
            incidentGeneration.incrementAndGet()
            return@withLock
        }

        val now = nowMs()
        val currentLevel = combined.combinedThreatLevel.name
        val fakeScore = combined.deepfakeResult?.averageFakeScore ?: 0f
        val phishingScore = combined.phishingScore

        val levelChanged = currentLevel != lastEmittedThreatLevel
        val scoreJumped = kotlin.math.abs(fakeScore - lastEmittedFakeScore) > emitScoreDelta ||
            kotlin.math.abs(phishingScore - lastEmittedPhishingScore) > emitScoreDelta

        // **H107**: intervalElapsed 제거 — 안정된 incident가 10s마다 heartbeat로 row/audio를
        // 중복 저장하던 문제 차단. 변동 없는 incident는 alert/persistence 모두 suppress하고,
        // 아래의 fallback 경로(H105 30s gate)가 긴 interval로만 기록을 남긴다.
        val shouldEmit = levelChanged || scoreJumped
        if (shouldEmit) {
            lastEmittedThreatLevel = currentLevel
            lastEmittedFakeScore = fakeScore
            lastEmittedPhishingScore = phishingScore
            lastEmittedTimeMs = now
            lastFallbackPersistTimeMs = now
            // snapshot은 함수 시작 시 한 번 샘플된 것 재사용 (metadata와 atomic pair 보장).
            pendingPersistenceJobs.incrementAndGet()
            try {
                _combinedEvents.emit(
                    combined.copy(
                        emitGeneration = incidentGeneration.get(),
                        audioSnapshot = snapshot,
                    )
                )
            } catch (t: Throwable) {
                pendingPersistenceJobs.decrementAndGet()
                throw t
            }
        } else if (combined.combinedThreatLevel.ordinal >= 1) {
            val fallbackIntervalElapsed =
                (now - lastFallbackPersistTimeMs) >= fallbackPersistIntervalMs
            if (fallbackIntervalElapsed) {
                lastFallbackPersistTimeMs = now
                pendingPersistenceJobs.incrementAndGet()
                try {
                    _combinedEvents.emit(
                        combined.copy(
                            suppressAlert = true,
                            emitGeneration = incidentGeneration.get(),
                            audioSnapshot = snapshot,
                        )
                    )
                } catch (t: Throwable) {
                    pendingPersistenceJobs.decrementAndGet()
                    throw t
                }
            }
        }
    }

    /**
     * In-flight inference가 완료될 때까지 대기.
     * inferenceMutex를 잡고 즉시 풀어 — 그 시점에 진행 중이던 analyzeSegments가
     * 끝났음을 보장한다. shutdown drain 단계에서 호출.
     */
    suspend fun awaitInferenceIdle() {
        inferenceMutex.withLock { /* lock acquired = previous holder finished */ }
    }

    /**
     * In-flight 통합 분석(updateCombinedResult / runPhishingAnalysis)이 완료될 때까지 대기.
     * combinedMutex를 잡고 풀어 — 진행 중이던 통합 분석이 끝났음을 보장.
     */
    suspend fun awaitCombinedIdle() {
        combinedMutex.withLock { /* lock acquired = previous holder finished */ }
    }

    /**
     * In-flight phishing analysis가 모두 완료될 때까지 대기 (10ms polling).
     * Counter 기반이라 detach 전에 호출하여 in-flight analysis가 detach guard로
     * drop되지 않도록 한다.
     */
    suspend fun awaitPhishingIdle() {
        while (pendingPhishingAnalyses.get() > 0) {
            kotlinx.coroutines.delay(10L)
        }
    }

    /** Deprecated — H48 fix로 제거됨. 호환성 유지를 위해 빈 구현. */
    @Deprecated("H48: 더 이상 사용 안 함. fence는 GoogleSttEngine.pendingResultJobs로.")
    suspend fun awaitTranscriptionDispatchIdle() { /* no-op */ }

    /**
     * Inference + combined + phishing 모두 idle.
     *
     * 순서가 중요: analyzeSegments는 inference 후 updateCombinedResult를 호출하므로
     * inference 먼저 await → combined → phishing 순. 그리고 두 번 반복하여
     * 첫 번째 awaitInferenceIdle 직후 새로 시작된 combined/phishing이 있을 가능성을 cover.
     */
    suspend fun awaitFullyIdle() {
        repeat(2) {
            // 1) inference detect (analyzeSegments 진입점).
            awaitInferenceIdle()
            // 2) combined 분석.
            awaitCombinedIdle()
            // 3) phishing analysis 자체 (STT publish worker가 직접 호출).
            awaitPhishingIdle()
            // 4) phishing 결과 → 다시 combined 업데이트.
            awaitCombinedIdle()
        }
    }

    /** STT 엔진 시작 (capability gate + 동의 확인 후). */
    fun startStt() {
        sttEngine?.start()
    }

    /** STT 엔진 중지. */
    suspend fun stopStt() {
        sttEngine?.stop()
    }

    /**
     * 런타임에 STT 엔진과 피싱 디텍터를 부착한다 (positive consent transition).
     * 기존 STT 엔진은 호출자가 destroy 책임을 진다.
     * Combined dedupe state도 reset하여 이전 STT 세션의 suppression이 새 세션에 영향 안 가게.
     */
    suspend fun attachSttAndPhishing(
        newSttEngine: SttEngine,
        newPhishingDetector: PhishingKeywordDetector?,
    ) {
        // **Pre-attach drain**: inference + combined + phishing 모두 await.
        // phishing이 누락되면 old 분석이 generation rollover와 겹쳐 new epoch로 누출.
        runCatching {
            kotlinx.coroutines.withTimeoutOrNull(1500L) {
                awaitFullyIdle()
            }
        }
        combinedMutex.withLock {
            // **Generation increment**: 이전 세션의 in-flight 분석이 이 attach 후 결과 적용 못 하게.
            attachGeneration.incrementAndGet()
            sttEngine = newSttEngine
            phishingDetector = newPhishingDetector
            // **Dedupe state reset** — 이전 세션의 suppression이 새 STT 세션의 첫 alert를 묻지 않도록.
            resetCombinedDedupeStateLocked()
            // 새 STT의 전사를 구독하는 collector 재시작.
            startSttTranscriptionCollector()
        }
    }

    /**
     * H98: 피싱 분석만 ON/OFF (STT 유지). `phishingDetectionEnabled` 토글에 대응.
     *
     * OFF 전환 시: phishingDetector를 null로 바꾸고 `_combinedResult.combinedThreatLevel`을
     * 즉시 deepfake-only로 강등, 피싱 점수/키워드를 클리어. STT는 계속 동작하므로
     * 전사 자체는 유지되지만 피싱 판단은 중단된다.
     *
     * ON 전환 시: 호출자가 새로운 detector를 주입한다 (일반적으로 attachSttAndPhishing
     * 경로에서 처리되지만, 이미 STT가 attach된 상태라면 여기로도 진입).
     */
    suspend fun setPhishingEnabled(
        enabled: Boolean,
        detector: PhishingKeywordDetector? = null,
    ) {
        // **H139**: OFF 전환은 mutex 진입 *이전*에 phishingDetector를 즉시 null로 설정 —
        // mutex 대기 윈도우 동안 updateCombinedResult가 이미 disable된 detector로 stale phishing
        // emit하는 race를 차단한다. @Volatile field이므로 다른 coroutine은 즉시 변화를 본다.
        if (!enabled) {
            phishingDetector = null
        }
        // Generation increment — in-flight 분석이 새 state에 영향 못 가도록.
        attachGeneration.incrementAndGet()
        // **H137**: toggle 전환 시 STT transcript window 클리어 — retroactive 분석 차단.
        runCatching { sttEngine?.resetTranscriptionWindow() }
        combinedMutex.withLock {
            if (!enabled) {
                phishingDetector = null
                _phishingResult.value = null
                resetCombinedDedupeStateLocked()
                sttEngine?.let { installPhishingListener(it) }
                _combinedResult.value = _combinedResult.value?.let { prev ->
                    val downgraded = CombinedThreatAggregator.deepfakeOnlyLevel(prev.deepfakeThreatLevel)
                    prev.copy(
                        matchedKeywords = emptyList(),
                        matchedPhrases = emptyList(),
                        phishingScore = 0f,
                        phishingResult = null,
                        combinedThreatLevel = downgraded,
                    )
                }
            } else if (detector != null) {
                phishingDetector = detector
                resetCombinedDedupeStateLocked()
                // **H135**: ON 시 backlog 분석 제거 — disabled 동안 누적된 transcript를 retroactive
                // 분석하면 toggle boundary 위반 (사용자가 disable한 시점의 음성이 alert를 트리거).
                // ON 이후 새 final transcription만 listener로 분석.
                sttEngine?.let { installPhishingListener(it) }
            }
        }
        if (!enabled) {
            // **H133/H136**: OFF 전환은 drained — in-flight 분석이 stale data publish 못 하게
            // mutex 해제 후 awaitPhishingIdle/awaitCombinedIdle. timeout은 명시 로깅하여
            // degraded state 가시화.
            val drained = kotlinx.coroutines.withTimeoutOrNull(1500L) {
                awaitPhishingIdle()
                awaitCombinedIdle()
                true
            }
            if (drained == null) {
                android.util.Log.w(
                    "AudioPipeline",
                    "setPhishingEnabled(false) drain TIMEOUT — late phishing/combined publish 가능"
                )
            }
        }
    }

    /**
     * 동의 철회/설정 OFF 시 STT/피싱 state를 파이프라인에서 완전히 분리한다.
     * 이후 combined 결과는 STT unavailable 상태로 계산되어 전사/키워드가 유출되지 않는다.
     */
    suspend fun detachSttAndPhishing() {
        // **Generation increment**: in-flight 분석이 detach 후 결과 적용 못 하게.
        attachGeneration.incrementAndGet()
        // 1. transcription collector 먼저 cancel — 이후 phishing 분석이 새로 트리거되지 않음.
        sttTranscriptionJob?.cancel()
        sttTranscriptionJob = null

        val engine = sttEngine
        // listener 해제 — detach 후 STT publish가 stale 분석을 트리거하지 않게.
        runCatching { engine?.setOnFinalTranscriptionListener(null) }
        sttEngine = null
        phishingDetector = null
        combinedMutex.withLock {
            _phishingResult.value = null
            // **Dedupe state reset** — 다음 STT 재부착 시 stale state로 인해 첫 알림이
            // suppress되지 않도록.
            resetCombinedDedupeStateLocked()
            // H96: detach 즉시 combinedThreatLevel도 deepfake-only 수준으로 강등.
            // 이 작업을 누락하면 HomeScreen(H93 canonical source)이 stale
            // CRITICAL/WARNING을 계속 표시할 수 있다 — 다음 deepfake 결과 갱신이
            // 올 때까지 UI가 이미 해제된 피싱 증거 기반 경고를 보여주는 false-alarm.
            _combinedResult.value = _combinedResult.value?.let { prev ->
                val downgraded = CombinedThreatAggregator.deepfakeOnlyLevel(prev.deepfakeThreatLevel)
                prev.copy(
                    transcription = "",
                    matchedKeywords = emptyList(),
                    matchedPhrases = emptyList(),
                    phishingScore = 0f,
                    sttAvailable = false,
                    phishingResult = null,
                    combinedThreatLevel = downgraded,
                )
            }
        }
        runCatching { engine?.destroy() }
    }

    /**
     * **H153**: phishingResult가 갱신될 때마다 expiry job을 스케줄. 60s 뒤 _phishingResult를
     * 비우고 _combinedResult를 deepfake-only로 강등 publish — 조용한 시간에도 stale phishing이
     * canonical UI state에 latched되지 않게 한다.
     */
    private fun schedulePhishingExpiry() {
        phishingExpiryJob?.cancel()
        phishingExpiryJob = scope.launch(workerDispatcher) {
            kotlinx.coroutines.delay(phishingExpiryMs)
            combinedMutex.withLock {
                val now = nowMs()
                if (now - lastPhishingResultTimeMs < phishingExpiryMs) return@withLock
                if (_phishingResult.value == null) return@withLock
                _phishingResult.value = null
                // _combinedResult downgrade.
                _combinedResult.value = _combinedResult.value?.let { prev ->
                    val downgraded = CombinedThreatAggregator.deepfakeOnlyLevel(prev.deepfakeThreatLevel)
                    prev.copy(
                        matchedKeywords = emptyList(),
                        matchedPhrases = emptyList(),
                        phishingScore = 0f,
                        phishingResult = null,
                        combinedThreatLevel = downgraded,
                    )
                }
                // **H155**: SAFE-transition cleanup 동일 수행 — alert dedupe / fallback timer /
                // incident generation 모두 reset. 다음 incident가 같은 레벨로 와도 fresh로 처리되고
                // pre-expiry in-flight 분석이 incident 재 elevate하지 못함.
                resetAlertDedupeOnlyLocked()
                lastFallbackPersistTimeMs = 0L
                incidentGeneration.incrementAndGet()
            }
            // **H157**: NotificationHelper 측 incident state도 정리하도록 외부에 통보.
            // mutex 외부에서 호출하여 listener 안의 별도 lock 사용 시에도 deadlock 회피.
            runCatching { phishingExpiredListener?.invoke() }
        }
    }

    /**
     * **H126**: alert dedupe state만 리셋. phishingSeq/lastAppliedPhishingSeq freshness
     * 가드는 보존하여 stale in-flight 분석이 reset 직후 CAS를 통과하지 않게 한다.
     */
    private fun resetAlertDedupeOnlyLocked() {
        lastEmittedThreatLevel = null
        lastEmittedFakeScore = -1f
        lastEmittedPhishingScore = -1f
        lastEmittedTimeMs = 0L
    }

    /** combinedMutex 보유 상태에서 dedupe state를 초기값으로 복원한다. */
    private fun resetCombinedDedupeStateLocked() {
        lastEmittedThreatLevel = null
        lastEmittedFakeScore = -1f
        lastEmittedPhishingScore = -1f
        lastEmittedTimeMs = 0L
        // Phishing sequence도 reset — detach/reattach 사이에 stale sequence가 새 세션에 영향 X.
        phishingSeq.set(0)
        lastAppliedPhishingSeq.set(0)
    }

    fun getStats(): PipelineStats = PipelineStats(
        segmentsAnalyzed = segmentsAnalyzed,
        detectionsCount = detectionsCount,
        phishingDetectionsCount = phishingDetectionsCount,
    )

    fun reset() {
        ringBuffer.clear()
        vadEngine.reset()
        aggregator.clear()
        segmentsAnalyzed = 0
        detectionsCount = 0
        phishingDetectionsCount = 0
        _latestResult.value = null
        _phishingResult.value = null
        _combinedResult.value = null
    }
}

data class PipelineStats(
    val segmentsAnalyzed: Long,
    val detectionsCount: Int,
    val phishingDetectionsCount: Int = 0,
)
