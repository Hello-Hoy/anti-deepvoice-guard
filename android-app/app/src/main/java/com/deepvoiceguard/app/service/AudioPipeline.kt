package com.deepvoiceguard.app.service

import com.deepvoiceguard.app.audio.RingBuffer
import com.deepvoiceguard.app.audio.SegmentExtractor
import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.DetectionAggregator
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.InferenceEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

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
) {
    val ringBuffer = RingBuffer()
    private val segmentExtractor = SegmentExtractor(ringBuffer)
    private val aggregator = DetectionAggregator(
        dangerThreshold = dangerThreshold,
        warningThreshold = (dangerThreshold - 0.2f).coerceAtLeast(0.5f),
        cautionThreshold = (dangerThreshold - 0.3f).coerceAtLeast(0.4f),
    )

    private val _latestResult = MutableStateFlow<AggregatedResult?>(null)
    val latestResult: StateFlow<AggregatedResult?> = _latestResult

    private val _detectionEvents = MutableSharedFlow<AggregatedResult>(extraBufferCapacity = 16)
    val detectionEvents: SharedFlow<AggregatedResult> = _detectionEvents

    private val _vadProbability = MutableStateFlow(0f)
    val vadProbability: StateFlow<Float> = _vadProbability

    private val inferenceMutex = Mutex()
    private var segmentsAnalyzed = 0L
    private var detectionsCount = 0

    // VAD에 512 단위로 공급 시 남는 tail samples 버퍼
    private var vadRemainder = FloatArray(0)

    init {
        vadEngine.listener = object : VadEngine.Listener {
            override fun onSpeechStart(timestampMs: Long) {
                // 음성 시작 — 로깅만
            }

            override fun onSpeechEnd(timestampMs: Long, durationMs: Long) {
                // 음성 종료 → 세그먼트 추출 → 추론
                scope.launch(Dispatchers.Default) {
                    analyzeSegments(durationMs)
                }
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

    private suspend fun analyzeSegments(durationMs: Long) = inferenceMutex.withLock {
        val segments = segmentExtractor.extract(durationMs)
        for (segment in segments) {
            val result: DetectionResult = inferenceEngine.detect(segment)
            segmentsAnalyzed++

            val aggregated = aggregator.add(result)
            _latestResult.value = aggregated

            if (aggregated.threatLevel.ordinal >= 1) {
                detectionsCount++
                _detectionEvents.emit(aggregated)
            }
        }
    }

    fun getStats(): PipelineStats = PipelineStats(
        segmentsAnalyzed = segmentsAnalyzed,
        detectionsCount = detectionsCount,
    )

    fun reset() {
        ringBuffer.clear()
        vadEngine.reset()
        aggregator.clear()
        segmentsAnalyzed = 0
        detectionsCount = 0
        _latestResult.value = null
    }
}

data class PipelineStats(
    val segmentsAnalyzed: Long,
    val detectionsCount: Int,
)
