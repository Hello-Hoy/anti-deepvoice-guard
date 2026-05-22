package com.deepvoiceguard.app.inference

import android.content.Context
import com.deepvoiceguard.app.phishing.PhishingKeywordDetector
import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.stt.SttStatus
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 데모용 파일 분석 파이프라인.
 * AudioCaptureService와 완전히 독립적으로 동작한다.
 * 오디오 파일 → AASIST 추론 + 사전 전사본 → 피싱 탐지 → 통합 결과.
 *
 * Demo WAV asset은 tools/preprocess_demo_wav.py로 narrowband 전처리된 상태로 저장되므로
 * 여기서는 별도 필터링 없이 직접 AASIST에 전달한다 (live mic 경로와 달리 runtime 필터 불필요).
 */
class DemoAnalysisPipeline(
    private val context: Context,
    private val inferenceEngine: InferenceEngine,
    private val phishingDetector: PhishingKeywordDetector,
    private val vadEngine: com.deepvoiceguard.app.audio.VadEngine? = null,
) {

    private val combinedAggregator = CombinedThreatAggregator()

    /**
     * 데모 시나리오를 분석한다.
     *
     * @param audioAssetPath assets 내 오디오 파일 경로 (예: "demo/demo_01.wav")
     * @param transcriptAssetPath assets 내 전사본 파일 경로 (예: "demo/demo_01_transcript.txt")
     * @return 분석 결과
     */
    /**
     * 시나리오 분석을 수행한다.
     *
     * **Fail-closed 정책:** 오디오 asset이 누락되었거나 ONNX 추론이 실패하면
     * [DemoAnalysisException]을 던진다. 전사본만으로 만든 결과를 성공으로
     * 보고하지 않음 — 딥보이스 탐지가 동작하지 않았다는 사실을 숨기지 않는다.
     */
    @Throws(DemoAnalysisException::class)
    suspend fun analyze(
        audioAssetPath: String,
        transcriptAssetPath: String? = null,
    ): DemoResult {
        // 1. 오디오 로드 (필수) → AASIST 추론
        val audio = loadWavFromAssets(audioAssetPath)
            ?: throw DemoAnalysisException(
                "오디오 asset을 찾을 수 없습니다: $audioAssetPath",
                DemoFailureReason.AUDIO_MISSING,
            )
        if (!inferenceEngine.isReady()) {
            throw DemoAnalysisException(
                "추론 엔진이 준비되지 않았습니다",
                DemoFailureReason.ENGINE_NOT_READY,
            )
        }
        // 시나리오마다 독립적인 aggregator — 이전 시나리오의 슬라이딩 윈도우가 오염되지 않음.
        // 4초보다 긴 오디오는 50% 오버랩 sliding window로 여러 세그먼트 추론해 aggregate.
        // (길이가 짧으면 pad 단일 세그먼트.)
        val detectionResult = try {
            val segments = segmentAudio(audio)
            val aggregator = DetectionAggregator()
            var last = aggregator.add(inferenceEngine.detect(segments.first()))
            for (i in 1 until segments.size) {
                last = aggregator.add(inferenceEngine.detect(segments[i]))
            }
            last
        } catch (e: Exception) {
            throw DemoAnalysisException(
                "AASIST 추론 실패: ${e.message}",
                DemoFailureReason.INFERENCE_FAILED,
                e,
            )
        }

        // 2. 전사본 로드 (사전 준비된 .txt) — **fail-closed**:
        // transcriptAssetPath가 지정됐는데 로드 실패/공백이면 예외로 명시 실패.
        // 이렇게 해야 broken transcript asset이 silent deepfake-only로 강등되지 않음.
        val transcript = if (transcriptAssetPath != null) {
            val loaded = loadTextFromAssets(transcriptAssetPath)
            if (loaded.isBlank()) {
                throw DemoAnalysisException(
                    "전사본을 읽을 수 없거나 비어 있습니다: $transcriptAssetPath",
                    DemoFailureReason.TRANSCRIPT_MISSING,
                )
            }
            loaded
        } else ""

        // 3. 피싱 키워드 탐지
        val phishingResult: PhishingResult? = if (transcript.isNotBlank()) {
            phishingDetector.analyze(transcript)
        } else null

        // 4. 통합 결과
        // **Offline/demo**: 사전 전사본이 있으면 LISTENING과 동등하게 취급한다.
        // live-capture LISTENING-only gate(M6)와 동등한 semantics를 demo path에도 제공.
        val combined = combinedAggregator.combine(
            deepfakeResult = detectionResult,
            phishingResult = phishingResult,
            sttStatus = if (transcript.isNotBlank()) SttStatus.LISTENING else SttStatus.UNAVAILABLE,
            transcription = transcript,
        )

        return DemoResult(
            deepfakeResult = detectionResult,
            phishingResult = phishingResult,
            combinedResult = combined,
            transcript = transcript,
            audioLoaded = true,
        )
    }

    /**
     * 재생 동기 replay용 타임라인. VAD(32ms 프레임) + 딥보이스(1초 슬라이딩 윈도우 누적)
     * + 100ms 격자별 피싱/통합위협. fail-closed는 analyze와 동일.
     */
    @Throws(DemoAnalysisException::class)
    suspend fun buildTimeline(
        audioAssetPath: String,
        transcriptAssetPath: String? = null,
    ): DemoTimeline {
        val audio = loadWavFromAssets(audioAssetPath)
            ?: throw DemoAnalysisException("오디오 asset을 찾을 수 없습니다: $audioAssetPath", DemoFailureReason.AUDIO_MISSING)
        if (!inferenceEngine.isReady()) {
            throw DemoAnalysisException("추론 엔진이 준비되지 않았습니다", DemoFailureReason.ENGINE_NOT_READY)
        }
        val sr = 16000
        val durationMs = (audio.size * 1000L / sr).toInt().coerceAtLeast(DemoTimelineMath.FRAME_MS)

        // 1) 딥보이스 1초 슬라이딩 윈도우 누적
        val stepMs = 1000
        val win = 64_600
        val steps = ArrayList<Pair<Int, AggregatedResult>>()
        val aggregator = DetectionAggregator()
        try {
            var endSample = stepMs * sr / 1000
            while (true) {
                val e = minOf(endSample, audio.size)
                val window = DemoTimelineMath.windowEndingAt(audio, e, win)
                val agg = aggregator.add(inferenceEngine.detect(window))
                steps.add((e * 1000L / sr).toInt() to agg)
                if (e >= audio.size) break
                endSample += stepMs * sr / 1000
            }
        } catch (ex: Exception) {
            throw DemoAnalysisException("AASIST 추론 실패: ${ex.message}", DemoFailureReason.INFERENCE_FAILED, ex)
        }
        val stepTimes = IntArray(steps.size) { steps[it].first }

        // 2) VAD 프레임별 speechProb (엔진 실패는 비치명)
        var vadAvailable = false
        val vadProbs: FloatArray = try {
            vadEngine?.let { engine ->
                engine.reset()
                val frameLen = 512
                val n = audio.size / frameLen
                FloatArray(n) { i ->
                    engine.process(audio.copyOfRange(i * frameLen, i * frameLen + frameLen))
                }.also { vadAvailable = true }
            } ?: FloatArray(0)
        } catch (_: Exception) {
            vadAvailable = false
            FloatArray(0)
        }
        val vadFrameMs = 512 * 1000 / sr

        // 3) 전사 (fail-closed)
        val transcript = if (transcriptAssetPath != null) {
            val loaded = loadTextFromAssets(transcriptAssetPath)
            if (loaded.isBlank()) throw DemoAnalysisException("전사본을 읽을 수 없거나 비어 있습니다: $transcriptAssetPath", DemoFailureReason.TRANSCRIPT_MISSING)
            loaded
        } else ""

        // 4) 100ms 격자 frame 생성
        val frames = ArrayList<DemoFrame>()
        var t = 0
        while (t <= durationMs) {
            val vadLookupMs = if (vadProbs.isEmpty()) t else minOf(t, (vadProbs.size - 1) * vadFrameMs)
            val vadActive = DemoTimelineMath.vadActiveAt(vadProbs, vadFrameMs, vadLookupMs, 0.5f)
            val stepIdx = DemoTimelineMath.stepHoldIndexAt(stepTimes, t)
            val agg = if (stepIdx >= 0) steps[stepIdx].second else null
            val chars = DemoTimelineMath.transcriptCharsAt(transcript.length, t, durationMs)
            val revealed = transcript.substring(0, chars)
            val phishing = if (revealed.isNotBlank()) phishingDetector.analyze(revealed) else null
            val combined = combinedAggregator.combine(
                deepfakeResult = agg,
                phishingResult = phishing,
                sttStatus = if (revealed.isNotBlank()) SttStatus.LISTENING else SttStatus.UNAVAILABLE,
                transcription = revealed,
            )
            frames.add(
                DemoFrame(
                    offsetMs = t,
                    vadActive = vadActive,
                    fakeScore = agg?.averageFakeScore ?: 0f,
                    deepfakeLevel = agg?.threatLevel ?: ThreatLevel.SAFE,
                    phishingScore = combined.phishingScore,
                    threatLevel = combined.combinedThreatLevel,
                    transcriptChars = chars,
                )
            )
            t += DemoTimelineMath.FRAME_MS
        }

        // 5) 최종 요약
        val lastAgg = steps.lastOrNull()?.second
        val finalPhishing = if (transcript.isNotBlank()) phishingDetector.analyze(transcript) else null
        val finalCombined = combinedAggregator.combine(
            deepfakeResult = lastAgg,
            phishingResult = finalPhishing,
            sttStatus = if (transcript.isNotBlank()) SttStatus.LISTENING else SttStatus.UNAVAILABLE,
            transcription = transcript,
        )
        val finalResult = DemoResult(
            deepfakeResult = lastAgg,
            phishingResult = finalPhishing,
            combinedResult = finalCombined,
            transcript = transcript,
            audioLoaded = true,
        )

        return DemoTimeline(durationMs, frames, transcript, vadAvailable, finalResult)
    }

    /** 주어진 asset 경로의 파일이 실제로 존재하는지 확인한다. */
    fun assetExists(path: String): Boolean = try {
        context.assets.open(path).use { true }
    } catch (_: Exception) {
        false
    }

    /** assets에서 WAV 파일을 16kHz mono float로 로드. */
    private fun loadWavFromAssets(path: String): FloatArray? {
        return try {
            context.assets.open(path).use { stream ->
                val bytes = stream.readBytes()
                parseWav(bytes)
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * WAV 파서: RIFF 서브청크를 스캔해 "data" 청크 위치를 찾는다 (16-bit PCM mono 가정).
     * 고정 44바이트 오프셋은 LIST 등 optional chunk가 있으면 metadata를 audio로 오독함.
     */
    private fun parseWav(bytes: ByteArray): FloatArray? {
        if (bytes.size < 12) return null
        val riff = String(bytes, 0, 4, Charsets.US_ASCII)
        val wave = String(bytes, 8, 4, Charsets.US_ASCII)
        if (riff != "RIFF" || wave != "WAVE") return null
        var offset = 12
        var dataStart = -1
        var dataSize = -1
        while (offset + 8 <= bytes.size) {
            val id = String(bytes, offset, 4, Charsets.US_ASCII)
            val size = ByteBuffer.wrap(bytes, offset + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
            if (size < 0) return null
            val contentStart = offset + 8
            if (id == "data") {
                dataStart = contentStart
                dataSize = minOf(size, bytes.size - contentStart).coerceAtLeast(0)
                break
            }
            offset = contentStart + size + (size and 1) // word-align padding
        }
        if (dataStart < 0 || dataSize <= 0) return null
        val numSamples = dataSize / 2
        if (numSamples <= 0) return null
        val buffer = ByteBuffer.wrap(bytes, dataStart, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(numSamples) { buffer.short / 32768f }
    }

    /**
     * 긴 오디오는 50% 오버랩으로 여러 세그먼트로 분할, 짧으면 단일 pad 세그먼트.
     * 경진대회 데모는 5-15초 범위이므로 보통 2-7개 세그먼트를 aggregate한다.
     */
    private fun segmentAudio(audio: FloatArray): List<FloatArray> {
        val targetLen = 64_600
        if (audio.size <= targetLen) return listOf(prepareSegment(audio))
        val hop = targetLen / 2  // 50% overlap
        val segments = mutableListOf<FloatArray>()
        var start = 0
        while (start + targetLen <= audio.size) {
            segments.add(audio.copyOfRange(start, start + targetLen))
            start += hop
        }
        // 마지막 tail 구간도 pad해서 포함 (이미 커버된 구간과 겹칠 수 있음).
        if (start < audio.size) {
            val tail = FloatArray(targetLen)
            audio.copyInto(tail, 0, start, audio.size)
            segments.add(tail)
        }
        return segments
    }

    /** 64600 samples로 pad/crop. */
    private fun prepareSegment(audio: FloatArray): FloatArray {
        val targetLen = 64600
        return when {
            audio.size == targetLen -> audio
            audio.size > targetLen -> audio.copyOfRange(0, targetLen)
            else -> {
                val padded = FloatArray(targetLen)
                audio.copyInto(padded)
                padded
            }
        }
    }

    private fun loadTextFromAssets(path: String): String {
        return try {
            context.assets.open(path).use { stream ->
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8))
                    .readText().trim()
            }
        } catch (_: Exception) {
            ""
        }
    }
}

/** 데모 분석 결과. */
data class DemoResult(
    val deepfakeResult: AggregatedResult?,
    val phishingResult: PhishingResult?,
    val combinedResult: CombinedThreatResult,
    val transcript: String,
    val audioLoaded: Boolean,
)

/** 데모 시나리오 정의. */
data class DemoScenario(
    val id: Int,
    val title: String,
    val description: String,
    val audioAsset: String,
    val transcriptAsset: String,
    val expectedResult: String,
)

/** 데모 분석 실패 원인. */
enum class DemoFailureReason {
    AUDIO_MISSING,       // 오디오 asset 파일이 없음
    TRANSCRIPT_MISSING,  // 전사본 asset이 없거나 비어 있음
    ENGINE_NOT_READY,    // ONNX 엔진이 초기화되지 않음
    INFERENCE_FAILED,    // AASIST 추론 실행 실패
}

/** 데모 fail-closed 예외. */
class DemoAnalysisException(
    message: String,
    val reason: DemoFailureReason,
    cause: Throwable? = null,
) : Exception(message, cause)
