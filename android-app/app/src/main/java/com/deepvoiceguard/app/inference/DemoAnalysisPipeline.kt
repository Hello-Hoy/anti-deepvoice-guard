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
 * 딥보이스 표시 바 전용 EMA 평활 계수. 모델이 real 통화의 일부 구간을 borderline(0.5~0.6)으로
 * 내며 표시 바가 통화 중간에 과하게 솟는 것(예: #5 은행정상 ~0.54)을 완만하게 낮춘다.
 * **표시값만** 평활한다 — 딥보이스 위협등급은 raw 점수로 판정해 불변.
 * α=0.04(시정수 ≈2.5초)면 문장 사이 묵음/숨 구간에서 fakeScore가 0.4~0.5로 잠깐 꺼져도
 * 표시 점수가 급락하지 않아(예: demo_11) 출렁임이 크게 줄고, 지속적 딥보이스는 0.9대를 유지한다.
 */
private const val DEEPVOICE_SMOOTH_ALPHA = 0.04f

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

    // 데모 표시 전용: 평활된 딥보이스 표시 점수로 등급을 매겨 점수-등급-통합을 한 화면에서 일치시킨다.
    // 임계는 DetectionAggregator(danger 0.9 / warning 0.7 / caution 0.6)와 동일.
    private fun deepfakeLevelFromScore(score: Float): ThreatLevel = when {
        score > 0.9f -> ThreatLevel.DANGER
        score > 0.7f -> ThreatLevel.WARNING
        score > 0.6f -> ThreatLevel.CAUTION
        else -> ThreatLevel.SAFE
    }

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
            // 타임스탬프 마커가 있으면 순수 텍스트만 추출(자막/피싱용). 없으면 원문 그대로.
            DemoTimelineMath.parseTranscript(loaded).text
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
        // 타임스탬프 마커(`[startMs-endMs] 텍스트`)가 있으면 세그먼트 기반으로 자막을 실제 발화
        // 시점에 동기화한다. 없으면 segments=null → 기존 시간 비례 공개로 fallback.
        val parsedTranscript = if (transcriptAssetPath != null) {
            val loaded = loadTextFromAssets(transcriptAssetPath)
            if (loaded.isBlank()) throw DemoAnalysisException("전사본을 읽을 수 없거나 비어 있습니다: $transcriptAssetPath", DemoFailureReason.TRANSCRIPT_MISSING)
            DemoTimelineMath.parseTranscript(loaded)
        } else DemoTimelineMath.ParsedTranscript("", null)
        val transcript = parsedTranscript.text
        val transcriptSegments = parsedTranscript.segments

        // 최종 요약(결과 카드)용 전체 전사 분석.
        val fullPhishing = if (transcript.isNotBlank()) phishingDetector.analyze(transcript) else null

        // 4) 100ms 격자 frame 생성
        val frames = ArrayList<DemoFrame>()
        var t = 0
        var deepfakeDisplay = 0f  // 딥보이스 표시 점수: raw averageFakeScore를 EMA로 완만하게 따라감
        // 표시 일관성 원칙(사용자 피드백 반영):
        //  - 딥보이스 점수(완만)와 괄호 등급·통합 위협을 **모두 표시 점수 기준**으로 산출해 일치시킨다.
        //    (점수는 리니어하게 오르는데 등급만 raw로 갑자기 DANGER로 튀던 어색함 제거.)
        //  - 피싱 점수는 진행률이 아니라 현재까지 전사의 **실제 score**를 그대로 반영한다(결과 카드와 동일).
        //  - 통합은 latching 없이 매 프레임 의사결정 테이블 반영. 최종 요약만 worstAgg로 산출(아래 5)).
        while (t <= durationMs) {
            val vadLookupMs = if (vadProbs.isEmpty()) t else minOf(t, (vadProbs.size - 1) * vadFrameMs)
            val vadActive = DemoTimelineMath.vadActiveAt(vadProbs, vadFrameMs, vadLookupMs, 0.5f)
            val stepIdx = DemoTimelineMath.stepHoldIndexAt(stepTimes, t)
            val agg = if (stepIdx >= 0) steps[stepIdx].second else null
            val chars = if (transcriptSegments != null) {
                DemoTimelineMath.transcriptCharsTimed(transcriptSegments, t)
            } else {
                DemoTimelineMath.transcriptCharsAt(transcript.length, t, durationMs)
            }.coerceIn(0, transcript.length)
            val revealed = transcript.substring(0, chars)
            val phishing = if (revealed.isNotBlank()) phishingDetector.analyze(revealed) else null

            // 딥보이스 표시 점수를 먼저 EMA로 완만하게 갱신한 뒤, 그 표시 점수로 등급을 매긴다.
            val deepfakeTarget = agg?.averageFakeScore ?: 0f
            deepfakeDisplay += (deepfakeTarget - deepfakeDisplay) * DEEPVOICE_SMOOTH_ALPHA
            val displayLevel = deepfakeLevelFromScore(deepfakeDisplay)
            // 통합 위협도 표시 등급 기준으로 산출(점수·괄호등급·통합을 한 화면에서 일치시키기 위함).
            val displayAgg = agg?.let {
                AggregatedResult(displayLevel, deepfakeDisplay, it.latestResult, it.consecutiveHighCount)
            }
            // 피싱 실시간 점수 = 현재까지 전사의 실제 score. 키워드/금액이 탐지되는 순간 그 값으로
            // 오르고, 추가 신호가 없으면 변동 없이 유지된다(결과 카드 score와 동일 기준).
            val phishingNow = phishing?.score ?: 0f
            val combined = combinedAggregator.combine(
                deepfakeResult = displayAgg,
                phishingResult = phishing,
                sttStatus = if (revealed.isNotBlank()) SttStatus.LISTENING else SttStatus.UNAVAILABLE,
                transcription = revealed,
            )
            frames.add(
                DemoFrame(
                    offsetMs = t,
                    vadActive = vadActive,
                    fakeScore = deepfakeDisplay,
                    deepfakeLevel = displayLevel,
                    phishingScore = phishingNow,
                    threatLevel = combined.combinedThreatLevel,
                    transcriptChars = chars,
                )
            )
            t += DemoTimelineMath.FRAME_MS
        }

        // 5) 최종 요약
        // 마지막 스텝이 아니라 통화 전체에서 가장 위협이 높았던 스텝(worstAgg)을 요약 딥보이스로
        // 삼는다. 묵음/저에너지 꼬리 윈도우가 결과 카드의 등급·점수를 낮추는 것을 방지.
        // tie는 동일 등급 내 평균 fakeScore가 높은 쪽.
        val worstAgg = steps.maxWithOrNull(
            compareBy({ it.second.threatLevel.ordinal }, { it.second.averageFakeScore })
        )?.second
        val lastAgg = worstAgg
        val finalPhishing = fullPhishing
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
