package com.deepvoiceguard.app

import com.deepvoiceguard.app.audio.NarrowbandPreprocessor
import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.InferenceEngine
import com.deepvoiceguard.app.inference.ModelInfo
import com.deepvoiceguard.app.service.AudioPipeline
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.runs
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * AudioPipeline + NarrowbandPreprocessor 통합 회귀 테스트 (deterministic).
 *
 * 라이브 AASIST ONNX 추론은 JVM 단위 테스트 classpath에 없어 실제 스코어 비교 대신
 * "필터가 iPhone-wideband 분포 이동 신호를 narrowband로 확실히 수렴시킨다"는 특성만 검증.
 * 실제 스코어 검증은 tools/score_demo_samples.py + 에뮬레이터/실기기 시연으로 수행.
 *
 * 시뮬레이션 신호 구성:
 *  - base: 1 kHz 음성 대역 tone (AASIST 학습 분포 안)
 *  - wideband leak: 4 kHz + 6 kHz tone + broadband noise (iPhone m4a AAC 특성 시뮬)
 *
 * 검증 invariant:
 *  (A) 필터 입력에는 ≥3 kHz 대역 에너지가 많음
 *  (B) NarrowbandPreprocessor 통과 후 ≥3 kHz 에너지가 ≥ 40 dB 감쇠
 *  (C) 1 kHz 대역 (음성 정보)은 거의 보존
 */
class AudioPipelinePreprocessorIntegrationTest {

    @Test
    fun `preprocessor attenuates above 3kHz while preserving speech band`() {
        val pre = NarrowbandPreprocessor()
        val signal = simulateLeakySignal(durationSec = 2.0)

        // 고주파 측정은 필터 cutoff(3kHz)에서 충분히 떨어진 5kHz 이상 대역으로.
        // cutoff 바로 위(3.5kHz)는 -6dB 근처라 전체 대역 평균에 희석됨.
        val originalHighBand = bandEnergyDb(signal, minHz = 5000.0, maxHz = 7500.0)
        val originalSpeechBand = bandEnergyDb(signal, minHz = 800.0, maxHz = 1200.0)
        assertTrue(
            "Simulated signal must carry measurable >5 kHz energy (check simulation setup)",
            originalHighBand > -40.0,
        )

        // 필터 후 — filterOnly 사용하여 RMS 정규화 영향 배제 (필터 자체 효과만 측정).
        // Kotlin internal 함수는 클래스 파일에서 `filterOnly$app_debug`로 name-mangling됨.
        val filterOnlyMethod = pre::class.java.declaredMethods
            .first { it.name.startsWith("filterOnly") }
            .apply { isAccessible = true }
        val filtered = filterOnlyMethod.invoke(pre, signal) as FloatArray

        val filteredHighBand = bandEnergyDb(filtered, minHz = 5000.0, maxHz = 7500.0)
        val filteredSpeechBand = bandEnergyDb(filtered, minHz = 800.0, maxHz = 1200.0)

        val highAttenuation = originalHighBand - filteredHighBand
        val speechAttenuation = originalSpeechBand - filteredSpeechBand

        // (B) 3.5 kHz 이상은 40 dB 이상 감쇠해야 함 (scipy 기준 5kHz -115dB, 7kHz -200dB).
        assertTrue(
            "High-band attenuation must be >=40 dB, got $highAttenuation dB",
            highAttenuation >= 40.0,
        )
        // (C) 1 kHz 음성 정보는 거의 보존 (± 2 dB 이내).
        assertTrue(
            "Speech-band change must be <=2 dB, got $speechAttenuation dB",
            kotlin.math.abs(speechAttenuation) <= 2.0,
        )
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun `AudioPipeline with preprocessor sends filtered segment to InferenceEngine detect`() = runTest {
        // 실제 AudioPipeline에 preprocessor를 주입하고, updateCombinedResult로 detect를 유도해
        // InferenceEngine.detect가 받는 byte sequence가 원본과 다른지 (=필터가 적용됐는지) 확인.
        val capturedBySpy = AtomicReference<FloatArray?>(null)
        val vad = mockk<VadEngine>(relaxed = true)
        every { vad.listener = any() } just runs

        val spyEngine = object : InferenceEngine {
            override suspend fun detect(audio: FloatArray): DetectionResult {
                capturedBySpy.set(audio.copyOf())
                return DetectionResult(
                    fakeScore = 0.1f, realScore = 0.9f, confidence = 0.9f, latencyMs = 1L,
                )
            }
            override fun isReady(): Boolean = true
            override fun getModelInfo(): ModelInfo = ModelInfo("spy", "1", "test")
            override fun close() {}
        }

        val preprocessor = NarrowbandPreprocessor()
        val dispatcher = StandardTestDispatcher(testScheduler)
        val pipeline = AudioPipeline(
            vadEngine = vad,
            inferenceEngine = spyEngine,
            scope = backgroundScope,
            sttEngine = null,
            phishingDetector = null,
            workerDispatcher = dispatcher,
            narrowbandPreprocessor = preprocessor,
        )

        // ring buffer에 원본 신호 주입 (필터되지 않은 wideband). 정확히 1 AASIST 세그먼트 크기.
        val originalSignal = simulateLeakySignal(durationSec = 64600.0 / 16_000.0)
        pipeline.ringBuffer.write(originalSignal)

        // analyzeSegments는 private이지만 VAD onSpeechEnd를 통해 간접 호출됨.
        // 직접 호출 가능하도록 테스트 helper reflection으로 접근.
        val analyze = AudioPipeline::class.java.getDeclaredMethod(
            "analyzeSegments", Long::class.java, kotlin.coroutines.Continuation::class.java,
        )
        analyze.isAccessible = true
        val cont = kotlinx.coroutines.CompletableDeferred<Any?>()
        val result = analyze.invoke(pipeline, 4_100L, object : kotlin.coroutines.Continuation<Any?> {
            override val context = kotlin.coroutines.EmptyCoroutineContext
            override fun resumeWith(r: Result<Any?>) {
                if (r.isSuccess) cont.complete(r.getOrNull()) else cont.completeExceptionally(r.exceptionOrNull()!!)
            }
        })
        if (result !== kotlin.coroutines.intrinsics.COROUTINE_SUSPENDED) {
            cont.complete(result)
        }
        advanceUntilIdle()
        // analyzeSegments 내부 실패/취소 검출.
        cont.await()

        val received = capturedBySpy.get()
        assertNotNull("InferenceEngine.detect must have been called", received)
        // detect가 받은 신호는 원본과 달라야 한다 — 필터 + normalize가 적용된 결과.
        // 원본의 5kHz+ 대역이 크게 줄어들고, RMS도 변경된다.
        val originalHigh = bandEnergyDb(originalSignal.copyOf(64600), minHz = 5000.0, maxHz = 7500.0)
        val receivedHigh = bandEnergyDb(received!!, minHz = 5000.0, maxHz = 7500.0)
        assertTrue(
            "Pipeline with preprocessor must strip >5kHz band by >=20 dB in actual detect call " +
                "(original=$originalHigh dB, received=$receivedHigh dB)",
            (originalHigh - receivedHigh) >= 20.0,
        )
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun `audioSnapshot stays raw while detect input is filtered`() = runTest {
        // 핵심 invariant: preprocessor 활성 시에도 ring buffer 저장 / audioSnapshot / VAD input 은
        // **원본 PCM**을 유지한다. 오직 inferenceEngine.detect 입력만 필터링되어야 함.
        val vad = mockk<VadEngine>(relaxed = true)
        every { vad.listener = any() } just runs
        val capturedDetectInput = AtomicReference<FloatArray?>(null)
        val detectSpy = object : InferenceEngine {
            override suspend fun detect(audio: FloatArray): DetectionResult {
                capturedDetectInput.set(audio.copyOf())
                return DetectionResult(0.1f, 0.9f, 0.9f, 1L)
            }
            override fun isReady() = true
            override fun getModelInfo() = ModelInfo("spy", "1", "test")
            override fun close() {}
        }

        val dispatcher = StandardTestDispatcher(testScheduler)
        val pipeline = AudioPipeline(
            vadEngine = vad,
            inferenceEngine = detectSpy,
            scope = backgroundScope,
            workerDispatcher = dispatcher,
            narrowbandPreprocessor = NarrowbandPreprocessor(),
        )

        // 4.04s 정확히 (64600 samples = AASIST NB_SAMP) → analyzeSegments가 detect를 딱 한 번 호출.
        val rawSignal = simulateLeakySignal(durationSec = 64600.0 / 16_000.0)
        pipeline.ringBuffer.write(rawSignal)
        val ringSnapshotBefore = pipeline.ringBuffer.getLast(64600)

        // reflection으로 analyzeSegments 호출 (driving path)
        val analyze = AudioPipeline::class.java.getDeclaredMethod(
            "analyzeSegments", Long::class.java, kotlin.coroutines.Continuation::class.java,
        ).apply { isAccessible = true }
        val cont = kotlinx.coroutines.CompletableDeferred<Any?>()
        val r = analyze.invoke(pipeline, 4_100L, object : kotlin.coroutines.Continuation<Any?> {
            override val context = kotlin.coroutines.EmptyCoroutineContext
            override fun resumeWith(r: Result<Any?>) {
                if (r.isSuccess) cont.complete(r.getOrNull()) else cont.completeExceptionally(r.exceptionOrNull()!!)
            }
        })
        if (r !== kotlin.coroutines.intrinsics.COROUTINE_SUSPENDED) cont.complete(r)
        advanceUntilIdle()
        // analyzeSegments 내부 실패/취소 검출 — await로 예외를 위로 전파.
        cont.await()

        // invariant 1: detect가 받은 신호는 고주파 크게 감쇠 (필터링됨)
        val detectInput = capturedDetectInput.get()
        assertNotNull(detectInput)
        val rawHigh = bandEnergyDb(rawSignal.copyOf(64600), minHz = 5000.0, maxHz = 7500.0)
        val detectHigh = bandEnergyDb(detectInput!!, minHz = 5000.0, maxHz = 7500.0)
        assertTrue(
            "detect input must have >=20dB attenuation in high band",
            (rawHigh - detectHigh) >= 20.0,
        )

        // invariant 2: ring buffer 내용은 analyzeSegments 전후로 **변하지 않아야** 함.
        val ringSnapshotAfter = pipeline.ringBuffer.getLast(64600)
        assertArrayEquals(
            "ring buffer must remain raw (unfiltered) after analyzeSegments",
            ringSnapshotBefore,
            ringSnapshotAfter,
            1e-6f,
        )

        // invariant 3: _combinedResult.audioSnapshot 도 원본 PCM이어야 — 필터링된 detect input
        // 과는 달라야 함. STT 없음 분기(analyzeSegments Else)에서도 반드시 _combinedResult가
        // audioSnapshot과 함께 publish되어야 하므로 **무조건** 검증한다. silently skip 금지.
        val combinedResult = pipeline.combinedResult.value
        assertNotNull("combinedResult must be published by analyzeSegments", combinedResult)
        val combinedAudioSnapshot = combinedResult!!.audioSnapshot
        assertTrue(
            "combinedResult.audioSnapshot must be non-empty (regression guard)",
            combinedAudioSnapshot.isNotEmpty(),
        )
        val snapshotHigh = bandEnergyDb(combinedAudioSnapshot, minHz = 5000.0, maxHz = 7500.0)
        assertTrue(
            "combinedResult.audioSnapshot must be raw (close to rawHigh, NOT filtered). " +
                "raw=$rawHigh dB, snapshot=$snapshotHigh dB, detect=$detectHigh dB",
            kotlin.math.abs(snapshotHigh - rawHigh) < 10.0,
        )
    }

    @Test
    fun `full pipeline preprocessor application is idempotent-safe on already-filtered signal`() {
        // 이미 필터 통과한 신호에 다시 process를 돌려도 RMS가 ~15% 이내로 안정해야 한다.
        // (이중 필터링이 Demo WAV 경로에 치명적 영향을 주지 않음을 보장 — 아키텍처 결정 B)
        val pre = NarrowbandPreprocessor()
        val base = simulateLeakySignal(durationSec = 1.5)
        val once = pre.process(base)
        val twice = pre.process(once)

        val rmsOnce = rms(once)
        val rmsTwice = rms(twice)
        val relativeChange = kotlin.math.abs(rmsTwice - rmsOnce) / rmsOnce
        assertTrue(
            "Twice-filtered RMS must stay within 15% of once-filtered (got ${relativeChange * 100}%)",
            relativeChange < 0.15,
        )
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private fun simulateLeakySignal(durationSec: Double): FloatArray {
        val sr = 16_000
        val count = (durationSec * sr).toInt()
        val out = FloatArray(count)
        val baseAmp = 0.3f   // 1 kHz speech tone
        val leakAmp4k = 0.25f  // simulated AAC shaping leak
        val leakAmp6k = 0.20f
        val rng = java.util.Random(42)
        for (i in 0 until count) {
            val t = i.toDouble() / sr
            val base = baseAmp * sin(2.0 * PI * 1000.0 * t).toFloat()
            val h4 = leakAmp4k * sin(2.0 * PI * 4000.0 * t).toFloat()
            val h6 = leakAmp6k * cos(2.0 * PI * 6000.0 * t).toFloat()
            // pink-ish broadband noise (low-amp)
            val noise = (rng.nextGaussian() * 0.05).toFloat()
            out[i] = (base + h4 + h6 + noise)
                .coerceIn(-1f, 1f)
        }
        return out
    }

    /**
     * 지정 대역(minHz ~ maxHz) 범위 RMS를 dB로 측정.
     * 4차 Butterworth 대역통과 biquad를 통과시켜 해당 대역 에너지만 남긴 뒤 RMS 계산.
     */
    private fun bandEnergyDb(signal: FloatArray, minHz: Double, maxHz: Double = 7800.0): Double {
        val sr = 16_000.0
        // 바이쿼드 대역통과를 수동 설계 (RBJ cookbook). 4차(=2개 biquad)는 단순 반복 적용.
        val centerHz = sqrt(minHz * maxHz)
        val q = centerHz / (maxHz - minHz).coerceAtLeast(1.0)
        val filtered = applyBandpassBiquad(signal, sr, centerHz, q)
        val filtered2 = applyBandpassBiquad(filtered, sr, centerHz, q)
        return 20.0 * log10(rms(filtered2) + 1e-12)
    }

    private fun applyBandpassBiquad(
        signal: FloatArray,
        sr: Double,
        centerHz: Double,
        q: Double,
    ): FloatArray {
        val omega = 2.0 * PI * centerHz / sr
        val alpha = sin(omega) / (2.0 * q)
        val cosw = cos(omega)
        // RBJ band-pass (constant 0 dB peak gain)
        val b0 = alpha
        val b1 = 0.0
        val b2 = -alpha
        val a0 = 1.0 + alpha
        val a1 = -2.0 * cosw
        val a2 = 1.0 - alpha
        val nb0 = b0 / a0
        val nb1 = b1 / a0
        val nb2 = b2 / a0
        val na1 = a1 / a0
        val na2 = a2 / a0

        val out = FloatArray(signal.size)
        var x1 = 0.0
        var x2 = 0.0
        var y1 = 0.0
        var y2 = 0.0
        for (i in signal.indices) {
            val x = signal[i].toDouble()
            val y = nb0 * x + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2
            out[i] = y.toFloat()
            x2 = x1; x1 = x; y2 = y1; y1 = y
        }
        return out
    }

    private fun rms(x: FloatArray): Double {
        if (x.isEmpty()) return 0.0
        var s = 0.0
        for (v in x) s += v.toDouble() * v.toDouble()
        return sqrt(s / x.size)
    }
}
