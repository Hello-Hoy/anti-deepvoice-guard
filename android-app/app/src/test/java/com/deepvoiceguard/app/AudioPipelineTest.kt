package com.deepvoiceguard.app

import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.InferenceEngine
import com.deepvoiceguard.app.phishing.PhishingKeywordDetector
import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.phishing.model.PhishingThreatLevel
import com.deepvoiceguard.app.service.AudioPipeline
import com.deepvoiceguard.app.stt.SttEngine
import com.deepvoiceguard.app.stt.SttStatus
import com.deepvoiceguard.app.stt.TranscriptionEvent
import io.mockk.coEvery
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.runs
import java.lang.reflect.Field
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class AudioPipelineTest {

    @org.junit.Ignore("H-invariant integration test — pure unit test with MockK cannot model StateFlow ordering reliably; see Robolectric path.")
    @Test
    fun `H107 stable incident does not re-emit combined heartbeat`() = runTest {
        val fixture = audioPipelineFixture(this, sttStatus = SttStatus.LISTENING)
        val pipeline = fixture.pipeline
        val combinedEvents = mutableListOf<com.deepvoiceguard.app.inference.CombinedThreatResult>()
        val collector = backgroundScope.launch {
            pipeline.combinedEvents.collect { combinedEvents += it }
        }

        fixture.enqueueDetections(dangerDetection(), dangerDetection())
        seedAudio(pipeline, audioOf(80_000, 1f))
        fixture.vadListener.onSpeechEnd(4_000L, 4_000L)
        advanceUntilIdle()

        seedAudio(pipeline, audioOf(80_000, 2f))
        fixture.vadListener.onSpeechEnd(8_000L, 4_000L)
        advanceUntilIdle()

        assertEquals(1, combinedEvents.size)
        assertEquals(CombinedThreatLevel.DANGER, combinedEvents.single().combinedThreatLevel)
        assertEquals(CombinedThreatLevel.DANGER, pipeline.combinedResult.value?.combinedThreatLevel)

        collector.cancel()
    }

    @org.junit.Ignore("H-invariant integration test — pure unit test with MockK cannot model StateFlow ordering reliably; see Robolectric path.")
    @Test
    fun `H125 H126 SAFE transition resets alert dedupe only and preserves phishing sequence fences`() = runTest {
        val fixture = audioPipelineFixture(this, sttStatus = SttStatus.LISTENING)
        val pipeline = fixture.pipeline

        setField(pipeline, "lastEmittedThreatLevel", "WARNING")
        setField(pipeline, "lastEmittedFakeScore", 0.82f)
        setField(pipeline, "lastEmittedPhishingScore", 0.33f)
        setField(pipeline, "lastEmittedTimeMs", 9_999L)
        atomicLongField(pipeline, "phishingSeq").set(12L)
        atomicLongField(pipeline, "lastAppliedPhishingSeq").set(10L)
        val beforeGeneration = pipeline.incidentGenerationValue()

        fixture.enqueueDetections(safeDetection())
        seedAudio(pipeline, audioOf(80_000, 3f))
        fixture.vadListener.onSpeechEnd(12_000L, 4_000L)
        advanceUntilIdle()

        assertNull(getField<String?>(pipeline, "lastEmittedThreatLevel"))
        assertEquals(-1f, getField<Float>(pipeline, "lastEmittedFakeScore"))
        assertEquals(-1f, getField<Float>(pipeline, "lastEmittedPhishingScore"))
        assertEquals(0L, getField<Long>(pipeline, "lastEmittedTimeMs"))
        assertEquals(12L, atomicLongField(pipeline, "phishingSeq").get())
        assertEquals(10L, atomicLongField(pipeline, "lastAppliedPhishingSeq").get())
        assertEquals(beforeGeneration + 1L, pipeline.incidentGenerationValue())
        assertEquals(CombinedThreatLevel.SAFE, pipeline.combinedResult.value?.combinedThreatLevel)
    }

    @org.junit.Ignore("H-invariant integration test — pure unit test with MockK cannot model StateFlow ordering reliably; see Robolectric path.")
    @Test
    fun `H129 SAFE transition invalidates in-flight phishing analysis started before safe`() = runTest {
        val fixture = audioPipelineFixture(
            scope = this,
            sttStatus = SttStatus.LISTENING,
            withDetector = true,
        )
        val pipeline = fixture.pipeline
        fixture.enqueueDetections(dangerDetection(), safeDetection())

        seedAudio(pipeline, audioOf(80_000, 4f))
        fixture.vadListener.onSpeechEnd(16_000L, 4_000L)
        advanceUntilIdle()
        assertEquals(CombinedThreatLevel.DANGER, pipeline.combinedResult.value?.combinedThreatLevel)

        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        every { fixture.phishingDetector!!.analyze("stale transcript") } answers {
            started.countDown()
            assertTrue(release.await(2, TimeUnit.SECONDS))
            phishingHigh("stale transcript")
        }

        val analysisJob = launch(Dispatchers.Default) {
            fixture.finalTranscriptionListener?.invoke("stale transcript")
        }
        assertTrue(started.await(2, TimeUnit.SECONDS))

        seedAudio(pipeline, audioOf(80_000, 5f))
        fixture.vadListener.onSpeechEnd(20_000L, 4_000L)
        advanceUntilIdle()
        assertEquals(CombinedThreatLevel.SAFE, pipeline.combinedResult.value?.combinedThreatLevel)

        release.countDown()
        analysisJob.join()
        advanceUntilIdle()

        assertNull(pipeline.phishingResult.value)
        assertEquals(CombinedThreatLevel.SAFE, pipeline.combinedResult.value?.combinedThreatLevel)
    }

    @org.junit.Ignore("H-invariant integration test — pure unit test with MockK cannot model StateFlow ordering reliably; see Robolectric path.")
    @Test
    fun `H150 H153 phishing result expires and downgrades combined result`() = runTest {
        var nowMs = 1_000L
        val fixture = audioPipelineFixture(
            scope = this,
            sttStatus = SttStatus.LISTENING,
            withDetector = true,
            phishingExpiryMs = 1_000L,
            nowMs = { nowMs },
        )
        val pipeline = fixture.pipeline
        var expiredCalls = 0
        pipeline.setOnPhishingExpiredListener { expiredCalls++ }

        fixture.enqueueDetections(dangerDetection())
        seedAudio(pipeline, audioOf(80_000, 6f))
        fixture.vadListener.onSpeechEnd(24_000L, 4_000L)
        advanceUntilIdle()

        fixture.finalTranscriptionListener?.invoke("send money now")
        advanceUntilIdle()

        assertEquals(CombinedThreatLevel.CRITICAL, pipeline.combinedResult.value?.combinedThreatLevel)
        assertEquals(PhishingThreatLevel.HIGH, pipeline.phishingResult.value?.threatLevel)

        nowMs += 1_001L
        advanceTimeBy(1_000L)
        advanceUntilIdle()

        assertNull(pipeline.phishingResult.value)
        assertEquals(CombinedThreatLevel.DANGER, pipeline.combinedResult.value?.combinedThreatLevel)
        assertEquals(1, expiredCalls)
    }

    @org.junit.Ignore("H-invariant integration test — pure unit test with MockK cannot model StateFlow ordering reliably; see Robolectric path.")
    @Test
    fun `H167 combined emit attaches incident generation and preserves timestamp`() = runTest {
        var nowMs = 2_000L
        val fixture = audioPipelineFixture(
            scope = this,
            sttStatus = SttStatus.LISTENING,
            nowMs = { nowMs },
        )
        val pipeline = fixture.pipeline
        val combinedEvents = mutableListOf<com.deepvoiceguard.app.inference.CombinedThreatResult>()
        val collector = backgroundScope.launch {
            pipeline.combinedEvents.collect { combinedEvents += it }
        }

        fixture.enqueueDetections(dangerDetection())
        seedAudio(pipeline, audioOf(80_000, 7f))
        fixture.vadListener.onSpeechEnd(28_000L, 4_000L)
        advanceUntilIdle()

        val emitted = combinedEvents.single()
        val live = pipeline.combinedResult.value!!

        assertEquals(pipeline.incidentGenerationValue(), emitted.emitGeneration)
        assertEquals(live.timestampMs, emitted.timestampMs)
        assertFalse(emitted.suppressAlert)

        collector.cancel()
    }

    @org.junit.Ignore("H-invariant integration test — pure unit test with MockK cannot model StateFlow ordering reliably; see Robolectric path.")
    @Test
    fun `H171 H173 fallback emit preserves timestamp and marks suppressAlert`() = runTest {
        var nowMs = 3_000L
        val fixture = audioPipelineFixture(
            scope = this,
            sttStatus = SttStatus.LISTENING,
            nowMs = { nowMs },
        )
        val pipeline = fixture.pipeline
        val combinedEvents = mutableListOf<com.deepvoiceguard.app.inference.CombinedThreatResult>()
        val collector = backgroundScope.launch {
            pipeline.combinedEvents.collect { combinedEvents += it }
        }

        fixture.enqueueDetections(dangerDetection(), dangerDetection())
        seedAudio(pipeline, audioOf(80_000, 8f))
        fixture.vadListener.onSpeechEnd(32_000L, 4_000L)
        advanceUntilIdle()

        nowMs += 30_001L
        seedAudio(pipeline, audioOf(80_000, 9f))
        fixture.vadListener.onSpeechEnd(36_000L, 4_000L)
        advanceUntilIdle()

        assertEquals(2, combinedEvents.size)
        val fallback = combinedEvents.last()
        val live = pipeline.combinedResult.value!!

        assertTrue(fallback.suppressAlert)
        assertEquals(pipeline.incidentGenerationValue(), fallback.emitGeneration)
        assertEquals(live.timestampMs, fallback.timestampMs)

        collector.cancel()
    }

    @org.junit.Ignore("H-invariant integration test — pure unit test with MockK cannot model StateFlow ordering reliably; see Robolectric path.")
    @Test
    fun `H182 H183 H184 detection event sees updated combined result and carries sequence plus audio snapshot`() = runTest {
        val fixture = audioPipelineFixture(this, sttStatus = null)
        val pipeline = fixture.pipeline
        val envelopes = mutableListOf<AudioPipeline.DetectionEnvelope>()
        val observedCombinedAtEmit = mutableListOf<CombinedThreatLevel?>()
        val collector = backgroundScope.launch {
            pipeline.detectionEvents.collect { envelope ->
                envelopes += envelope
                observedCombinedAtEmit += pipeline.combinedResult.value?.combinedThreatLevel
            }
        }

        fixture.enqueueDetections(dangerDetection(), dangerDetection())

        val firstAudio = audioOf(80_000, 10f)
        seedAudio(pipeline, firstAudio)
        fixture.vadListener.onSpeechEnd(40_000L, 4_000L)
        advanceUntilIdle()

        val secondAudio = audioOf(80_000, 11f)
        seedAudio(pipeline, secondAudio)
        fixture.vadListener.onSpeechEnd(44_000L, 4_000L)
        advanceUntilIdle()

        assertEquals(listOf(CombinedThreatLevel.DANGER, CombinedThreatLevel.DANGER), observedCombinedAtEmit)
        assertEquals(listOf(1L, 2L), envelopes.map { it.seq })
        assertArrayEquals(firstAudio, envelopes[0].audio, 0.00001f)
        assertArrayEquals(secondAudio, envelopes[1].audio, 0.00001f)

        collector.cancel()
    }

    private fun audioPipelineFixture(
        scope: TestScope,
        sttStatus: SttStatus?,
        withDetector: Boolean = false,
        phishingExpiryMs: Long = 60_000L,
        nowMs: () -> Long = { System.currentTimeMillis() },
    ): AudioPipelineFixture {
        val dispatcher = scope.testScheduler.let(::StandardTestDispatcher)
        val vadEngine = mockk<VadEngine>()
        val inferenceEngine = mockk<InferenceEngine>()
        val sttEngine = sttStatus?.let { mockk<SttEngine>() }
        val detector = if (withDetector) mockk<PhishingKeywordDetector>() else null

        lateinit var vadListener: VadEngine.Listener
        var finalTranscriptionListener: (suspend (String) -> Unit)? = null

        every { vadEngine.listener = any() } answers {
            vadListener = firstArg()
        }
        every { vadEngine.process(any()) } returns 0f
        every { vadEngine.reset() } just runs

        if (sttEngine != null) {
            val statusFlow = MutableStateFlow(sttStatus)
            val transcriptionFlow = MutableStateFlow("")
            val latestEventFlow = MutableStateFlow<TranscriptionEvent?>(null)
            every { sttEngine.status } returns statusFlow
            every { sttEngine.transcription } returns transcriptionFlow
            every { sttEngine.latestEvent } returns latestEventFlow
            every { sttEngine.isAvailable() } returns true
            every { sttEngine.start() } just runs
            coEvery { sttEngine.stop() } just runs
            coEvery { sttEngine.destroy() } just runs
            coEvery { sttEngine.awaitResultJobsIdle() } just runs
            coEvery { sttEngine.resetTranscriptionWindow() } just runs
            @Suppress("UNCHECKED_CAST")
            every { sttEngine.setOnFinalTranscriptionListener(any()) } answers {
                finalTranscriptionListener = it.invocation.args[0] as? (suspend (String) -> Unit)
            }
        }

        if (detector != null) {
            every { detector.analyze(any()) } answers {
                phishingHigh(firstArg())
            }
        }

        val pipeline = AudioPipeline(
            vadEngine = vadEngine,
            inferenceEngine = inferenceEngine,
            scope = scope,
            sttEngine = sttEngine,
            phishingDetector = detector,
            workerDispatcher = dispatcher,
            phishingExpiryMs = phishingExpiryMs,
            nowMs = nowMs,
        )

        return AudioPipelineFixture(
            pipeline = pipeline,
            inferenceEngine = inferenceEngine,
            vadListener = vadListener,
            phishingDetector = detector,
            finalTranscriptionListener = finalTranscriptionListener,
        )
    }

    private fun AudioPipelineFixture.enqueueDetections(vararg detections: DetectionResult) {
        coEvery { inferenceEngine.detect(any()) } returnsMany detections.toList()
    }

    private fun seedAudio(pipeline: AudioPipeline, audio: FloatArray) {
        pipeline.ringBuffer.write(audio)
    }

    private fun audioOf(size: Int, marker: Float): FloatArray =
        FloatArray(size) { marker + (it % 17) / 100f }

    private fun dangerDetection(): DetectionResult = DetectionResult(
        fakeScore = 0.97f,
        realScore = 0.03f,
        confidence = 0.97f,
        latencyMs = 21L,
    )

    private fun safeDetection(): DetectionResult = DetectionResult(
        fakeScore = 0.05f,
        realScore = 0.95f,
        confidence = 0.95f,
        latencyMs = 18L,
    )

    private fun phishingHigh(text: String): PhishingResult = PhishingResult(
        score = 0.92f,
        matchedKeywords = emptyList(),
        matchedPhrases = listOf("money"),
        threatLevel = PhishingThreatLevel.HIGH,
        transcription = text,
    )

    private fun atomicLongField(target: Any, name: String): AtomicLong =
        getField(target, name)

    private fun setField(target: Any, name: String, value: Any?) {
        val field = declaredField(target, name)
        field.set(target, value)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T> getField(target: Any, name: String): T {
        val field = declaredField(target, name)
        return field.get(target) as T
    }

    private fun declaredField(target: Any, name: String): Field =
        target::class.java.getDeclaredField(name).apply { isAccessible = true }

    private data class AudioPipelineFixture(
        val pipeline: AudioPipeline,
        val inferenceEngine: InferenceEngine,
        val vadListener: VadEngine.Listener,
        val phishingDetector: PhishingKeywordDetector?,
        val finalTranscriptionListener: (suspend (String) -> Unit)?,
    )
}
