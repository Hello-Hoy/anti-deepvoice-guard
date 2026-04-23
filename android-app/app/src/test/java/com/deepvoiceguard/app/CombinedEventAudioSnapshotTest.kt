package com.deepvoiceguard.app

import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.CombinedThreatResult
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.InferenceEngine
import com.deepvoiceguard.app.service.AudioPipeline
import com.deepvoiceguard.app.stt.SttEngine
import com.deepvoiceguard.app.stt.SttStatus
import io.mockk.coEvery
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.runs
import java.lang.reflect.Method
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * **H185/H186 회귀 테스트**
 *
 * 실제 AudioPipeline을 구동하여 다음 invariant을 검증:
 *  (1) emit 시점에 ring buffer에 있던 오디오가 _combinedEvents의 audioSnapshot에 담긴다.
 *  (2) emit 이후 ring buffer가 전진해도 해당 event의 audioSnapshot은 변하지 않는다.
 *  (3) _combinedResult.value도 매 update 시점의 audioSnapshot을 원자적으로 포함한다.
 *
 * Codex adversarial review HIGH #1에 대한 직접 방어 — 과거 collector가 DB 저장 시점에
 * ringBuffer.getLast(80_000)을 재호출해 발생하던 evidence/metadata 불일치를 차단한다.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class CombinedEventAudioSnapshotTest {

    @Test
    fun `updateCombinedResult sets _combinedResult value with buffer snapshot at call time`() = runTest {
        val fixture = pipelineFixture(this, backgroundScope)
        val pipeline = fixture.pipeline

        val markerA = FloatArray(80_000) { 0.25f }
        pipeline.ringBuffer.write(markerA)

        // 2-consecutive DANGER 정책: 첫 add는 CAUTION, 두번째 add가 DANGER 로 승격.
        fixture.aggregator.add(dangerDetectionResult())
        val dangerDetection = fixture.aggregator.add(dangerDetectionResult())
        invokeUpdateCombinedResult(pipeline, dangerDetection)
        advanceUntilIdle()

        val live = pipeline.combinedResult.value
        assertNotNull("combinedResult must be populated after updateCombinedResult", live)
        assertEquals(CombinedThreatLevel.DANGER, live!!.combinedThreatLevel)
        assertEquals(80_000, live.audioSnapshot.size)
        assertArrayEquals(
            "combinedResult.audioSnapshot must equal ring buffer contents at update time",
            markerA,
            live.audioSnapshot,
            0f,
        )

        // Buffer를 전진시켜도 이미 저장된 snapshot은 변하지 않아야 한다.
        val markerB = FloatArray(80_000) { -0.75f }
        pipeline.ringBuffer.write(markerB)
        val liveAfterAdvance = pipeline.combinedResult.value!!
        assertArrayEquals(
            "Snapshot inside previously-captured _combinedResult must not change when buffer advances",
            markerA,
            liveAfterAdvance.audioSnapshot,
            0f,
        )
    }

    @Test
    fun `second update with different buffer produces a different combinedResult audio`() = runTest {
        // 두 번의 연속 update_CombinedResult가 각자 다른 buffer snapshot을 남기는지 확인 —
        // StateFlow 기반 invariant. SharedFlow collector 타이밍과 무관하게 테스트 가능.
        val fixture = pipelineFixture(this, backgroundScope)
        val pipeline = fixture.pipeline

        val audioA = FloatArray(80_000) { 0.3f }
        pipeline.ringBuffer.write(audioA)
        // 2-consecutive DANGER 정책 — warm-up 1회 후 실제 DANGER emit.
        fixture.aggregator.add(dangerDetectionResult())
        val dangerA = fixture.aggregator.add(dangerDetectionResult())
        invokeUpdateCombinedResult(pipeline, dangerA)
        advanceUntilIdle()
        val snapshotA = pipeline.combinedResult.value!!.audioSnapshot.copyOf()
        assertArrayEquals("First update captured audioA", audioA, snapshotA, 0f)

        // Ring buffer 전진 + 새 detection result — 두번째 emit은 audioB 기반 snapshot을 가져야 한다.
        val audioB = FloatArray(80_000) { -0.6f }
        pipeline.ringBuffer.write(audioB)
        val dangerB = fixture.aggregator.add(
            DetectionResult(fakeScore = 0.99f, realScore = 0.01f, confidence = 0.99f, latencyMs = 22L)
        )
        invokeUpdateCombinedResult(pipeline, dangerB)
        advanceUntilIdle()
        val snapshotB = pipeline.combinedResult.value!!.audioSnapshot
        assertArrayEquals("Second update captured audioB", audioB, snapshotB, 0f)
        // snapshotA는 원본 capture가 그대로 유지 (copy 방어적 — 실제 production에서는 다른 instance).
        assertArrayEquals("Previously captured snapshotA unchanged", audioA, snapshotA, 0f)
    }

    @Test
    fun `combinedEvents emit delivers audioSnapshot captured at emit time`() = runTest {
        val fixture = pipelineFixture(this, backgroundScope)
        val pipeline = fixture.pipeline

        val emitTimeAudio = FloatArray(80_000) { 0.42f }
        pipeline.ringBuffer.write(emitTimeAudio)

        val firstEvent = kotlinx.coroutines.CompletableDeferred<CombinedThreatResult>()
        backgroundScope.launch(start = kotlinx.coroutines.CoroutineStart.UNDISPATCHED) {
            pipeline.combinedEvents.collect {
                if (!firstEvent.isCompleted) firstEvent.complete(it)
            }
        }

        // 2-consecutive DANGER 정책 — warm-up 1회 후 실제 DANGER emit.
        fixture.aggregator.add(dangerDetectionResult())
        val dangerDetection = fixture.aggregator.add(dangerDetectionResult())
        invokeUpdateCombinedResult(pipeline, dangerDetection)
        advanceUntilIdle()

        // Emit 후 buffer가 다른 데이터로 바뀌어도 이미 전달된 event snapshot은 변하지 않는다.
        pipeline.ringBuffer.write(FloatArray(80_000) { -0.9f })
        advanceUntilIdle()

        val received = firstEvent.await()
        assertEquals(CombinedThreatLevel.DANGER, received.combinedThreatLevel)
        assertArrayEquals(
            "combinedEvents emit must deliver the buffer snapshot from emit time",
            emitTimeAudio,
            received.audioSnapshot,
            0f,
        )
    }

    @Test
    fun `safe transition does not emit combined event with elevated audio snapshot`() = runTest {
        val fixture = pipelineFixture(this, backgroundScope)
        val pipeline = fixture.pipeline
        pipeline.ringBuffer.write(FloatArray(80_000) { 0.1f })

        val received = mutableListOf<CombinedThreatResult>()
        val collector = backgroundScope.launch {
            pipeline.combinedEvents.collect { received += it }
        }

        val safeDetection = fixture.aggregator.add(safeDetectionResult())
        invokeUpdateCombinedResult(pipeline, safeDetection)
        advanceUntilIdle()

        assertTrue("SAFE transition must not emit combined event", received.isEmpty())
        assertEquals(
            CombinedThreatLevel.SAFE,
            pipeline.combinedResult.value?.combinedThreatLevel,
        )

        collector.cancel()
    }

    private fun pipelineFixture(
        scope: TestScope,
        backgroundScope: kotlinx.coroutines.CoroutineScope,
    ): PipelineFixture {
        val vadEngine = mockk<VadEngine>(relaxed = true)
        val inferenceEngine = mockk<InferenceEngine>(relaxed = true)
        coEvery { inferenceEngine.detect(any()) } returns dangerDetectionResult()

        val statusFlow = MutableStateFlow(SttStatus.LISTENING)
        val transcriptionFlow = MutableStateFlow("")
        val sttEngine = mockk<SttEngine>(relaxed = true)
        every { sttEngine.status } returns statusFlow
        every { sttEngine.transcription } returns transcriptionFlow
        every { sttEngine.isAvailable() } returns true
        every { sttEngine.start() } just runs

        val dispatcher = StandardTestDispatcher(scope.testScheduler)
        // pipeline의 collector Job은 test 종료 시 자동 취소되도록 backgroundScope 사용.
        val pipeline = AudioPipeline(
            vadEngine = vadEngine,
            inferenceEngine = inferenceEngine,
            scope = backgroundScope,
            sttEngine = sttEngine,
            phishingDetector = null,
            workerDispatcher = dispatcher,
        )
        val aggregator = com.deepvoiceguard.app.inference.DetectionAggregator()
        return PipelineFixture(pipeline, aggregator)
    }

    private suspend fun invokeUpdateCombinedResult(
        pipeline: AudioPipeline,
        aggregated: com.deepvoiceguard.app.inference.AggregatedResult,
    ) {
        // @VisibleForTesting internal suspend fun — 직접 호출.
        pipeline.updateCombinedResult(aggregated)
    }

    private fun dangerDetectionResult(): DetectionResult = DetectionResult(
        fakeScore = 0.97f,
        realScore = 0.03f,
        confidence = 0.97f,
        latencyMs = 21L,
    )

    private fun safeDetectionResult(): DetectionResult = DetectionResult(
        fakeScore = 0.05f,
        realScore = 0.95f,
        confidence = 0.95f,
        latencyMs = 18L,
    )

    private data class PipelineFixture(
        val pipeline: AudioPipeline,
        val aggregator: com.deepvoiceguard.app.inference.DetectionAggregator,
    )
}
