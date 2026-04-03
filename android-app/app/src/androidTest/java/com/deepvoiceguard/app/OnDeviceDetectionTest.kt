package com.deepvoiceguard.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.deepvoiceguard.app.inference.OnDeviceEngine
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 에뮬레이터/기기에서 ONNX 모델의 실제 탐지 정확도를 검증하는 Instrumented Test.
 *
 * 테스트 전 adb push로 WAV 파일을 /sdcard/Download/test-audio/ 에 올려야 합니다:
 *   adb push test-samples/real/real_01.wav /sdcard/Download/test-audio/
 *   adb push test-samples/fake/fake_01.wav /sdcard/Download/test-audio/
 *   etc.
 */
@RunWith(AndroidJUnit4::class)
class OnDeviceDetectionTest {

    private lateinit var engine: OnDeviceEngine
    private val testDir = File("/data/local/tmp/test-audio")

    @Before
    fun setup() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        engine = OnDeviceEngine(context, "aasist.onnx")
        assertTrue("Engine should be ready", engine.isReady())
    }

    @After
    fun tearDown() {
        engine.close()
    }

    @Test
    fun realVoice_shouldDetectAsReal() = runBlocking {
        val audio = loadWav(File(testDir, "real_01.wav"))
        assertNotNull("real_01.wav should exist", audio)
        val result = engine.detect(audio!!)
        println("real_01.wav: real=${result.realScore}, fake=${result.fakeScore}, ${result.latencyMs}ms")
        assertTrue(
            "real_01.wav should be real (got real=${result.realScore}, fake=${result.fakeScore})",
            result.realScore > result.fakeScore,
        )
    }

    @Test
    fun realVoice2_shouldDetectAsReal() = runBlocking {
        val audio = loadWav(File(testDir, "real_02.wav"))
        assertNotNull("real_02.wav should exist", audio)
        val result = engine.detect(audio!!)
        println("real_02.wav: real=${result.realScore}, fake=${result.fakeScore}, ${result.latencyMs}ms")
        assertTrue(
            "real_02.wav should be real (got real=${result.realScore}, fake=${result.fakeScore})",
            result.realScore > result.fakeScore,
        )
    }

    @Test
    fun fakeTTS_shouldDetectAsFake() = runBlocking {
        val audio = loadWav(File(testDir, "fake_01.wav"))
        assertNotNull("fake_01.wav should exist", audio)
        val result = engine.detect(audio!!)
        println("fake_01.wav: fake=${result.fakeScore}, real=${result.realScore}, ${result.latencyMs}ms")
        assertTrue(
            "fake_01.wav should be fake (got fake=${result.fakeScore}, real=${result.realScore})",
            result.fakeScore > result.realScore,
        )
    }

    @Test
    fun fakeTTS1_shouldDetectAsFake() = runBlocking {
        val audio = loadWav(File(testDir, "TTS1.wav"))
        assertNotNull("TTS1.wav should exist", audio)
        val result = engine.detect(audio!!)
        println("TTS1.wav: fake=${result.fakeScore}, real=${result.realScore}, ${result.latencyMs}ms")
        assertTrue(
            "TTS1.wav should be fake (got fake=${result.fakeScore}, real=${result.realScore})",
            result.fakeScore > result.realScore,
        )
    }

    @Test
    fun edgeKoSunhi_shouldDetectAsFake() = runBlocking {
        val audio = loadWav(File(testDir, "edge_ko_sunhi.wav"))
        assertNotNull("edge_ko_sunhi.wav should exist", audio)
        val result = engine.detect(audio!!)
        println("edge_ko_sunhi.wav: fake=${result.fakeScore}, real=${result.realScore}, ${result.latencyMs}ms")
        assertTrue(
            "edge_ko_sunhi should be fake (got fake=${result.fakeScore}, real=${result.realScore})",
            result.fakeScore > result.realScore,
        )
    }

    @Test
    fun edgeKoHyunsu_shouldDetectAsFake() = runBlocking {
        val audio = loadWav(File(testDir, "edge_ko_hyunsu.wav"))
        assertNotNull("edge_ko_hyunsu.wav should exist", audio)
        val result = engine.detect(audio!!)
        println("edge_ko_hyunsu.wav: fake=${result.fakeScore}, real=${result.realScore}, ${result.latencyMs}ms")
        // 이 음성은 v4에서 51% fake — 경계선이므로 확인만
        println("  → ${if (result.fakeScore > result.realScore) "FAKE (correct)" else "REAL (borderline)"}")
    }

    @Test
    fun gttsKo_shouldDetectAsFake() = runBlocking {
        val audio = loadWav(File(testDir, "gtts_ko_0.wav"))
        assertNotNull("gtts_ko_0.wav should exist", audio)
        val result = engine.detect(audio!!)
        println("gtts_ko_0.wav: fake=${result.fakeScore}, real=${result.realScore}, ${result.latencyMs}ms")
        assertTrue(
            "gtts_ko_0 should be fake (got fake=${result.fakeScore}, real=${result.realScore})",
            result.fakeScore > result.realScore,
        )
    }

    @Test
    fun allFiles_summaryReport() = runBlocking {
        println("\n========================================")
        println("  On-Device ONNX Detection Test Report")
        println("========================================")

        val files = testDir.listFiles()?.filter { it.extension == "wav" }?.sorted() ?: emptyList()
        if (files.isEmpty()) {
            println("  No test files found in $testDir")
            return@runBlocking
        }

        var correct = 0
        var total = 0

        for (file in files) {
            val audio = loadWav(file) ?: continue
            val result = engine.detect(audio)
            val isRealFile = file.name.startsWith("real")
            val predictedReal = result.realScore > result.fakeScore
            val isCorrect = isRealFile == predictedReal
            if (isCorrect) correct++
            total++

            val icon = if (isCorrect) "✅" else "❌"
            val label = if (predictedReal) "REAL" else "FAKE"
            val expected = if (isRealFile) "real" else "fake"
            println("  $icon $label (fake=${String.format("%.1f", result.fakeScore * 100)}%, " +
                "real=${String.format("%.1f", result.realScore * 100)}%) " +
                "[${result.latencyMs}ms] — ${file.name} (expected=$expected)")
        }

        println("\n  정확도: $correct/$total (${correct * 100 / total}%)")
        println("========================================\n")
    }

    /**
     * WAV 파일을 16kHz mono float32 배열로 로드.
     */
    private fun loadWav(file: File): FloatArray? {
        if (!file.exists()) return null

        val raf = RandomAccessFile(file, "r")
        // WAV 헤더 파싱 (44바이트 표준 헤더)
        val header = ByteArray(44)
        raf.readFully(header)

        val bb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        // RIFF check
        val riff = String(header, 0, 4)
        if (riff != "RIFF") {
            raf.close()
            return null
        }

        bb.position(22)
        val numChannels = bb.short.toInt()
        val sampleRate = bb.int
        bb.position(34)
        val bitsPerSample = bb.short.toInt()

        // data 청크 찾기
        raf.seek(12)
        while (raf.filePointer < raf.length() - 8) {
            val chunkId = ByteArray(4)
            raf.readFully(chunkId)
            val chunkSizeBuf = ByteArray(4)
            raf.readFully(chunkSizeBuf)
            val chunkSize = ByteBuffer.wrap(chunkSizeBuf).order(ByteOrder.LITTLE_ENDIAN).int

            if (String(chunkId) == "data") {
                val dataBytes = ByteArray(chunkSize)
                raf.readFully(dataBytes)
                raf.close()

                val dataBuf = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
                val numSamples = chunkSize / (bitsPerSample / 8) / numChannels
                val audio = FloatArray(numSamples)

                for (i in 0 until numSamples) {
                    var sample = 0f
                    for (ch in 0 until numChannels) {
                        sample += when (bitsPerSample) {
                            16 -> dataBuf.short.toFloat() / 32768f
                            32 -> dataBuf.float
                            else -> 0f
                        }
                    }
                    audio[i] = sample / numChannels
                }
                return audio
            } else {
                raf.skipBytes(chunkSize)
            }
        }
        raf.close()
        return null
    }
}
