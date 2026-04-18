package com.deepvoiceguard.app

import com.deepvoiceguard.app.audio.FileAudioSource
import io.mockk.mockk
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class FileAudioSourceTest {

    private val source = FileAudioSource(mockk(relaxed = true))

    @Test
    fun `emits N full frames plus zero-padded remainder from mono 16kHz 16bit WAV`() = runTest {
        val totalSamples = 512 * 5 + 100  // 5 full frames + 100 leftover (zero-padded → 6th frame)
        val wav = buildWav(sampleRate = 16000, channels = 1, bitsPerSample = 16, sampleCount = totalSamples)
        val frames = mutableListOf<Pair<ShortArray, Int>>()

        source.streamFromInputStream(
            ByteArrayInputStream(wav),
            realtimePacing = false,
        ) { frame, readSize -> frames += frame.copyOf(readSize) to readSize }

        assertEquals("Should emit 5 full + 1 padded remainder = 6 frames", 6, frames.size)
        frames.forEach { (_, size) -> assertEquals(512, size) }
        // 마지막 frame: 앞 100은 원본(0 값), 나머지 412는 zero-pad. 모두 0이지만 size=512 유지.
        val lastFrame = frames.last().first
        assertEquals(512, lastFrame.size)
    }

    @Test
    fun `each emitted frame matches original PCM data in order`() = runTest {
        val totalSamples = 512 * 3
        val pcm = ShortArray(totalSamples) { (it * 37 % 30000).toShort() }
        val wav = buildWav(16000, 1, 16, totalSamples, pcm)
        val received = mutableListOf<Short>()

        source.streamFromInputStream(ByteArrayInputStream(wav), realtimePacing = false) { frame, size ->
            for (i in 0 until size) received += frame[i]
        }

        assertEquals(totalSamples, received.size)
        for (i in pcm.indices) assertEquals("sample $i mismatch", pcm[i], received[i])
    }

    @Test(expected = FileAudioSource.UnsupportedWavFormatException::class)
    fun `rejects stereo WAV`() = runTest {
        val wav = buildWav(sampleRate = 16000, channels = 2, bitsPerSample = 16, sampleCount = 1024)
        source.streamFromInputStream(ByteArrayInputStream(wav), realtimePacing = false) { _, _ -> }
    }

    @Test(expected = FileAudioSource.UnsupportedWavFormatException::class)
    fun `rejects 48kHz WAV`() = runTest {
        val wav = buildWav(sampleRate = 48000, channels = 1, bitsPerSample = 16, sampleCount = 1024)
        source.streamFromInputStream(ByteArrayInputStream(wav), realtimePacing = false) { _, _ -> }
    }

    @Test(expected = FileAudioSource.UnsupportedWavFormatException::class)
    fun `rejects 8-bit WAV`() = runTest {
        val wav = buildWav(sampleRate = 16000, channels = 1, bitsPerSample = 8, sampleCount = 1024)
        source.streamFromInputStream(ByteArrayInputStream(wav), realtimePacing = false) { _, _ -> }
    }

    @Test
    fun `cancellation during pacing stops emission cleanly`() = runTest {
        val totalSamples = 512 * 10
        val wav = buildWav(16000, 1, 16, totalSamples)
        var emitted = 0
        try {
            kotlinx.coroutines.withTimeout(50) {
                source.streamFromInputStream(ByteArrayInputStream(wav), realtimePacing = true) { _, _ ->
                    emitted++
                }
            }
            fail("Expected TimeoutCancellationException")
        } catch (_: kotlinx.coroutines.TimeoutCancellationException) {
            // expected — cancellation propagated through delay()
        }
        // 50ms / 32ms-per-frame → 최대 2~3 frame 정도만 emit.
        assertTrue("Expected <= 5 frames before cancel, got $emitted", emitted <= 5)
    }

    @Test
    fun `handles WAV with optional LIST chunk before data chunk`() = runTest {
        val pcm = ShortArray(512) { it.toShort() }
        val wav = buildWavWithListChunk(16000, 1, 16, pcm)
        val frames = mutableListOf<ShortArray>()
        source.streamFromInputStream(ByteArrayInputStream(wav), realtimePacing = false) { frame, size ->
            frames += frame.copyOf(size)
        }
        assertEquals(1, frames.size)
        for (i in pcm.indices) assertEquals(pcm[i], frames[0][i])
    }

    // ── WAV bytes builder ──

    private fun buildWav(
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int,
        sampleCount: Int,
        pcm: ShortArray? = null,
    ): ByteArray {
        val bytesPerSample = bitsPerSample / 8
        val dataSize = sampleCount * bytesPerSample * channels
        val totalCapacity = 44 + dataSize + 16  // safety margin
        val out = ByteBuffer.allocate(totalCapacity).order(ByteOrder.LITTLE_ENDIAN)
        out.put("RIFF".toByteArray(Charsets.US_ASCII))
        out.putInt(36 + dataSize)
        out.put("WAVE".toByteArray(Charsets.US_ASCII))
        out.put("fmt ".toByteArray(Charsets.US_ASCII))
        out.putInt(16)
        out.putShort(1)  // PCM
        out.putShort(channels.toShort())
        out.putInt(sampleRate)
        out.putInt(sampleRate * channels * bytesPerSample)
        out.putShort((channels * bytesPerSample).toShort())
        out.putShort(bitsPerSample.toShort())
        out.put("data".toByteArray(Charsets.US_ASCII))
        out.putInt(dataSize)
        when (bitsPerSample) {
            16 -> {
                if (pcm != null) for (s in pcm) out.putShort(s)
                else repeat(sampleCount) { out.putShort(0) }
            }
            8 -> repeat(sampleCount * channels) { out.put(0) }
            else -> throw IllegalArgumentException("unsupported bits: $bitsPerSample")
        }
        return out.array().copyOf(out.position())
    }

    /** fmt 청크 이후, data 청크 이전에 "LIST" 메타데이터 청크를 끼워 넣은 WAV. */
    private fun buildWavWithListChunk(
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int,
        pcm: ShortArray,
    ): ByteArray {
        val bytesPerSample = bitsPerSample / 8
        val dataSize = pcm.size * bytesPerSample * channels
        val listPayload = "INFOISFT\u0000\u0000\u0000\u0008Lavf\u0000\u0000\u0000\u0000".toByteArray(Charsets.ISO_8859_1)
        val listSize = listPayload.size
        val totalSize = 36 + 8 + listSize + 8 + dataSize
        val out = ByteBuffer.allocate(totalSize + 8).order(ByteOrder.LITTLE_ENDIAN)
        out.put("RIFF".toByteArray(Charsets.US_ASCII))
        out.putInt(totalSize)
        out.put("WAVE".toByteArray(Charsets.US_ASCII))
        out.put("fmt ".toByteArray(Charsets.US_ASCII))
        out.putInt(16)
        out.putShort(1)
        out.putShort(channels.toShort())
        out.putInt(sampleRate)
        out.putInt(sampleRate * channels * bytesPerSample)
        out.putShort((channels * bytesPerSample).toShort())
        out.putShort(bitsPerSample.toShort())
        out.put("LIST".toByteArray(Charsets.US_ASCII))
        out.putInt(listSize)
        out.put(listPayload)
        out.put("data".toByteArray(Charsets.US_ASCII))
        out.putInt(dataSize)
        for (s in pcm) out.putShort(s)
        return out.array().copyOf(out.position())
    }
}
