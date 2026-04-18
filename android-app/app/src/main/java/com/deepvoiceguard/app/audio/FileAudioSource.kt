package com.deepvoiceguard.app.audio

import android.content.Context
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive

/**
 * WAV asset을 읽어 16-bit PCM short frame을 순차 emit하는 파일 기반 오디오 소스.
 *
 * **목적**: Apple Silicon Mac + Android Studio 에뮬레이터에서 호스트 마이크 라우팅이
 * 동작하지 않아 AudioRecord에 데이터가 들어오지 않는 문제 우회. FILE mode에서 이 소스가
 * `AudioCaptureService`의 AudioRecord 루프를 대체하며, 이후 VAD → AASIST → 알림까지의
 * live 파이프라인 전체를 동일하게 통과시킨다.
 *
 * **설계 원칙**:
 *  - `AudioRecord` 경로는 건드리지 않고 AudioCaptureService의 추가 분기에서만 사용.
 *  - WAV 포맷 요구: 16-bit PCM, mono, 16000Hz. 그 외 형식은 예외.
 *  - 실시간 페이싱(`realtimePacing=true`)은 32ms/frame delay로 데모 UX 재현 — 모델 정확도와는
 *    무관하므로 테스트에서는 false로 즉시 실행 가능.
 *  - 구조적 동시성: 호출자 coroutine 취소 시 `delay()`가 CancellationException으로 복구되어
 *    while 루프가 자연 종료.
 */
class FileAudioSource(private val context: Context) {

    companion object {
        const val EXPECTED_SAMPLE_RATE = 16000
        const val FRAME_SIZE = 512  // 32ms @ 16kHz — AudioRecord/VAD와 동일 단위.
        private const val FRAME_PACING_MS = 32L
    }

    class UnsupportedWavFormatException(message: String) : Exception(message)

    /**
     * Asset 또는 절대 파일 경로에서 WAV를 스트리밍.
     *
     * @param assetPath assets 루트 기준 상대 경로 (예: "demo/demo_04.wav")
     * @param realtimePacing true면 32ms delay로 실제 녹음 속도 재현. false면 최대속.
     * @param onFrame 512 샘플 단위 ShortArray emit. 호출자가 `AudioPipeline.processAudioChunk`로 전달.
     * @throws UnsupportedWavFormatException WAV 헤더가 16-bit mono 16kHz가 아닌 경우.
     */
    suspend fun streamAsset(
        assetPath: String,
        realtimePacing: Boolean = true,
        onFrame: suspend (ShortArray, Int) -> Unit,
    ) {
        context.assets.open(assetPath).use { stream ->
            streamFromInputStream(stream, realtimePacing, onFrame)
        }
    }

    internal suspend fun streamFromInputStream(
        stream: InputStream,
        realtimePacing: Boolean,
        onFrame: suspend (ShortArray, Int) -> Unit,
    ) {
        // 파일 read를 chunk 단위로 분할하고 중간마다 ensureActive로 취소 반응. 대형 asset에서
        // STOP 지연을 줄인다. asset은 Android AssetManager 상 최대 수 MB로 알려져 있어 합리적 상한.
        val bytes = readAllWithCancellation(stream)
        coroutineContext.ensureActive()
        val (dataStart, dataSize) = findWavDataChunk(bytes)
            ?: throw UnsupportedWavFormatException("Invalid or missing WAV data chunk")
        validateWavFormat(bytes)

        val numSamples = dataSize / 2  // 16-bit
        val buffer = ByteBuffer.wrap(bytes, dataStart, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        val frame = ShortArray(FRAME_SIZE)
        var emitted = 0
        while (emitted + FRAME_SIZE <= numSamples) {
            coroutineContext.ensureActive()
            for (i in 0 until FRAME_SIZE) {
                frame[i] = buffer.short
            }
            onFrame(frame, FRAME_SIZE)
            emitted += FRAME_SIZE
            if (realtimePacing) delay(FRAME_PACING_MS)
        }
        // 마지막 잔여 샘플(<512)이 있으면 zero-pad하여 1 frame으로 emit.
        // AudioPipeline.processAudioChunk의 vadRemainder도 동일 단위로 동작하므로
        // 나머지를 그냥 흘리면 마지막 VAD onSpeechEnd가 뜨지 않아 마지막 detection 분실.
        val remainder = numSamples - emitted
        if (remainder > 0) {
            coroutineContext.ensureActive()
            for (i in 0 until remainder) {
                frame[i] = buffer.short
            }
            for (i in remainder until FRAME_SIZE) {
                frame[i] = 0
            }
            onFrame(frame, FRAME_SIZE)
        }
    }

    /**
     * InputStream을 모두 읽되 64KB 청크 단위로 ensureActive를 체크한다.
     * 대형 asset에서 STOP/취소 반응성 보장.
     */
    private suspend fun readAllWithCancellation(stream: InputStream): ByteArray {
        val chunkSize = 64 * 1024
        val buffer = ByteArray(chunkSize)
        val sink = java.io.ByteArrayOutputStream()
        while (true) {
            coroutineContext.ensureActive()
            val n = stream.read(buffer)
            if (n < 0) break
            sink.write(buffer, 0, n)
        }
        return sink.toByteArray()
    }

    /** WAV `fmt ` 청크에서 sample rate / channel / bits per sample 검증. */
    private fun validateWavFormat(bytes: ByteArray) {
        if (bytes.size < 36) throw UnsupportedWavFormatException("WAV too short")
        val fmtStart = findChunk(bytes, "fmt ")
            ?: throw UnsupportedWavFormatException("Missing 'fmt ' chunk")
        val buf = ByteBuffer.wrap(bytes, fmtStart, 16).order(ByteOrder.LITTLE_ENDIAN)
        val audioFormat = buf.short.toInt() and 0xFFFF  // 1 = PCM
        val channels = buf.short.toInt() and 0xFFFF
        val sampleRate = buf.int
        buf.int  // byte rate (skip)
        buf.short  // block align (skip)
        val bitsPerSample = buf.short.toInt() and 0xFFFF
        if (audioFormat != 1) {
            throw UnsupportedWavFormatException("Non-PCM WAV (format=$audioFormat)")
        }
        if (channels != 1) {
            throw UnsupportedWavFormatException("Expected mono, got $channels channels")
        }
        if (sampleRate != EXPECTED_SAMPLE_RATE) {
            throw UnsupportedWavFormatException(
                "Expected ${EXPECTED_SAMPLE_RATE}Hz, got ${sampleRate}Hz",
            )
        }
        if (bitsPerSample != 16) {
            throw UnsupportedWavFormatException("Expected 16-bit, got $bitsPerSample-bit")
        }
    }

    private fun findWavDataChunk(bytes: ByteArray): Pair<Int, Int>? {
        val start = findChunk(bytes, "data") ?: return null
        val size = ByteBuffer.wrap(bytes, start - 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
        val actual = minOf(size, bytes.size - start).coerceAtLeast(0)
        return start to actual
    }

    /** RIFF 청크 스캔 — "RIFF" + size + "WAVE" 이후 id 일치 청크의 data 시작 오프셋 반환. */
    private fun findChunk(bytes: ByteArray, id: String): Int? {
        if (bytes.size < 12) return null
        val riff = String(bytes, 0, 4, Charsets.US_ASCII)
        val wave = String(bytes, 8, 4, Charsets.US_ASCII)
        if (riff != "RIFF" || wave != "WAVE") return null
        var offset = 12
        while (offset + 8 <= bytes.size) {
            val chunkId = String(bytes, offset, 4, Charsets.US_ASCII)
            val size = ByteBuffer.wrap(bytes, offset + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
            if (size < 0) return null
            if (chunkId == id) return offset + 8
            offset += 8 + size + (size and 1)
        }
        return null
    }
}
