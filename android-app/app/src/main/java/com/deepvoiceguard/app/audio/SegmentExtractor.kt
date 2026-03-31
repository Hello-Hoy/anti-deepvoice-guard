package com.deepvoiceguard.app.audio

/**
 * VAD 이벤트를 받아 RingBuffer에서 오디오 세그먼트를 추출한다.
 * AASIST 모델의 입력 크기(80,000 samples = 5초 @ 16kHz)에 맞게 패딩/분할한다.
 */
class SegmentExtractor(
    private val ringBuffer: RingBuffer,
    private val sampleRate: Int = 16000,
) {
    companion object {
        const val MODEL_INPUT_SAMPLES = 80_000  // 5초 @ 16kHz
        const val MIN_DURATION_MS = 1000L       // 최소 1초
        const val MAX_DURATION_MS = 10000L      // 최대 10초
        const val WINDOW_SAMPLES = 80_000       // 5초 윈도우
        const val OVERLAP_SAMPLES = 40_000      // 2.5초 오버랩
        const val PADDING_SAMPLES = 8_000       // 0.5초 전후 패딩
    }

    /**
     * 음성 종료 시점에서 세그먼트를 추출한다.
     *
     * @param durationMs 음성 구간의 길이 (ms)
     * @return 모델 입력 크기(80,000)로 맞춰진 세그먼트 리스트
     */
    fun extract(durationMs: Long): List<FloatArray> {
        val clampedDuration = durationMs.coerceIn(MIN_DURATION_MS, MAX_DURATION_MS)
        val durationSamples = ((clampedDuration * sampleRate) / 1000).toInt()
        val totalSamples = durationSamples + 2 * PADDING_SAMPLES

        // RingBuffer에서 패딩 포함한 전체 구간 추출
        val rawSegment = ringBuffer.getLast(totalSamples)
        if (rawSegment.isEmpty()) return emptyList()

        // 5초 이하 → 단일 세그먼트, 패딩/크롭하여 반환
        if (rawSegment.size <= MODEL_INPUT_SAMPLES) {
            return listOf(padOrCrop(rawSegment))
        }

        // 5초 초과 → 윈도우 분할 (2.5초 오버랩)
        val segments = mutableListOf<FloatArray>()
        var offset = 0
        while (offset < rawSegment.size) {
            val end = minOf(offset + WINDOW_SAMPLES, rawSegment.size)
            val window = rawSegment.copyOfRange(offset, end)
            segments.add(padOrCrop(window))
            offset += WINDOW_SAMPLES - OVERLAP_SAMPLES
            if (end == rawSegment.size) break
        }
        return segments
    }

    private fun padOrCrop(audio: FloatArray): FloatArray {
        if (audio.isEmpty()) return FloatArray(MODEL_INPUT_SAMPLES)
        if (audio.size >= MODEL_INPUT_SAMPLES) {
            return audio.copyOfRange(0, MODEL_INPUT_SAMPLES)
        }
        val result = FloatArray(MODEL_INPUT_SAMPLES)
        var written = 0
        while (written < MODEL_INPUT_SAMPLES) {
            val toCopy = minOf(audio.size, MODEL_INPUT_SAMPLES - written)
            audio.copyInto(result, written, 0, toCopy)
            written += toCopy
        }
        return result
    }
}
