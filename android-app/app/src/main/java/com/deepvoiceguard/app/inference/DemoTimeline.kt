package com.deepvoiceguard.app.inference

/** 재생 시점별 5지표 스냅샷 (100ms 격자). */
data class DemoFrame(
    val offsetMs: Int,
    val vadActive: Boolean,
    val fakeScore: Float,
    val deepfakeLevel: ThreatLevel,
    val phishingScore: Float,
    val threatLevel: CombinedThreatLevel,
    val transcriptChars: Int,
)

/** 데모 타임라인: 재생 동기 replay용. */
data class DemoTimeline(
    val durationMs: Int,
    val frames: List<DemoFrame>,
    val transcript: String,
    val vadAvailable: Boolean,
    val finalResult: DemoResult,
) {
    /** 재생 위치(ms)에 해당하는 프레임(없으면 null). */
    fun frameAt(offsetMs: Int): DemoFrame? {
        if (frames.isEmpty()) return null
        val idx = (offsetMs / DemoTimelineMath.FRAME_MS).coerceIn(0, frames.size - 1)
        return frames[idx]
    }
}

/** 순수 파생 로직 — Android 의존 없이 단위테스트 가능. */
object DemoTimelineMath {
    const val FRAME_MS = 100

    fun vadActiveAt(probs: FloatArray, frameMs: Int, offsetMs: Int, threshold: Float): Boolean {
        if (probs.isEmpty() || frameMs <= 0) return false
        val idx = offsetMs / frameMs
        return idx in probs.indices && probs[idx] > threshold
    }

    /** time<=offset 중 가장 마지막 step의 인덱스. 없으면 -1. (stepTimesMs 오름차순) */
    fun stepHoldIndexAt(stepTimesMs: IntArray, offsetMs: Int): Int {
        var result = -1
        for (i in stepTimesMs.indices) {
            if (stepTimesMs[i] <= offsetMs) result = i else break
        }
        return result
    }

    fun transcriptCharsAt(length: Int, offsetMs: Int, durationMs: Int): Int {
        if (durationMs <= 0) return 0
        val ratio = (offsetMs.toFloat() / durationMs).coerceIn(0f, 1f)
        return (length * ratio).toInt().coerceIn(0, length)
    }

    /** audio에서 endSample에 끝나는 length 윈도우. 모자라면 왼쪽 0 패딩. */
    fun windowEndingAt(audio: FloatArray, endSample: Int, length: Int): FloatArray {
        val out = FloatArray(length)
        val end = endSample.coerceIn(0, audio.size)
        val start = (end - length).coerceAtLeast(0)
        val count = end - start
        if (count > 0) audio.copyInto(out, destinationOffset = length - count, startIndex = start, endIndex = end)
        return out
    }
}
