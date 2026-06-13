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

    /**
     * 발화 구간 끝(plainText 내 누적 글자 인덱스 charEnd)에 해당하는 시각 범위.
     * STT 세그먼트 타임스탬프를 받아 자막을 실제 발화 시점에 맞춘다.
     */
    data class TimedSegment(val startMs: Int, val endMs: Int, val charEnd: Int)

    /** 파싱된 전사: 자막/피싱용 순수 텍스트 + (있으면) 타임스탬프 세그먼트. */
    data class ParsedTranscript(val text: String, val segments: List<TimedSegment>?)

    private val TIMED_LINE = Regex("""^\[(\d+)-(\d+)]\s*(.*)$""")

    /**
     * 전사 원문을 파싱한다.
     * - 모든 줄이 `[startMs-endMs] 텍스트` 패턴이면 타임스탬프 모드(세그먼트 텍스트를 공백으로 이어 plainText 구성).
     * - 한 줄이라도 패턴이 아니면 전체를 plain text로 취급(기존 시간 비례 fallback) — 마커 노출 방지.
     */
    fun parseTranscript(raw: String): ParsedTranscript {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return ParsedTranscript("", null)
        val lines = trimmed.lines().map { it.trim() }.filter { it.isNotEmpty() }
        val matches = lines.map { TIMED_LINE.matchEntire(it) }
        if (lines.isEmpty() || matches.any { it == null }) {
            return ParsedTranscript(trimmed, null)
        }
        val sb = StringBuilder()
        val segments = ArrayList<TimedSegment>(matches.size)
        for (m in matches) {
            val (s, e, txt) = m!!.destructured
            if (sb.isNotEmpty()) sb.append(' ')
            sb.append(txt)
            segments.add(TimedSegment(s.toInt(), e.toInt(), sb.length))
        }
        return ParsedTranscript(sb.toString(), segments)
    }

    /**
     * 타임스탬프 세그먼트 기준으로 offsetMs 시점까지 공개할 글자 수.
     * - 완료된 세그먼트는 전체 공개.
     * - 진행 중 세그먼트는 (startMs..endMs) 내 시간 비례로 보간 → 발화 중에만 글자가 차오름.
     * - 세그먼트 사이 침묵(이전 endMs ~ 다음 startMs)에는 글자가 멈춘다.
     */
    fun transcriptCharsTimed(segments: List<TimedSegment>, offsetMs: Int): Int {
        if (segments.isEmpty()) return 0
        var prevCharEnd = 0
        for (seg in segments) {
            when {
                offsetMs >= seg.endMs -> prevCharEnd = seg.charEnd
                offsetMs <= seg.startMs -> return prevCharEnd
                else -> {
                    val span = (seg.endMs - seg.startMs).coerceAtLeast(1)
                    val ratio = (offsetMs - seg.startMs).toFloat() / span
                    val segChars = (seg.charEnd - prevCharEnd).coerceAtLeast(0)
                    return prevCharEnd + (segChars * ratio).toInt().coerceIn(0, segChars)
                }
            }
        }
        return prevCharEnd
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
