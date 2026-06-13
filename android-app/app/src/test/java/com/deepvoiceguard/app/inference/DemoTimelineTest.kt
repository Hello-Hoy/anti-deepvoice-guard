package com.deepvoiceguard.app.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoTimelineTest {

    @Test
    fun vadActiveAt_picks_frame_by_offset_and_threshold() {
        val probs = floatArrayOf(0.1f, 0.9f, 0.2f)
        assertFalse(DemoTimelineMath.vadActiveAt(probs, frameMs = 32, offsetMs = 0, threshold = 0.5f))
        assertTrue(DemoTimelineMath.vadActiveAt(probs, frameMs = 32, offsetMs = 40, threshold = 0.5f))
        assertFalse(DemoTimelineMath.vadActiveAt(probs, frameMs = 32, offsetMs = 70, threshold = 0.5f))
    }

    @Test
    fun vadActiveAt_out_of_range_is_false() {
        val probs = floatArrayOf(0.9f)
        assertFalse(DemoTimelineMath.vadActiveAt(probs, 32, offsetMs = 9999, threshold = 0.5f))
        assertFalse(DemoTimelineMath.vadActiveAt(FloatArray(0), 32, 0, 0.5f))
    }

    @Test
    fun stepHoldIndexAt_returns_last_step_at_or_before_offset() {
        val stepTimesMs = intArrayOf(1000, 2000, 3000)
        assertEquals(-1, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 500))
        assertEquals(0, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 1000))
        assertEquals(0, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 1999))
        assertEquals(1, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 2000))
        assertEquals(2, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 99999))
    }

    @Test
    fun transcriptCharsAt_is_proportional_and_clamped() {
        assertEquals(0, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 0, durationMs = 1000))
        assertEquals(5, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 500, durationMs = 1000))
        assertEquals(10, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 5000, durationMs = 1000))
        assertEquals(0, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 100, durationMs = 0))
    }

    @Test
    fun parseTranscript_plain_text_has_no_segments() {
        val p = DemoTimelineMath.parseTranscript("안녕하세요. 반갑습니다.")
        assertEquals("안녕하세요. 반갑습니다.", p.text)
        assertEquals(null, p.segments)
    }

    @Test
    fun parseTranscript_timed_lines_build_text_and_segments() {
        val raw = "[0-1000] 가나\n[1500-2000] 다라마"
        val p = DemoTimelineMath.parseTranscript(raw)
        // 세그먼트 텍스트는 공백으로 이어진다.
        assertEquals("가나 다라마", p.text)
        val segs = p.segments!!
        assertEquals(2, segs.size)
        assertEquals(DemoTimelineMath.TimedSegment(0, 1000, 2), segs[0])      // "가나" → charEnd=2
        assertEquals(DemoTimelineMath.TimedSegment(1500, 2000, 6), segs[1])   // " 다라마" → charEnd=6
    }

    @Test
    fun parseTranscript_falls_back_to_plain_when_any_line_unmarked() {
        val raw = "[0-1000] 가나\n마커없는줄"
        val p = DemoTimelineMath.parseTranscript(raw)
        assertEquals(null, p.segments)
        assertEquals(raw, p.text)
    }

    @Test
    fun transcriptCharsTimed_interpolates_holds_and_completes() {
        // "가나"(0~1000ms, charEnd=2), 침묵(1000~1500), "다라마"(1500~2000, charEnd=6)
        val segs = listOf(
            DemoTimelineMath.TimedSegment(0, 1000, 2),
            DemoTimelineMath.TimedSegment(1500, 2000, 6),
        )
        assertEquals(0, DemoTimelineMath.transcriptCharsTimed(segs, 0))
        assertEquals(1, DemoTimelineMath.transcriptCharsTimed(segs, 500))     // seg1 중간
        assertEquals(2, DemoTimelineMath.transcriptCharsTimed(segs, 1000))    // seg1 완료
        assertEquals(2, DemoTimelineMath.transcriptCharsTimed(segs, 1250))    // 침묵 구간 → 유지
        assertEquals(4, DemoTimelineMath.transcriptCharsTimed(segs, 1750))    // seg2 중간(2 + 4*0.5)
        assertEquals(6, DemoTimelineMath.transcriptCharsTimed(segs, 2000))    // seg2 완료
        assertEquals(6, DemoTimelineMath.transcriptCharsTimed(segs, 9999))    // 끝 이후
    }

    @Test
    fun transcriptCharsTimed_empty_is_zero() {
        assertEquals(0, DemoTimelineMath.transcriptCharsTimed(emptyList(), 500))
    }

    @Test
    fun windowEndingAt_left_pads_when_short() {
        val audio = floatArrayOf(1f, 2f, 3f, 4f)
        val w = DemoTimelineMath.windowEndingAt(audio, endSample = 2, length = 4)
        assertEquals(4, w.size)
        assertEquals(0f, w[0], 0f); assertEquals(0f, w[1], 0f)
        assertEquals(1f, w[2], 0f); assertEquals(2f, w[3], 0f)
    }

    @Test
    fun windowEndingAt_takes_tail_when_long() {
        val audio = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val w = DemoTimelineMath.windowEndingAt(audio, endSample = 5, length = 3)
        assertEquals(3, w.size)
        assertEquals(3f, w[0], 0f); assertEquals(4f, w[1], 0f); assertEquals(5f, w[2], 0f)
    }
}
