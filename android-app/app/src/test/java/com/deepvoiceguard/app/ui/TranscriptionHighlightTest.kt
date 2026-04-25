package com.deepvoiceguard.app.ui

import androidx.compose.ui.text.font.FontWeight
import com.deepvoiceguard.app.phishing.model.MatchedKeyword
import com.deepvoiceguard.app.phishing.model.PhishingCategory
import com.deepvoiceguard.app.ui.components.buildHighlightedTranscript
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TranscriptionHighlightTest {

    private fun mk(matchedText: String, keyword: String = matchedText) = MatchedKeyword(
        keyword = keyword,
        category = PhishingCategory.FINANCIAL_ACTION,
        weight = 1.0f,
        matchedText = matchedText,
    )

    @Test
    fun `no matches returns plain text`() {
        val result = buildHighlightedTranscript(
            transcription = "안녕하세요 좋은 아침입니다",
            matchedKeywords = emptyList(),
        )
        assertEquals("안녕하세요 좋은 아침입니다", result.text)
        assertTrue("no spans expected", result.spanStyles.isEmpty())
    }

    @Test
    fun `single matchedText is highlighted at correct range`() {
        val result = buildHighlightedTranscript(
            transcription = "계좌 이체를 부탁드립니다",
            matchedKeywords = listOf(mk(matchedText = "계좌")),
        )
        assertEquals("계좌 이체를 부탁드립니다", result.text)
        assertEquals(1, result.spanStyles.size)
        val span = result.spanStyles[0]
        assertEquals(0, span.start)
        assertEquals(2, span.end)
        assertEquals(FontWeight.Bold, span.item.fontWeight)
    }

    @Test
    fun `multiple occurrences of same matchedText all highlighted`() {
        // detector가 동일 matchedText 1개만 반환해도 UI는 transcription 전체에서 모든 occurrence 강조.
        val result = buildHighlightedTranscript(
            transcription = "계좌 비밀번호 계좌 이체",
            matchedKeywords = listOf(mk(matchedText = "계좌")),
        )
        val starts = result.spanStyles.map { it.start }
        // 두 번째 "계좌" 위치: "계좌"(2) + " "(1) + "비밀번호"(4) + " "(1) = index 8
        assertEquals(listOf(0, 8), starts)
    }

    @Test
    fun `multiple distinct matchedTexts all highlighted`() {
        val result = buildHighlightedTranscript(
            transcription = "비밀번호와 계좌번호를 알려주세요",
            matchedKeywords = listOf(
                mk(matchedText = "비밀번호"),
                mk(matchedText = "계좌번호"),
            ),
        )
        assertEquals(2, result.spanStyles.size)
        // 위치 검증 (0, 6)
        assertEquals(0, result.spanStyles[0].start)
        assertEquals(6, result.spanStyles[1].start)
    }

    @Test
    fun `matchedText with whitespace inside is highlighted`() {
        // detector 가 "계좌 이체" 처럼 공백 포함 substring을 반환할 수 있음 (PhraseMatcher 결과).
        val result = buildHighlightedTranscript(
            transcription = "당장 계좌 이체 해주세요",
            matchedKeywords = listOf(mk(matchedText = "계좌 이체")),
        )
        assertEquals(1, result.spanStyles.size)
        val span = result.spanStyles[0]
        assertEquals(3, span.start)
        assertEquals(8, span.end)
    }

    @Test
    fun `nested or overlapping matchedTexts longer wins`() {
        // 같은 위치에 두 매칭이 있으면 더 긴 substring 우선 (시각적 일관성).
        // 예: "검찰" 과 "검찰청" 둘 다 detector 가 반환한 케이스.
        val result = buildHighlightedTranscript(
            transcription = "검찰청에서 연락 드립니다",
            matchedKeywords = listOf(
                mk(matchedText = "검찰"),
                mk(matchedText = "검찰청"),
            ),
        )
        assertEquals(1, result.spanStyles.size)
        val span = result.spanStyles[0]
        assertEquals(0, span.start)
        assertEquals(3, span.end)  // "검찰청"
    }

    @Test
    fun `matchedText not found in transcription is silently skipped`() {
        // detector 가 정규화 거친 후 매칭 — 원본 transcription에 정확한 substring 이 없을 수도 있음.
        // 그 경우 UI는 강조 없이 plain text 유지 (절대 throw 하지 않음).
        val result = buildHighlightedTranscript(
            transcription = "안녕하세요 좋은 아침",
            matchedKeywords = listOf(mk(matchedText = "안 녕")),  // 공백 위치 다름
        )
        assertEquals("안녕하세요 좋은 아침", result.text)
        assertTrue(result.spanStyles.isEmpty())
    }

    @Test
    fun `empty transcription returns empty annotated string`() {
        val result = buildHighlightedTranscript(
            transcription = "",
            matchedKeywords = listOf(mk(matchedText = "키워드")),
        )
        assertEquals("", result.text)
        assertTrue(result.spanStyles.isEmpty())
    }

    @Test
    fun `empty matchedText is ignored`() {
        // 방어적 — detector 가 빈 문자열 반환할 일은 없지만 안전장치.
        val result = buildHighlightedTranscript(
            transcription = "안녕하세요",
            matchedKeywords = listOf(mk(matchedText = "")),
        )
        assertEquals("안녕하세요", result.text)
        assertTrue(result.spanStyles.isEmpty())
    }
}
