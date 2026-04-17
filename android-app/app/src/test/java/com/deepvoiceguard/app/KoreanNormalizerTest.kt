package com.deepvoiceguard.app

import com.deepvoiceguard.app.phishing.KoreanNormalizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class KoreanNormalizerTest {

    @Test
    fun `tokenize strips common particles from nouns`() {
        assertEquals(listOf("검찰청"), KoreanNormalizer.tokenize("검찰청에서"))
        assertEquals(listOf("계좌"), KoreanNormalizer.tokenize("계좌로"))
    }

    @Test
    fun `honorific endings normalize to the same canonical form`() {
        assertEquals("이체", KoreanNormalizer.normalize("이체하십시오"))
        assertEquals("이체", KoreanNormalizer.normalize("이체합니다"))
        assertEquals("이체", KoreanNormalizer.normalize("이체해요"))
    }

    @Test
    fun `money expressions are detected across korean words and numerals`() {
        assertTrue(KoreanNormalizer.containsMoneyPattern("삼백만원을 안전계좌로 보내세요"))
        assertTrue(KoreanNormalizer.containsMoneyPattern("300만원만 먼저 이체해 주세요"))
    }
}
