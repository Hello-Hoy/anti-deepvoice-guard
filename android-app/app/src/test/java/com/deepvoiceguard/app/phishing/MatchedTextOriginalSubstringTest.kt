package com.deepvoiceguard.app.phishing

import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Phase 1.5 — matchedText 가 원본 transcription substring 우선되는지 검증.
 *
 * Robolectric 미적용 환경. mockk(relaxed=true) 로 Context 생성 시 assets.open() 이
 * mock InputStream 반환 → JSONObject 파싱 실패 → catch 블록에서 null 반환 →
 * builtinKeywords() fallback 으로 진입 (PhishingKeywordDictionary.kt:loadFromAssets).
 * 따라서 본 테스트는 builtin 사전 항목들 (예: "계좌이체" + synonym "계좌 이체") 을 활용.
 */
class MatchedTextOriginalSubstringTest {

    private val dict: PhishingKeywordDictionary by lazy {
        PhishingKeywordDictionary(mockk(relaxed = true))
    }

    @Test
    fun `matchedText preserves whitespace from original transcription`() {
        // builtin "계좌이체" entry 의 synonym "계좌 이체" → 사용자 발화 "계좌 이체" 와 일치.
        // 보강 후 matchedText 는 원본 substring "계좌 이체" 가 들어가야 (helper 가 candidate
        // list 를 plain 검색 — synonym 도 후보).
        val text = "지금 당장 계좌 이체 해주세요"
        val matches = dict.findMatches(text)
        val financial = matches.firstOrNull { it.keyword == "계좌이체" }
        assertTrue("계좌이체 entry must match", financial != null)
        // 원본에 "계좌 이체" 가 정확히 들어있으므로 helper 가 그것을 우선 반환.
        assertEquals("계좌 이체", financial!!.matchedText)
    }

    @Test
    fun `matchedText for synonym uses original substring when present`() {
        // builtin "검찰" entry 의 synonym 들: "검찰청", "검사".
        // 사용자가 "검찰청에서" 말함 → matchedText="검찰청" (원본 substring).
        val text = "검찰청에서 연락드립니다"
        val matches = dict.findMatches(text)
        val impersonation = matches.firstOrNull { it.keyword == "검찰" }
        assertTrue("검찰 entry must match", impersonation != null)
        assertTrue(
            "matchedText 는 원본 transcription 에 plain 검색으로 들어있어야",
            text.contains(impersonation!!.matchedText),
        )
    }

    @Test
    fun `case-insensitive original substring (ATM example)`() {
        // builtin "atm" entry — 사용자가 대문자 "ATM" 사용. helper 의 ignoreCase 검색이
        // 원본 "ATM" substring 을 살려야 — UI indexOf 는 case-sensitive 기본이라 lowercase
        // "atm" 으로는 못 찾기 때문.
        val text = "ATM 이용해주세요"
        val matches = dict.findMatches(text)
        val atm = matches.firstOrNull { it.keyword == "atm" }
        assertTrue("atm entry must match", atm != null)
        // 원본 케이스 "ATM" 보존
        assertEquals("ATM", atm!!.matchedText)
    }

    @Test
    fun `falls back to dictionary literal when nothing matches in original`() {
        // edge case: 매칭 자체가 없으면 빈 결과.
        val text = "별 의미 없는 텍스트"
        val matches = dict.findMatches(text)
        assertTrue("매칭 자체가 없으면 빈 결과", matches.isEmpty())
    }
}
