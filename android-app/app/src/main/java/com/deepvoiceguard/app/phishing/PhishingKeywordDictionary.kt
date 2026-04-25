package com.deepvoiceguard.app.phishing

import android.content.Context
import com.deepvoiceguard.app.phishing.model.KeywordEntry
import com.deepvoiceguard.app.phishing.model.MatchedKeyword
import com.deepvoiceguard.app.phishing.model.PhishingCategory
import org.json.JSONObject

/**
 * 보이스피싱 키워드 사전.
 * assets/phishing_keywords.json에서 키워드를 로드하며,
 * 빌트인 기본 사전도 포함한다.
 */
class PhishingKeywordDictionary(context: Context) {

    private val keywords: List<KeywordEntry>

    init {
        keywords = loadFromAssets(context) ?: builtinKeywords()
    }

    /**
     * 텍스트에서 매칭되는 키워드를 찾는다.
     * 정규화된 텍스트와 원본 텍스트 모두에서 검색.
     */
    fun findMatches(text: String): List<MatchedKeyword> {
        val normalized = KoreanNormalizer.normalize(text)
        val matches = mutableListOf<MatchedKeyword>()
        val seen = mutableSetOf<String>()

        for (entry in keywords) {
            // 메인 키워드 매칭
            if (normalized.contains(entry.keyword) && entry.keyword !in seen) {
                matches.add(
                    MatchedKeyword(
                        keyword = entry.keyword,
                        category = entry.category,
                        weight = entry.weight,
                        matchedText = resolveOriginalSubstring(text, entry.keyword, entry.synonyms),
                    )
                )
                seen.add(entry.keyword)
            }
            // 동의어 매칭
            for (synonym in entry.synonyms) {
                if (normalized.contains(synonym) && entry.keyword !in seen) {
                    matches.add(
                        MatchedKeyword(
                            keyword = entry.keyword,
                            category = entry.category,
                            weight = entry.weight,
                            matchedText = resolveOriginalSubstring(text, entry.keyword, entry.synonyms),
                        )
                    )
                    seen.add(entry.keyword)
                    break
                }
            }
        }
        return matches
    }

    /**
     * 원본 [text] 에서 [keyword] 또는 [synonyms] 중 하나를 plain (case-insensitive) 검색해
     * 발견된 substring 을 원본 케이스 그대로 반환. 못 찾으면 [keyword] 를 fallback (사전 literal).
     *
     * 매칭 자체는 detector 가 정규화 후 결정한 것이므로, 이 함수의 역할은 "정규화 거친 원본
     * 위치가 plain 검색으로 안 맞는" edge case (lowercase normalize, 어미 제거 등) 만 책임진다.
     * UI ([buildHighlightedTranscript]) 의 indexOf 검색 신뢰도를 ~100% 로 끌어올리는 보강.
     */
    private fun resolveOriginalSubstring(
        text: String,
        keyword: String,
        synonyms: List<String>,
    ): String {
        val candidates = listOf(keyword) + synonyms
        for (cand in candidates) {
            val idx = text.indexOf(cand, ignoreCase = true)
            if (idx >= 0) {
                return text.substring(idx, idx + cand.length)
            }
        }
        return keyword  // fallback: 사전 literal
    }

    /**
     * 매칭된 키워드의 가중치 합 기반 점수 (0.0~1.0).
     * 카테고리 다양성에 보너스 부여.
     */
    fun calculateScore(matches: List<MatchedKeyword>): Float {
        if (matches.isEmpty()) return 0f
        val weightSum = matches.sumOf { it.weight.toDouble() }.toFloat()
        val categoryBonus = matches.map { it.category }.distinct().size * 0.05f
        return (weightSum + categoryBonus).coerceIn(0f, 1f)
    }

    private fun loadFromAssets(context: Context): List<KeywordEntry>? {
        return try {
            val json = context.assets.open("phishing_keywords.json")
                .bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val arr = root.getJSONArray("keywords")
            (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                val synonyms = if (obj.has("synonyms")) {
                    val synArr = obj.getJSONArray("synonyms")
                    (0 until synArr.length()).map { synArr.getString(it) }
                } else emptyList()
                KeywordEntry(
                    keyword = obj.getString("keyword"),
                    category = PhishingCategory.valueOf(obj.getString("category")),
                    weight = obj.getDouble("weight").toFloat(),
                    synonyms = synonyms,
                )
            }
        } catch (_: Exception) {
            null
        }
    }

    /** JSON 파일이 없을 때 사용하는 빌트인 기본 사전. */
    private fun builtinKeywords(): List<KeywordEntry> = listOf(
        // 사칭
        KeywordEntry("검찰", PhishingCategory.IMPERSONATION, 0.4f, listOf("검찰청", "검사")),
        KeywordEntry("경찰", PhishingCategory.IMPERSONATION, 0.4f, listOf("경찰청", "경찰관", "수사관")),
        KeywordEntry("금융감독원", PhishingCategory.IMPERSONATION, 0.5f, listOf("금감원")),
        KeywordEntry("국세청", PhishingCategory.IMPERSONATION, 0.4f),
        KeywordEntry("법원", PhishingCategory.IMPERSONATION, 0.3f),
        KeywordEntry("수사관", PhishingCategory.IMPERSONATION, 0.4f),
        KeywordEntry("담당자", PhishingCategory.IMPERSONATION, 0.2f),
        // 긴급
        KeywordEntry("긴급", PhishingCategory.URGENCY, 0.2f, listOf("긴급하게")),
        KeywordEntry("즉시", PhishingCategory.URGENCY, 0.2f, listOf("당장", "지금 바로", "지금 당장")),
        KeywordEntry("빨리", PhishingCategory.URGENCY, 0.1f, listOf("서둘러")),
        KeywordEntry("오늘 안에", PhishingCategory.URGENCY, 0.2f, listOf("오늘 중으로")),
        // 금융행위
        KeywordEntry("계좌이체", PhishingCategory.FINANCIAL_ACTION, 0.4f, listOf("계좌 이체", "송금")),
        KeywordEntry("현금 인출", PhishingCategory.FINANCIAL_ACTION, 0.4f, listOf("현금인출", "출금")),
        KeywordEntry("atm", PhishingCategory.FINANCIAL_ACTION, 0.3f, listOf("ATM", "현금인출기")),
        KeywordEntry("안전한 계좌", PhishingCategory.FINANCIAL_ACTION, 0.5f, listOf("안전계좌", "보호계좌")),
        KeywordEntry("대포통장", PhishingCategory.FINANCIAL_ACTION, 0.4f),
        KeywordEntry("입금", PhishingCategory.FINANCIAL_ACTION, 0.2f),
        // 개인정보
        KeywordEntry("주민등록번호", PhishingCategory.PERSONAL_INFO, 0.3f, listOf("주민번호", "주민등록")),
        KeywordEntry("계좌번호", PhishingCategory.PERSONAL_INFO, 0.3f, listOf("통장번호")),
        KeywordEntry("비밀번호", PhishingCategory.PERSONAL_INFO, 0.3f, listOf("비번", "패스워드")),
        KeywordEntry("otp", PhishingCategory.PERSONAL_INFO, 0.3f, listOf("OTP", "인증번호", "보안카드")),
        // 위협
        KeywordEntry("체포", PhishingCategory.THREAT, 0.3f, listOf("체포영장")),
        KeywordEntry("구속", PhishingCategory.THREAT, 0.3f, listOf("구속영장")),
        KeywordEntry("범죄", PhishingCategory.THREAT, 0.2f, listOf("범죄 연루", "연루")),
        KeywordEntry("혐의", PhishingCategory.THREAT, 0.2f, listOf("피의자")),
        KeywordEntry("수사", PhishingCategory.THREAT, 0.2f, listOf("수사 중", "조사")),
        KeywordEntry("명의도용", PhishingCategory.THREAT, 0.3f, listOf("명의 도용", "도용")),
    )
}
