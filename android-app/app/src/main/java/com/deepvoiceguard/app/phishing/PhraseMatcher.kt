package com.deepvoiceguard.app.phishing

import android.content.Context
import com.deepvoiceguard.app.phishing.model.PhrasePattern
import com.deepvoiceguard.app.phishing.model.PhishingCategory
import org.json.JSONObject

/**
 * 보이스피싱 구문 패턴 매처 (Tier 2).
 * bigram/trigram 기반으로 사전 정의된 피싱 시나리오 구문과 유사도를 측정한다.
 */
class PhraseMatcher(context: Context) {

    private val patterns: List<PhrasePattern>

    init {
        patterns = loadFromAssets(context) ?: builtinPatterns()
    }

    /**
     * 텍스트에서 매칭되는 구문 패턴을 찾고 유사도 점수를 반환한다.
     * @return Pair(매칭 점수 0.0~1.0, 매칭된 구문 리스트)
     */
    fun match(text: String): Pair<Float, List<String>> {
        if (text.isBlank() || patterns.isEmpty()) return 0f to emptyList()

        val normalized = KoreanNormalizer.normalize(text)
        val matchedPhrases = mutableListOf<String>()
        var totalWeight = 0f

        for (pattern in patterns) {
            if (containsPhrase(normalized, pattern.phrase)) {
                matchedPhrases.add(pattern.phrase)
                totalWeight += pattern.weight
            }
        }

        // 전체 패턴 대비 매칭 비율 + 가중치
        val maxPossibleWeight = patterns.sumOf { it.weight.toDouble() }.toFloat()
        val score = if (maxPossibleWeight > 0) {
            (totalWeight / maxPossibleWeight).coerceIn(0f, 1f)
        } else 0f

        return score to matchedPhrases
    }

    /**
     * 구문이 텍스트에 포함되는지 확인.
     * 정확한 연속 매칭 + 부분 토큰 매칭 지원.
     */
    private fun containsPhrase(normalizedText: String, phrase: String): Boolean {
        // 1차: 직접 포함
        if (normalizedText.contains(KoreanNormalizer.normalize(phrase))) {
            return true
        }
        // 2차: 토큰 기반 — 구문의 모든 토큰이 텍스트에 존재
        val phraseTokens = KoreanNormalizer.tokenize(phrase)
        val textTokens = KoreanNormalizer.tokenize(normalizedText)
        if (phraseTokens.isEmpty()) return false
        return phraseTokens.all { pt -> textTokens.any { tt -> tt.contains(pt) || pt.contains(tt) } }
    }

    private fun loadFromAssets(context: Context): List<PhrasePattern>? {
        return try {
            val json = context.assets.open("phishing_phrases.json")
                .bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val arr = root.getJSONArray("phrases")
            (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                PhrasePattern(
                    phrase = obj.getString("phrase"),
                    weight = obj.optDouble("weight", 1.0).toFloat(),
                    category = try {
                        PhishingCategory.valueOf(obj.optString("category", "IMPERSONATION"))
                    } catch (_: Exception) {
                        PhishingCategory.IMPERSONATION
                    },
                )
            }
        } catch (_: Exception) {
            null
        }
    }

    /** JSON 파일이 없을 때 사용하는 빌트인 구문 패턴. */
    private fun builtinPatterns(): List<PhrasePattern> = listOf(
        // 검찰/경찰 사칭
        PhrasePattern("검찰청 수사관", 1.5f, PhishingCategory.IMPERSONATION),
        PhrasePattern("수사에 협조", 1.2f, PhishingCategory.IMPERSONATION),
        PhrasePattern("사건 번호", 1.0f, PhishingCategory.IMPERSONATION),
        PhrasePattern("출석 요구", 1.0f, PhishingCategory.IMPERSONATION),
        // 금융 사기
        PhrasePattern("안전한 계좌로 이체", 2.0f, PhishingCategory.FINANCIAL_ACTION),
        PhrasePattern("계좌가 범죄에 연루", 1.8f, PhishingCategory.FINANCIAL_ACTION),
        PhrasePattern("자금 세탁", 1.5f, PhishingCategory.FINANCIAL_ACTION),
        PhrasePattern("보호 조치", 1.2f, PhishingCategory.FINANCIAL_ACTION),
        // 개인정보 탈취
        PhrasePattern("명의가 도용", 1.5f, PhishingCategory.PERSONAL_INFO),
        PhrasePattern("개인정보가 유출", 1.5f, PhishingCategory.PERSONAL_INFO),
        PhrasePattern("본인 확인을 위해", 0.8f, PhishingCategory.PERSONAL_INFO),
        // 위협/긴급
        PhrasePattern("체포 영장", 1.5f, PhishingCategory.THREAT),
        PhrasePattern("구속 영장이 발부", 1.8f, PhishingCategory.THREAT),
        PhrasePattern("지금 바로 현금", 1.5f, PhishingCategory.URGENCY),
        PhrasePattern("오늘 안에 처리", 1.2f, PhishingCategory.URGENCY),
    )
}
