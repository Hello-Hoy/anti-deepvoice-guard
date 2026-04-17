package com.deepvoiceguard.app.phishing

import android.content.Context
import com.deepvoiceguard.app.phishing.model.MatchedKeyword
import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.phishing.model.PhishingThreatLevel

/**
 * 보이스피싱 키워드 탐지 엔진.
 * 2단계 하이브리드 탐지:
 *  - Tier 1: 키워드 사전 매칭 (즉시, 정확)
 *  - Tier 2: 구문 패턴 매칭 (ML 수준, bigram/trigram 기반)
 *
 * 두 결과를 결합하여 최종 피싱 점수/위협 레벨을 산출한다.
 */
class PhishingKeywordDetector(context: Context) {

    private val dictionary = PhishingKeywordDictionary(context)
    private val phraseMatcher = PhraseMatcher(context)

    companion object {
        // MEDIUM 7: scoring 매직 넘버를 의미 있는 상수로 승격.
        private const val HIGH_THRESHOLD = 0.5f
        private const val MEDIUM_THRESHOLD = 0.25f
        private const val LOW_THRESHOLD = 0.1f
        private const val MONEY_BONUS = 0.1f
        private const val SYNERGY_BONUS = 0.1f
        private const val SYNERGY_FLOOR = 0.1f
    }

    /**
     * 텍스트를 분석하여 피싱 탐지 결과를 반환한다.
     *
     * @param text 분석할 전사 텍스트 (60초 윈도우)
     * @return PhishingResult (점수, 매칭 키워드, 매칭 구문, 위협 레벨)
     */
    fun analyze(text: String): PhishingResult {
        if (text.isBlank()) {
            return PhishingResult(
                score = 0f,
                matchedKeywords = emptyList(),
                matchedPhrases = emptyList(),
                threatLevel = PhishingThreatLevel.NONE,
                transcription = text,
            )
        }

        // Tier 1: 키워드 사전 매칭
        val matchedKeywords: List<MatchedKeyword> = dictionary.findMatches(text)
        val keywordScore = dictionary.calculateScore(matchedKeywords)

        // Tier 2: 구문 패턴 매칭
        val (phraseScore, matchedPhrases) = phraseMatcher.match(text)

        // 금액 패턴 보너스
        val moneyBonus = if (KoreanNormalizer.containsMoneyPattern(text)) MONEY_BONUS else 0f

        // 최종 점수 = max(키워드, 구문) + 금액 보너스
        // 두 시그널이 모두 있으면 시너지 보너스
        val synergyBonus = if (
            keywordScore > SYNERGY_FLOOR && phraseScore > SYNERGY_FLOOR
        ) {
            SYNERGY_BONUS
        } else {
            0f
        }
        val finalScore = (maxOf(keywordScore, phraseScore) + moneyBonus + synergyBonus)
            .coerceIn(0f, 1f)

        val threatLevel = when {
            finalScore >= HIGH_THRESHOLD -> PhishingThreatLevel.HIGH
            finalScore >= MEDIUM_THRESHOLD -> PhishingThreatLevel.MEDIUM
            finalScore >= LOW_THRESHOLD -> PhishingThreatLevel.LOW
            else -> PhishingThreatLevel.NONE
        }

        return PhishingResult(
            score = finalScore,
            matchedKeywords = matchedKeywords,
            matchedPhrases = matchedPhrases,
            threatLevel = threatLevel,
            transcription = text,
        )
    }
}
