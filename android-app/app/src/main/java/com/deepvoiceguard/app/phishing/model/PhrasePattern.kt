package com.deepvoiceguard.app.phishing.model

/** 보이스피싱 구문 패턴. */
data class PhrasePattern(
    val phrase: String,
    val weight: Float = 1.0f,
    val category: PhishingCategory = PhishingCategory.IMPERSONATION,
)

/** 피싱 탐지 결과. */
data class PhishingResult(
    val score: Float,                     // 0.0 ~ 1.0
    val matchedKeywords: List<MatchedKeyword>,
    val matchedPhrases: List<String>,
    val threatLevel: PhishingThreatLevel,
    val transcription: String,
)

/** 피싱 위협 수준 (독립 — 딥보이스 ThreatLevel과 별개). */
enum class PhishingThreatLevel {
    NONE,       // 피싱 징후 없음
    LOW,        // 약한 징후 (단일 키워드)
    MEDIUM,     // 중간 (다수 키워드 또는 구문 매칭)
    HIGH,       // 높음 (강한 패턴 매칭)
}
