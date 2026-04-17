package com.deepvoiceguard.app.phishing.model

/** 보이스피싱 키워드 카테고리. */
enum class PhishingCategory(val displayName: String) {
    IMPERSONATION("사칭"),
    URGENCY("긴급"),
    FINANCIAL_ACTION("금융행위"),
    PERSONAL_INFO("개인정보"),
    THREAT("위협"),
}

/** 키워드 사전 항목. */
data class KeywordEntry(
    val keyword: String,
    val category: PhishingCategory,
    val weight: Float,
    val synonyms: List<String> = emptyList(),
)

/** 키워드 매칭 결과. */
data class MatchedKeyword(
    val keyword: String,
    val category: PhishingCategory,
    val weight: Float,
    val matchedText: String,
)
