package com.deepvoiceguard.app.phishing

/**
 * 간이 한국어 텍스트 정규화.
 * 형태소 분석기 없이 조사/어미 제거, 공백 정규화, 동의어 통합을 수행한다.
 */
object KoreanNormalizer {

    // 자주 등장하는 한국어 조사/어미 패턴 (길이 내림차순 정렬)
    private val suffixes = listOf(
        "에서는", "으로는", "이라고", "에서", "으로", "부터", "까지",
        "에게", "한테", "보다", "처럼", "같이", "마다", "이나", "거나",
        "이고", "하고", "에는", "에도", "이든", "든지",
        "은", "는", "이", "가", "을", "를", "의", "에", "로", "도",
        "와", "과", "나", "며",
    )

    // 존칭 어미 → 기본형
    private val endingNormalizations = mapOf(
        "하십시오" to "",
        "하세요" to "",
        "합니다" to "",
        "됩니다" to "",
        "입니다" to "",
        "습니다" to "",
        "세요" to "",
        "해요" to "",
        "에요" to "",
        "이요" to "",
    )

    /**
     * 텍스트를 정규화한다.
     * 1. 공백 정규화
     * 2. 조사/어미 제거
     * 3. 소문자 변환 (영문)
     */
    fun normalize(text: String): String {
        return text
            .trim()
            .replace(Regex("\\s+"), " ")
            .lowercase()
            .let { removeEndings(it) }
    }

    /** 단어 단위로 조사를 제거한 토큰 리스트 반환. */
    fun tokenize(text: String): List<String> {
        return normalize(text)
            .split(" ")
            .map { stripSuffix(it) }
            .filter { it.isNotBlank() && it.length > 1 }
    }

    /** bigram 생성 (키워드 탐지용). */
    fun bigrams(tokens: List<String>): List<String> {
        if (tokens.size < 2) return emptyList()
        return tokens.windowed(2) { it.joinToString(" ") }
    }

    /** trigram 생성. */
    fun trigrams(tokens: List<String>): List<String> {
        if (tokens.size < 3) return emptyList()
        return tokens.windowed(3) { it.joinToString(" ") }
    }

    private fun stripSuffix(word: String): String {
        if (word.length <= 2) return word
        for (suffix in suffixes) {
            if (word.length > suffix.length && word.endsWith(suffix)) {
                return word.dropLast(suffix.length)
            }
        }
        return word
    }

    private fun removeEndings(text: String): String {
        var result = text
        for ((ending, replacement) in endingNormalizations) {
            result = result.replace(ending, replacement)
        }
        return result.trim()
    }

    /**
     * 금액 패턴 감지. "삼백만원", "300만원", "5천만 원" 등.
     * 반환: true이면 금액 관련 표현이 포함됨.
     */
    fun containsMoneyPattern(text: String): Boolean {
        val patterns = listOf(
            Regex("[0-9]+만\\s?원"),
            Regex("[0-9]+억\\s?원"),
            Regex("[0-9]+천만\\s?원"),
            Regex("[일이삼사오육칠팔구십백천만억]+\\s?원"),
        )
        return patterns.any { it.containsMatchIn(text) }
    }
}
