package com.deepvoiceguard.app.inference

import com.deepvoiceguard.app.phishing.model.MatchedKeyword
import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.stt.SttStatus

/**
 * 통합 위협 레벨 (딥보이스 ThreatLevel과 별개).
 * 딥보이스 + 피싱 결과를 결합한 최종 위협 수준.
 */
enum class CombinedThreatLevel {
    SAFE,       // 정상
    CAUTION,    // 주의
    WARNING,    // 경고
    DANGER,     // 위험
    CRITICAL,   // 최고 위험 (딥보이스 + 피싱 동시)
}

/**
 * 통합 위협 분석 결과.
 * 딥보이스 탐지 결과와 피싱 키워드 탐지 결과를 결합한 최종 결과.
 */
data class CombinedThreatResult(
    val combinedThreatLevel: CombinedThreatLevel,

    // 딥보이스 결과
    val deepfakeResult: AggregatedResult?,
    val deepfakeThreatLevel: ThreatLevel,

    // 피싱 결과
    val phishingResult: PhishingResult?,
    val phishingScore: Float,
    val matchedKeywords: List<MatchedKeyword>,
    val matchedPhrases: List<String>,

    // STT 상태
    val sttStatus: SttStatus,
    val sttAvailable: Boolean,

    // 전사 텍스트
    val transcription: String,

    val timestampMs: Long = System.currentTimeMillis(),

    /**
     * **H103**: dedupe-suppressed fallback 여부. 이 flag가 true면 collector는 row를
     * persistence 경로에만 저장하고 `showCombinedAlert()`는 호출하지 않는다. combined dedupe
     * 로직이 alert spam을 suppress했지만 history/metric은 유지해야 할 때 사용.
     */
    val suppressAlert: Boolean = false,

    /**
     * **H167**: emit 시점의 incident generation. collector가 이 값과 현재 incident generation을
     * 비교하여 stale event 판별. SAFE 전환 등으로 generation이 advance했으면 이 event는 obsolete.
     */
    val emitGeneration: Long = 0L,

    /**
     * **H185**: emit 시점에 캡처된 audio snapshot (딥보이스/피싱 분석 시점의 원음).
     * collector는 DB 저장 시 반드시 이 값을 사용하고 ringBuffer를 재차 읽지 않아야 한다.
     * STT/phishing 분석 지연 동안 ringBuffer는 이미 다음 구간으로 넘어갔을 수 있으므로,
     * 이벤트와 저장 audio의 정합성은 emit 시점 snapshot으로만 보장된다.
     *
     * 크기가 0인 빈 배열이면 snapshot이 부재함을 의미 (하위 호환용; 기본값).
     */
    val audioSnapshot: FloatArray = FloatArray(0),
) {
    /**
     * FloatArray 동등성 비교가 성능/정합성상 부적합하므로 data class의 기본 equals/hashCode를
     * 재정의한다. identity 기반 비교로 충분 — event 자체는 SharedFlow에서 id로 유일.
     */
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}
