package com.deepvoiceguard.app.inference

/** 탐지 위험 수준. */
enum class ThreatLevel {
    SAFE,       // 정상
    CAUTION,    // 주의 (평균 > cautionThreshold)
    WARNING,    // 경고 (연속 3개 > warningThreshold)
    DANGER,     // 위험 (연속 2개 > dangerThreshold)
}

/** 집계된 탐지 판정 결과. */
data class AggregatedResult(
    val threatLevel: ThreatLevel,
    val averageFakeScore: Float,
    val latestResult: DetectionResult,
    val consecutiveHighCount: Int,
)

/**
 * 최근 N개 탐지 결과를 슬라이딩 윈도우로 집계하여 위험 수준을 판정한다.
 */
class DetectionAggregator(
    private val windowSize: Int = 5,
    private val dangerThreshold: Float = 0.9f,
    private val warningThreshold: Float = 0.7f,
    private val cautionThreshold: Float = 0.6f,
) {

    private val results = ArrayDeque<DetectionResult>(windowSize)

    /** 새 탐지 결과를 추가하고 판정 결과를 반환. */
    fun add(result: DetectionResult): AggregatedResult {
        if (results.size >= windowSize) {
            results.removeFirst()
        }
        results.addLast(result)

        val avgFake = results.map { it.fakeScore }.average().toFloat()
        val consecutiveHigh = countConsecutiveHigh(threshold = warningThreshold)
        // 실기기 도메인 mismatch 완화: 같은 실음성에서도 AASIST 점수가 세그먼트마다 0.02~0.97 사이로
        // 널뛰는 현상이 있어(Galaxy Xcover 5 기준), 단일 스파이크로는 DANGER 띄우지 않는다.
        // 연속 2개 이상이 dangerThreshold를 넘어야만 DANGER.
        val consecutiveVeryHigh = countConsecutiveHigh(threshold = dangerThreshold)

        val threatLevel = when {
            consecutiveVeryHigh >= 2 -> ThreatLevel.DANGER
            consecutiveHigh >= 3 -> ThreatLevel.WARNING
            avgFake > cautionThreshold -> ThreatLevel.CAUTION
            else -> ThreatLevel.SAFE
        }

        return AggregatedResult(
            threatLevel = threatLevel,
            averageFakeScore = avgFake,
            latestResult = result,
            consecutiveHighCount = consecutiveHigh,
        )
    }

    /** 최근 결과에서 연속으로 [threshold] 이상인 횟수를 카운트. */
    private fun countConsecutiveHigh(threshold: Float): Int {
        var count = 0
        for (r in results.reversed()) {
            if (r.fakeScore > threshold) count++ else break
        }
        return count
    }

    fun clear() = results.clear()
}
