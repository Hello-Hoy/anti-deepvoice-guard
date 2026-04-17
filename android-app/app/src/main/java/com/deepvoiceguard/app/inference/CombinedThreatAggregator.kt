package com.deepvoiceguard.app.inference

import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.phishing.model.PhishingThreatLevel
import com.deepvoiceguard.app.stt.SttStatus

/**
 * 딥보이스 탐지 결과와 피싱 키워드 탐지 결과를 결합하여
 * 통합 위협 레벨을 산출하는 의사결정 엔진.
 *
 * 의사결정 테이블 (13행, doc drift fix):
 * | 딥보이스 | 피싱   | STT  | 통합         |
 * |---------|--------|------|-------------|
 * | DANGER  | High   | OK   | CRITICAL    |
 * | DANGER  | Low/-  | OK   | DANGER      |
 * | DANGER  | -      | 불가 | DANGER      |
 * | WARNING | High   | OK   | DANGER(상향) |
 * | WARNING | Low/-  | OK   | WARNING     |
 * | WARNING | -      | 불가 | WARNING     |
 * | CAUTION | High   | OK   | WARNING     |
 * | CAUTION | Low/-  | OK   | CAUTION     |
 * | SAFE    | High   | OK   | WARNING     |
 * | SAFE    | Medium/Low | OK | CAUTION    |
 * | SAFE    | None   | OK   | SAFE        |
 * | CAUTION | -      | 불가 | CAUTION     |
 * | SAFE    | -      | 불가 | SAFE        |
 */
class CombinedThreatAggregator {

    /**
     * 딥보이스 + 피싱 결과를 결합한다.
     *
     * @param deepfakeResult 딥보이스 탐지 결과 (null이면 탐지 미수행)
     * @param phishingResult 피싱 키워드 탐지 결과 (null이면 STT 미가용)
     * @param sttStatus 현재 STT 상태
     * @param transcription 전사 텍스트
     */
    fun combine(
        deepfakeResult: AggregatedResult?,
        phishingResult: PhishingResult?,
        sttStatus: SttStatus,
        transcription: String,
    ): CombinedThreatResult {
        val deepfakeThreat = deepfakeResult?.threatLevel ?: ThreatLevel.SAFE
        // **LISTENING만 sttAvailable=true** — READY는 prepared/stopped 상태.
        // AudioPipeline.isSttUsable()과 일관되게 처리하여 post-stop 업데이트가
        // STT-derived 저장/알림 경로로 흘러들지 않게 한다.
        val sttAvailable = sttStatus == SttStatus.LISTENING
        val phishingHigh = phishingResult != null &&
                phishingResult.threatLevel == PhishingThreatLevel.HIGH
        val phishingMedium = phishingResult != null &&
                phishingResult.threatLevel == PhishingThreatLevel.MEDIUM
        val phishingLow = phishingResult != null &&
                phishingResult.threatLevel == PhishingThreatLevel.LOW

        val combinedLevel = when {
            // STT 불가 → 딥보이스 결과만 적용
            !sttAvailable || phishingResult == null -> deepfakeOnlyLevel(deepfakeThreat)

            // DANGER + 피싱 High → CRITICAL
            deepfakeThreat == ThreatLevel.DANGER && phishingHigh -> CombinedThreatLevel.CRITICAL
            // DANGER + 나머지 → DANGER
            deepfakeThreat == ThreatLevel.DANGER -> CombinedThreatLevel.DANGER

            // WARNING + 피싱 High → DANGER (상향)
            deepfakeThreat == ThreatLevel.WARNING && phishingHigh -> CombinedThreatLevel.DANGER
            // WARNING + 나머지 → WARNING
            deepfakeThreat == ThreatLevel.WARNING -> CombinedThreatLevel.WARNING

            // CAUTION + 피싱 High → WARNING
            deepfakeThreat == ThreatLevel.CAUTION && phishingHigh -> CombinedThreatLevel.WARNING
            // CAUTION + 나머지 → CAUTION
            deepfakeThreat == ThreatLevel.CAUTION -> CombinedThreatLevel.CAUTION

            // SAFE + 피싱 High → WARNING
            deepfakeThreat == ThreatLevel.SAFE && phishingHigh -> CombinedThreatLevel.WARNING
            // SAFE + 피싱 Medium/Low → CAUTION
            deepfakeThreat == ThreatLevel.SAFE && (phishingMedium || phishingLow) ->
                CombinedThreatLevel.CAUTION

            // SAFE + 피싱 None → SAFE
            else -> CombinedThreatLevel.SAFE
        }

        return CombinedThreatResult(
            combinedThreatLevel = combinedLevel,
            deepfakeResult = deepfakeResult,
            deepfakeThreatLevel = deepfakeThreat,
            phishingResult = phishingResult,
            phishingScore = phishingResult?.score ?: 0f,
            matchedKeywords = phishingResult?.matchedKeywords ?: emptyList(),
            matchedPhrases = phishingResult?.matchedPhrases ?: emptyList(),
            sttStatus = sttStatus,
            sttAvailable = sttAvailable,
            transcription = transcription,
        )
    }
    companion object {
        /**
         * 외부에서 STT 데이터를 무효화한 뒤 통합 레벨을 재계산할 때 사용한다.
         * 동의 철회 후 큐된 이벤트의 sanitize 경로에서 호출.
         */
        fun deepfakeOnlyLevel(threat: ThreatLevel): CombinedThreatLevel = when (threat) {
            ThreatLevel.DANGER -> CombinedThreatLevel.DANGER
            ThreatLevel.WARNING -> CombinedThreatLevel.WARNING
            ThreatLevel.CAUTION -> CombinedThreatLevel.CAUTION
            ThreatLevel.SAFE -> CombinedThreatLevel.SAFE
        }
    }
}
