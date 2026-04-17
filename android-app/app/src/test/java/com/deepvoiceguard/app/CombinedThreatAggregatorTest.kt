package com.deepvoiceguard.app

import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.CombinedThreatAggregator
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.ThreatLevel
import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.phishing.model.PhishingThreatLevel
import com.deepvoiceguard.app.stt.SttStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CombinedThreatAggregatorTest {

    private val aggregator = CombinedThreatAggregator()

    @Test
    fun `decision table matches expected combined threat levels`() {
        data class Case(
            val name: String,
            val deepfake: ThreatLevel,
            val phishing: PhishingThreatLevel?,
            val sttStatus: SttStatus,
            val expected: CombinedThreatLevel,
        )

        val cases = listOf(
            Case("danger + phishing high + listening", ThreatLevel.DANGER, PhishingThreatLevel.HIGH, SttStatus.LISTENING, CombinedThreatLevel.CRITICAL),
            Case("danger + phishing low + listening", ThreatLevel.DANGER, PhishingThreatLevel.LOW, SttStatus.LISTENING, CombinedThreatLevel.DANGER),
            Case("danger + phishing none + listening", ThreatLevel.DANGER, PhishingThreatLevel.NONE, SttStatus.LISTENING, CombinedThreatLevel.DANGER),
            Case("danger + stt unavailable", ThreatLevel.DANGER, null, SttStatus.UNAVAILABLE, CombinedThreatLevel.DANGER),
            Case("warning + phishing high + listening", ThreatLevel.WARNING, PhishingThreatLevel.HIGH, SttStatus.LISTENING, CombinedThreatLevel.DANGER),
            Case("warning + phishing low + listening", ThreatLevel.WARNING, PhishingThreatLevel.LOW, SttStatus.LISTENING, CombinedThreatLevel.WARNING),
            Case("warning + phishing none + listening", ThreatLevel.WARNING, PhishingThreatLevel.NONE, SttStatus.LISTENING, CombinedThreatLevel.WARNING),
            Case("warning + stt unavailable", ThreatLevel.WARNING, null, SttStatus.UNAVAILABLE, CombinedThreatLevel.WARNING),
            Case("caution + phishing high + listening", ThreatLevel.CAUTION, PhishingThreatLevel.HIGH, SttStatus.LISTENING, CombinedThreatLevel.WARNING),
            Case("caution + phishing low + listening", ThreatLevel.CAUTION, PhishingThreatLevel.LOW, SttStatus.LISTENING, CombinedThreatLevel.CAUTION),
            Case("caution + stt unavailable", ThreatLevel.CAUTION, null, SttStatus.UNAVAILABLE, CombinedThreatLevel.CAUTION),
            Case("safe + phishing high + listening", ThreatLevel.SAFE, PhishingThreatLevel.HIGH, SttStatus.LISTENING, CombinedThreatLevel.WARNING),
            Case("safe + phishing low + listening", ThreatLevel.SAFE, PhishingThreatLevel.LOW, SttStatus.LISTENING, CombinedThreatLevel.CAUTION),
            Case("safe + phishing none + listening", ThreatLevel.SAFE, PhishingThreatLevel.NONE, SttStatus.LISTENING, CombinedThreatLevel.SAFE),
            Case("safe + stt unavailable", ThreatLevel.SAFE, null, SttStatus.UNAVAILABLE, CombinedThreatLevel.SAFE),
            Case("safe + medium phishing + listening", ThreatLevel.SAFE, PhishingThreatLevel.MEDIUM, SttStatus.LISTENING, CombinedThreatLevel.CAUTION),
        )

        cases.forEach { case ->
            val result = aggregator.combine(
                deepfakeResult = aggregated(case.deepfake),
                phishingResult = case.phishing?.let(::phishing),
                sttStatus = case.sttStatus,
                transcription = case.name,
            )

            assertEquals(case.name, case.expected, result.combinedThreatLevel)
            assertEquals(case.deepfake, result.deepfakeThreatLevel)
            assertEquals(case.sttStatus == SttStatus.LISTENING, result.sttAvailable)
            if (case.phishing == null) {
                assertEquals(0f, result.phishingScore)
                assertTrue(result.matchedKeywords.isEmpty())
                assertTrue(result.matchedPhrases.isEmpty())
            } else {
                assertEquals(case.phishing, result.phishingResult?.threatLevel)
                assertFalse(result.phishingScore < 0f)
            }
        }
    }

    @Test
    fun `deepfake only mapping preserves severity`() {
        assertEquals(CombinedThreatLevel.SAFE, CombinedThreatAggregator.deepfakeOnlyLevel(ThreatLevel.SAFE))
        assertEquals(CombinedThreatLevel.CAUTION, CombinedThreatAggregator.deepfakeOnlyLevel(ThreatLevel.CAUTION))
        assertEquals(CombinedThreatLevel.WARNING, CombinedThreatAggregator.deepfakeOnlyLevel(ThreatLevel.WARNING))
        assertEquals(CombinedThreatLevel.DANGER, CombinedThreatAggregator.deepfakeOnlyLevel(ThreatLevel.DANGER))
    }

    private fun aggregated(level: ThreatLevel): AggregatedResult = AggregatedResult(
        threatLevel = level,
        averageFakeScore = when (level) {
            ThreatLevel.SAFE -> 0.1f
            ThreatLevel.CAUTION -> 0.65f
            ThreatLevel.WARNING -> 0.78f
            ThreatLevel.DANGER -> 0.96f
        },
        latestResult = DetectionResult(
            fakeScore = 0.8f,
            realScore = 0.2f,
            confidence = 0.8f,
            latencyMs = 25L,
        ),
        consecutiveHighCount = if (level.ordinal >= ThreatLevel.WARNING.ordinal) 3 else 0,
    )

    private fun phishing(level: PhishingThreatLevel): PhishingResult = PhishingResult(
        score = when (level) {
            PhishingThreatLevel.NONE -> 0f
            PhishingThreatLevel.LOW -> 0.15f
            PhishingThreatLevel.MEDIUM -> 0.35f
            PhishingThreatLevel.HIGH -> 0.85f
        },
        matchedKeywords = emptyList(),
        matchedPhrases = emptyList(),
        threatLevel = level,
        transcription = "fixture",
    )
}
