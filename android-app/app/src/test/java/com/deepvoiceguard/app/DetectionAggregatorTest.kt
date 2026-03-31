package com.deepvoiceguard.app

import com.deepvoiceguard.app.inference.DetectionAggregator
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.ThreatLevel
import org.junit.Assert.assertEquals
import org.junit.Test

class DetectionAggregatorTest {

    private fun result(fakeScore: Float) = DetectionResult(
        fakeScore = fakeScore,
        realScore = 1f - fakeScore,
        confidence = maxOf(fakeScore, 1f - fakeScore),
        latencyMs = 50,
    )

    @Test
    fun `single safe result returns SAFE`() {
        val agg = DetectionAggregator()
        val r = agg.add(result(0.2f))
        assertEquals(ThreatLevel.SAFE, r.threatLevel)
    }

    @Test
    fun `single high score returns DANGER`() {
        val agg = DetectionAggregator()
        val r = agg.add(result(0.95f))
        assertEquals(ThreatLevel.DANGER, r.threatLevel)
    }

    @Test
    fun `three consecutive high scores returns WARNING`() {
        val agg = DetectionAggregator()
        agg.add(result(0.75f))
        agg.add(result(0.80f))
        val r = agg.add(result(0.72f))
        assertEquals(ThreatLevel.WARNING, r.threatLevel)
        assertEquals(3, r.consecutiveHighCount)
    }

    @Test
    fun `average above 0_6 returns CAUTION`() {
        val agg = DetectionAggregator()
        agg.add(result(0.65f))
        agg.add(result(0.55f))
        val r = agg.add(result(0.68f))
        assertEquals(ThreatLevel.CAUTION, r.threatLevel)
    }

    @Test
    fun `clear resets state`() {
        val agg = DetectionAggregator()
        agg.add(result(0.95f))
        agg.clear()
        val r = agg.add(result(0.1f))
        assertEquals(ThreatLevel.SAFE, r.threatLevel)
    }
}
