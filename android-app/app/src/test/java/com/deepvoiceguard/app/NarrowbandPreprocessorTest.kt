package com.deepvoiceguard.app

import com.deepvoiceguard.app.audio.NarrowbandPreprocessor
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NarrowbandPreprocessorTest {

    private val preprocessor = NarrowbandPreprocessor()

    @Test
    fun passbandAttenuation_1kHz() {
        val input = sineWave(frequencyHz = 1_000.0, amplitude = 0.5f, durationSeconds = 2.0)

        val filtered = preprocessor.filterOnly(input)
        val gainDb = gainDb(input, filtered, startSample = SAMPLE_RATE / 2, length = SAMPLE_RATE)

        assertTrue("Expected 1 kHz gain within +/-1.5 dB but was $gainDb dB", gainDb in -1.5..1.5)
    }

    @Test
    fun passbandAttenuation_3kHz_edge() {
        val input = sineWave(frequencyHz = 3_000.0, amplitude = 0.5f, durationSeconds = 2.0)

        val filtered = preprocessor.filterOnly(input)
        val gainDb = gainDb(input, filtered, startSample = SAMPLE_RATE / 2, length = SAMPLE_RATE)

        assertTrue("Expected 3 kHz gain near -6 dB but was $gainDb dB", gainDb in -8.0..-4.0)
    }

    @Test
    fun stopbandRejection_5kHz() {
        val input = sineWave(frequencyHz = 5_000.0, amplitude = 0.5f, durationSeconds = 2.0)

        val filtered = preprocessor.filterOnly(input)
        val gainDb = gainDb(input, filtered, startSample = SAMPLE_RATE / 2, length = SAMPLE_RATE)

        assertTrue("Expected 5 kHz rejection below -40 dB but was $gainDb dB", gainDb < -40.0)
    }

    @Test
    fun stopbandRejection_100Hz_edge() {
        val input = sineWave(frequencyHz = 100.0, amplitude = 0.5f, durationSeconds = 2.0)

        val filtered = preprocessor.filterOnly(input)
        val gainDb = gainDb(input, filtered, startSample = SAMPLE_RATE / 2, length = SAMPLE_RATE)

        assertTrue("Expected 100 Hz gain near -6 dB but was $gainDb dB", gainDb in -8.0..-4.0)
    }

    @Test
    fun dcBlocks() {
        val input = FloatArray(SAMPLE_RATE * 2) { 0.5f }

        val filtered = preprocessor.filterOnly(input)
        val rms = rms(filtered, startSample = SAMPLE_RATE / 2, length = SAMPLE_RATE)

        assertTrue("Expected DC RMS < 1e-4 but was $rms", rms < 1e-4)
    }

    @Test
    fun rmsNormalization() {
        val input = sineWave(frequencyHz = 1_000.0, amplitude = 0.2f, durationSeconds = 2.0)

        val processed = preprocessor.process(input)
        val rms = rms(processed)

        assertTrue(
            "Expected RMS near $TARGET_RMS but was $rms",
            rms in (TARGET_RMS * 0.9)..(TARGET_RMS * 1.1),
        )
    }

    @Test
    fun peakClamp() {
        val input = FloatArray(SAMPLE_RATE * 2).also { samples ->
            samples[samples.size / 2] = 0.9f
        }

        val processed = preprocessor.process(input)
        val peak = processed.fold(0.0) { current, sample -> max(current, abs(sample.toDouble())) }

        assertTrue("Expected peak <= $PEAK_LIMIT but was $peak", peak <= PEAK_LIMIT + 1e-4)
    }

    @Test
    fun idempotency() {
        val input = sineWave(frequencyHz = 1_000.0, amplitude = 0.5f, durationSeconds = 2.0)

        val once = preprocessor.process(input)
        val twice = preprocessor.process(once)
        val rmsChangeDb = 20.0 * log10((rms(twice) + EPSILON) / (rms(once) + EPSILON))

        assertTrue("Expected RMS change < 2 dB but was $rmsChangeDb dB", abs(rmsChangeDb) < 2.0)
    }

    @Test
    fun doesNotMutateInput() {
        val input = sineWave(frequencyHz = 1_000.0, amplitude = 0.5f, durationSeconds = 2.0)
        val original = input.copyOf()

        preprocessor.process(input)

        assertArrayEquals(original, input, 0f)
    }

    @Test
    fun emptyInput() {
        val result = preprocessor.process(FloatArray(0))

        assertEquals(0, result.size)
    }

    private fun sineWave(
        frequencyHz: Double,
        amplitude: Float,
        durationSeconds: Double,
    ): FloatArray {
        val count = (durationSeconds * SAMPLE_RATE).toInt()
        return FloatArray(count) { index ->
            (amplitude * sin(2.0 * PI * frequencyHz * index / SAMPLE_RATE)).toFloat()
        }
    }

    private fun gainDb(
        input: FloatArray,
        output: FloatArray,
        startSample: Int,
        length: Int,
    ): Double {
        val inputRms = rms(input, startSample, length)
        val outputRms = rms(output, startSample, length)
        return 20.0 * log10((outputRms + EPSILON) / (inputRms + EPSILON))
    }

    private fun rms(
        samples: FloatArray,
        startSample: Int = 0,
        length: Int = samples.size,
    ): Double {
        if (length == 0) return 0.0
        var sumSquares = 0.0
        val end = startSample + length
        for (index in startSample until end) {
            val sample = samples[index].toDouble()
            sumSquares += sample * sample
        }
        return sqrt(sumSquares / length)
    }

    companion object {
        private const val SAMPLE_RATE = 16_000
        private const val TARGET_RMS = 0.03162277660168379
        private const val PEAK_LIMIT = 0.31622776601683794
        private const val EPSILON = 1e-12
    }
}
