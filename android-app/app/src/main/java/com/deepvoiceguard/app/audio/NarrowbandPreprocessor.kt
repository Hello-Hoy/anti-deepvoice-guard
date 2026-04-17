package com.deepvoiceguard.app.audio

import androidx.annotation.VisibleForTesting
import kotlin.math.abs
import kotlin.math.sqrt

class NarrowbandPreprocessor {

    fun process(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return FloatArray(0)

        val filtered = filterOnly(samples)
        normalizeInPlace(filtered)
        return filtered
    }

    @VisibleForTesting
    internal fun filterOnly(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return FloatArray(0)

        val output = samples.copyOf()
        applyCascade(output)
        reverseInPlace(output)
        applyCascade(output)
        reverseInPlace(output)
        return output
    }

    private fun applyCascade(signal: FloatArray) {
        for (section in SOS_COEFFS) {
            val b0 = section[0]
            val b1 = section[1]
            val b2 = section[2]
            val a1 = section[4]
            val a2 = section[5]

            var state1 = 0.0
            var state2 = 0.0

            for (index in signal.indices) {
                val input = signal[index].toDouble()
                val output = (b0 * input) + state1
                state1 = (b1 * input) - (a1 * output) + state2
                state2 = (b2 * input) - (a2 * output)
                signal[index] = output.toFloat()
            }
        }
    }

    private fun normalizeInPlace(signal: FloatArray) {
        var sumSquares = 0.0
        var peak = 0.0

        for (sample in signal) {
            val value = sample.toDouble()
            sumSquares += value * value
            val magnitude = abs(value)
            if (magnitude > peak) {
                peak = magnitude
            }
        }

        val rms = sqrt(sumSquares / signal.size)
        var scale = if (rms > RMS_EPSILON) TARGET_RMS / rms else 1.0
        if (peak > 0.0 && peak * scale > PEAK_LIMIT) {
            scale = PEAK_LIMIT / peak
        }

        for (index in signal.indices) {
            val scaled = signal[index].toDouble() * scale
            signal[index] = scaled.coerceIn(-1.0, 1.0).toFloat()
        }
    }

    private fun reverseInPlace(signal: FloatArray) {
        var left = 0
        var right = signal.lastIndex
        while (left < right) {
            val temp = signal[left]
            signal[left] = signal[right]
            signal[right] = temp
            left++
            right--
        }
    }

    companion object {
        private const val TARGET_RMS = 0.03162277660168379
        private const val PEAK_LIMIT = 0.31622776601683794
        private const val RMS_EPSILON = 1e-6

        private val SOS_COEFFS: Array<DoubleArray> = arrayOf(
            doubleArrayOf(1.21461357580271933e-03, 2.42922715160543866e-03, 1.21461357580271933e-03, 1.00000000000000000e+00, -4.59589672689680651e-01, 6.27265322017917792e-02),
            doubleArrayOf(1.00000000000000000e+00, 2.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00, -4.78996540700302065e-01, 1.49155246474870828e-01),
            doubleArrayOf(1.00000000000000000e+00, 2.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00, -5.33493786315158425e-01, 3.43614828340291389e-01),
            doubleArrayOf(1.00000000000000000e+00, 2.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00, -6.57001069716789843e-01, 7.09199400930942581e-01),
            doubleArrayOf(1.00000000000000000e+00, -2.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00, -1.91999698823906861e+00, 9.21666405486491813e-01),
            doubleArrayOf(1.00000000000000000e+00, -2.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00, -1.93391985224181151e+00, 9.35538394258280670e-01),
            doubleArrayOf(1.00000000000000000e+00, -2.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00, -1.95681447320801305e+00, 9.58376358055870292e-01),
            doubleArrayOf(1.00000000000000000e+00, -2.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00, -1.98406740411534166e+00, 9.85604373136429901e-01),
        )
    }
}
