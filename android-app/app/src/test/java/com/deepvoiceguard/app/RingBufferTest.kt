package com.deepvoiceguard.app

import com.deepvoiceguard.app.audio.RingBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class RingBufferTest {

    @Test
    fun `write and read back basic samples`() {
        val buffer = RingBuffer(capacitySamples = 10)
        buffer.write(floatArrayOf(1f, 2f, 3f))
        val result = buffer.getLast(3)
        assertArrayEquals(floatArrayOf(1f, 2f, 3f), result, 0.001f)
    }

    @Test
    fun `wraps around when full`() {
        val buffer = RingBuffer(capacitySamples = 5)
        buffer.write(floatArrayOf(1f, 2f, 3f, 4f, 5f))
        buffer.write(floatArrayOf(6f, 7f))

        val result = buffer.getLast(5)
        assertArrayEquals(floatArrayOf(3f, 4f, 5f, 6f, 7f), result, 0.001f)
    }

    @Test
    fun `getLast returns less than requested if not enough data`() {
        val buffer = RingBuffer(capacitySamples = 100)
        buffer.write(floatArrayOf(1f, 2f))
        val result = buffer.getLast(10)
        assertEquals(2, result.size)
    }

    @Test
    fun `getSegment returns correct range`() {
        val buffer = RingBuffer(capacitySamples = 20)
        buffer.write(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f))
        val result = buffer.getSegment(samplesAgo = 5, length = 3)
        assertArrayEquals(floatArrayOf(6f, 7f, 8f), result, 0.001f)
    }

    @Test
    fun `totalSamplesWritten tracks correctly`() {
        val buffer = RingBuffer(capacitySamples = 5)
        buffer.write(floatArrayOf(1f, 2f, 3f))
        buffer.write(floatArrayOf(4f, 5f))
        assertEquals(5L, buffer.totalSamplesWritten())
    }

    @Test
    fun `clear resets buffer`() {
        val buffer = RingBuffer(capacitySamples = 10)
        buffer.write(floatArrayOf(1f, 2f, 3f))
        buffer.clear()
        assertEquals(0L, buffer.totalSamplesWritten())
        assertEquals(0, buffer.getLast(10).size)
    }
}
