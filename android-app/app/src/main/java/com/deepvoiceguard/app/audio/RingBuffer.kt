package com.deepvoiceguard.app.audio

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Thread-safe 순환 오디오 버퍼.
 * 최근 [capacitySamples] 샘플을 유지하며, 오래된 데이터는 자동으로 덮어씌워진다.
 *
 * @param capacitySamples 버퍼 크기 (기본 30초 = 480,000 samples @ 16kHz)
 */
class RingBuffer(private val capacitySamples: Int = 30 * 16000) {

    private val buffer = FloatArray(capacitySamples)
    private var writePos = 0
    private var totalWritten = 0L
    private val lock = ReentrantLock()

    /** 샘플 배열을 버퍼에 추가. */
    fun write(samples: FloatArray) = lock.withLock {
        for (sample in samples) {
            buffer[writePos] = sample
            writePos = (writePos + 1) % capacitySamples
        }
        totalWritten += samples.size
    }

    /** 최근 [length] 샘플을 추출. 버퍼에 충분한 데이터가 없으면 가용한 만큼 반환. */
    fun getLast(length: Int): FloatArray = lock.withLock {
        val available = minOf(length, totalWritten.toInt(), capacitySamples)
        if (available == 0) return@withLock FloatArray(0)

        val result = FloatArray(available)
        var readPos = (writePos - available + capacitySamples) % capacitySamples
        for (i in 0 until available) {
            result[i] = buffer[readPos]
            readPos = (readPos + 1) % capacitySamples
        }
        return@withLock result
    }

    /**
     * 특정 시점부터 [length] 샘플을 추출.
     * [samplesAgo]는 현재 writePos로부터 몇 샘플 전인지를 나타낸다.
     */
    fun getSegment(samplesAgo: Int, length: Int): FloatArray = lock.withLock {
        val maxAgo = minOf(samplesAgo, totalWritten.toInt(), capacitySamples)
        val actualLength = minOf(length, maxAgo)
        if (actualLength == 0) return@withLock FloatArray(0)

        val result = FloatArray(actualLength)
        var readPos = (writePos - maxAgo + capacitySamples) % capacitySamples
        for (i in 0 until actualLength) {
            result[i] = buffer[readPos]
            readPos = (readPos + 1) % capacitySamples
        }
        return@withLock result
    }

    /** 총 기록된 샘플 수. */
    fun totalSamplesWritten(): Long = lock.withLock { totalWritten }

    /** 버퍼 초기화. */
    fun clear() = lock.withLock {
        buffer.fill(0f)
        writePos = 0
        totalWritten = 0
    }
}
