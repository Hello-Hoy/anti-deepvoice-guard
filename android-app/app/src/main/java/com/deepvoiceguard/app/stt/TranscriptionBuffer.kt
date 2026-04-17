package com.deepvoiceguard.app.stt

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * 60초 슬라이딩 윈도우 전사 텍스트 버퍼.
 * 피싱 키워드 탐지를 위해 최근 전사 텍스트를 유지한다.
 */
class TranscriptionBuffer(
    private val windowDurationMs: Long = 60_000L,
) {

    private data class Entry(val text: String, val timestampMs: Long)

    private val entries = mutableListOf<Entry>()
    private val mutex = Mutex()

    /** 새 전사 텍스트 추가. */
    suspend fun append(text: String, timestampMs: Long = System.currentTimeMillis()) {
        if (text.isBlank()) return
        mutex.withLock {
            entries.add(Entry(text.trim(), timestampMs))
            pruneOld(timestampMs)
        }
    }

    /** 윈도우 내의 전체 전사 텍스트 반환 (공백 구분 연결). */
    suspend fun getWindowText(): String = mutex.withLock {
        pruneOld(System.currentTimeMillis())
        entries.joinToString(" ") { it.text }
    }

    /** 윈도우 내의 전사 항목 목록 반환. */
    suspend fun getEntries(): List<Pair<String, Long>> = mutex.withLock {
        pruneOld(System.currentTimeMillis())
        entries.map { it.text to it.timestampMs }
    }

    /** 버퍼 초기화. */
    suspend fun clear() = mutex.withLock {
        entries.clear()
    }

    private fun pruneOld(now: Long) {
        val cutoff = now - windowDurationMs
        entries.removeAll { it.timestampMs < cutoff }
    }
}
