package com.deepvoiceguard.app.hybrid

import android.util.Log
import com.deepvoiceguard.app.inference.CombinedThreatResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

/**
 * 하이브리드 데이터 수집기.
 * 동의 게이트: consentLevel이 "none"이면 아무것도 전송하지 않음.
 * 배치 전송 + bounded 오프라인 큐 (최대 [MAX_QUEUE_SIZE]).
 */
class HybridDataCollector(
    private val serverUrl: String,
    private val scope: CoroutineScope,
    private val consentLevelProvider: () -> String,
    private val apiKeyProvider: () -> String,
) {

    companion object {
        private const val TAG = "HybridDataCollector"
        private const val MODEL_VERSION = "aasist-v5.1"
        private const val MAX_QUEUE_SIZE = 200
        private const val BATCH_SIZE = 5
    }

    private val api: HybridApi? = try {
        Retrofit.Builder()
            .baseUrl(serverUrl.trimEnd('/') + "/")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(HybridApi::class.java)
    } catch (e: Exception) {
        Log.w(TAG, "Failed to create HybridApi: ${e.message}")
        null
    }

    private val queueMutex = Mutex()
    private val pendingEvents = ArrayDeque<TelemetryEvent>()

    /** 탐지 결과를 수집한다. 동의 레벨에 따라 전송 여부 결정. */
    fun collect(result: CombinedThreatResult) {
        val consentLevel = consentLevelProvider()
        if (consentLevel == "none") return

        val event = TelemetryEvent(
            deepfakeScore = result.deepfakeResult?.averageFakeScore ?: 0f,
            phishingScore = result.phishingScore,
            phishingKeywords = result.matchedKeywords.map { it.keyword },
            combinedThreatLevel = result.combinedThreatLevel.name,
            modelVersion = MODEL_VERSION,
            inferenceMode = "on_device",
            sttAvailable = result.sttAvailable,
        )

        scope.launch(Dispatchers.IO) {
            enqueue(event)
            val size = queueMutex.withLock { pendingEvents.size }
            if (size >= BATCH_SIZE) flushAsync()
        }
    }

    /** 보류 중인 이벤트를 서버로 전송. */
    fun flush() {
        flushAsync()
    }

    private suspend fun enqueue(event: TelemetryEvent) {
        queueMutex.withLock {
            // Bounded: 가장 오래된 이벤트부터 드롭.
            while (pendingEvents.size >= MAX_QUEUE_SIZE) {
                pendingEvents.removeFirst()
            }
            pendingEvents.addLast(event)
        }
    }

    private fun flushAsync() {
        val api = this.api ?: return
        val apiKey = apiKeyProvider().ifBlank {
            Log.d(TAG, "No API key available, skip flush")
            return
        }

        scope.launch(Dispatchers.IO) {
            val eventsToSend = queueMutex.withLock {
                if (pendingEvents.isEmpty()) return@withLock emptyList()
                val events = pendingEvents.toList()
                pendingEvents.clear()
                events
            }
            if (eventsToSend.isEmpty()) return@launch

            val failed = mutableListOf<TelemetryEvent>()
            for (event in eventsToSend) {
                try {
                    val response = api.sendTelemetry(event, apiKey = apiKey)
                    if (!response.isSuccessful) {
                        Log.w(TAG, "Telemetry send failed: ${response.code()}")
                        // 4xx는 재시도 무의미 → 드롭, 5xx만 재큐.
                        if (response.code() in 500..599) failed.add(event)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Telemetry send error: ${e.message}")
                    failed.add(event)
                }
            }

            if (failed.isNotEmpty()) {
                queueMutex.withLock {
                    // Bounded 재큐.
                    for (event in failed) {
                        while (pendingEvents.size >= MAX_QUEUE_SIZE) {
                            pendingEvents.removeFirst()
                        }
                        pendingEvents.addLast(event)
                    }
                }
            }
        }
    }
}
