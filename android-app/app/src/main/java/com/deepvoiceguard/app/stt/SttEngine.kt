package com.deepvoiceguard.app.stt

import kotlinx.coroutines.flow.StateFlow

/** STT 엔진 상태. */
enum class SttStatus {
    UNAVAILABLE,    // 기기에서 사용 불가
    CONSENT_NEEDED, // 사용자 동의 필요
    READY,          // 준비 완료 (미시작)
    LISTENING,      // 인식 중
    PAUSED,         // 일시 중지 (backoff 재시작 대기)
    ERROR,          // 오류 발생
}

/** STT 전사 결과. */
data class TranscriptionEvent(
    val text: String,
    val isFinal: Boolean,
    val timestampMs: Long = System.currentTimeMillis(),
)

/**
 * STT(Speech-to-Text) 엔진 인터페이스.
 * Capability-gated: 기기에서 사용 불가능하면 [status]가 [SttStatus.UNAVAILABLE]로 유지.
 */
interface SttEngine {

    /** 현재 STT 상태. */
    val status: StateFlow<SttStatus>

    /** 누적 전사 텍스트 (60초 슬라이딩 윈도우). */
    val transcription: StateFlow<String>

    /** 최신 전사 이벤트 (partial/final). */
    val latestEvent: StateFlow<TranscriptionEvent?>

    /** STT가 현재 기기에서 사용 가능한지 확인. */
    fun isAvailable(): Boolean

    /** 음성 인식 시작. [SttStatus.CONSENT_NEEDED]이면 시작하지 않음. */
    fun start()

    /**
     * 음성 인식 중지 — main-thread cleanup이 완료될 때까지 await.
     * off-main 호출자도 return 시점에 recognizer가 실제로 stop된 상태를 보장.
     */
    suspend fun stop()

    /**
     * 리소스 해제 — main-thread cleanup이 완료될 때까지 await.
     * off-main 호출자도 return 시점에 recognizer가 실제로 destroy된 상태를 보장.
     */
    suspend fun destroy()

    /**
     * In-flight onResults worker(전사 publish + buffer append)가 모두 완료될 때까지 대기.
     * shutdown drain에서 stop() 직후 호출하여 stale work가 destroy 후에 publish하지 않도록 보장.
     */
    suspend fun awaitResultJobsIdle()

    /**
     * 최종 transcription을 받았을 때 publish worker 안에서 직접 호출되는 listener.
     * StateFlow conflation 회피용 — phishing analysis가 collector dispatch 시점에
     * 의존하지 않고, publish와 같은 worker에서 동기적으로 트리거된다.
     * `null`이면 비활성화. listener는 suspend.
     */
    fun setOnFinalTranscriptionListener(listener: (suspend (String) -> Unit)?)

    /**
     * **H137**: 내부 transcription buffer (60s sliding window)와 publish된 누적 transcript를
     * 즉시 클리어. phishing toggle 전환 시 disabled 동안 누적된 텍스트가 ON 후 retroactive
     * 분석되는 것을 차단하기 위함.
     */
    suspend fun resetTranscriptionWindow()
}
