package com.deepvoiceguard.app.inference

/** Deepfake 음성 탐지 결과. */
data class DetectionResult(
    val fakeScore: Float,     // 0.0 ~ 1.0 (softmax output[1])
    val realScore: Float,     // 0.0 ~ 1.0 (softmax output[0])
    val confidence: Float,    // max(fakeScore, realScore)
    val latencyMs: Long,
)

/** 모델 정보. */
data class ModelInfo(
    val name: String,
    val version: String,
    val type: String,  // "on_device" or "server"
)

/** 추론 엔진 인터페이스. On-Device와 Server 구현체가 이 인터페이스를 구현한다. */
interface InferenceEngine {
    /** 오디오 세그먼트에 대해 deepfake 탐지 추론을 수행한다. */
    suspend fun detect(audio: FloatArray): DetectionResult

    /** 엔진이 준비 상태인지 반환. */
    fun isReady(): Boolean

    /** 모델 정보를 반환. */
    fun getModelInfo(): ModelInfo

    /** 리소스 해제. */
    fun close()
}
