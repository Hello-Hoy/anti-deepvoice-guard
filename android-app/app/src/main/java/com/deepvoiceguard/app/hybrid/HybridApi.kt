package com.deepvoiceguard.app.hybrid

import com.google.gson.annotations.SerializedName
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.POST

/** 하이브리드 서버 v2 API Retrofit 인터페이스. */
interface HybridApi {

    /** 익명 메타데이터 전송. */
    @POST("api/v2/telemetry")
    suspend fun sendTelemetry(
        @Body event: TelemetryEvent,
        @Header("X-API-Version") version: String = "2",
        @Header("X-API-Key") apiKey: String,
    ): Response<Unit>

    /** 피싱 규칙 동기화. */
    @GET("api/v2/phishing/rules")
    suspend fun getPhishingRules(
        @Header("If-None-Match") etag: String? = null,
        @Header("X-API-Version") version: String = "2",
    ): Response<PhishingRulesResponse>

    /** 모델 최신 버전 확인. */
    @GET("api/v2/model/latest")
    suspend fun getLatestModel(
        @Header("X-API-Version") version: String = "2",
    ): Response<ModelVersionResponse>
}

/**
 * 텔레메트리 이벤트. 서버 Pydantic 모델(snake_case)과 정합.
 * `@SerializedName`으로 Gson 직렬화 시 필드명을 명시한다.
 */
data class TelemetryEvent(
    @SerializedName("deepfake_score") val deepfakeScore: Float,
    @SerializedName("phishing_score") val phishingScore: Float,
    @SerializedName("phishing_keywords") val phishingKeywords: List<String>,
    @SerializedName("combined_threat_level") val combinedThreatLevel: String,
    @SerializedName("model_version") val modelVersion: String,
    @SerializedName("inference_mode") val inferenceMode: String,
    @SerializedName("stt_available") val sttAvailable: Boolean,
)

/** 피싱 규칙 응답. */
data class PhishingRulesResponse(
    @SerializedName("version") val version: Int,
    @SerializedName("checksum") val checksum: String,
    @SerializedName("keywords") val keywords: List<Any>,
    @SerializedName("phrases") val phrases: List<Any>,
)

/** 모델 버전 응답. */
data class ModelVersionResponse(
    @SerializedName("version") val version: String,
    @SerializedName("checksum") val checksum: String,
    @SerializedName("download_url") val downloadUrl: String?,
)
