package com.deepvoiceguard.app.inference

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.util.concurrent.TimeUnit

/** 서버 응답 DTO. */
data class ServerDetectionResponse(
    val fake_score: Float,
    val real_score: Float,
    val confidence: Float,
    val model_version: String,
    val inference_ms: Float,
)

/** Retrofit API 인터페이스. */
interface DetectionApi {
    @Multipart
    @POST("api/v1/detect")
    suspend fun detect(@Part audio: MultipartBody.Part): ServerDetectionResponse
}

/**
 * 서버 기반 추론 엔진.
 * FastAPI 서버에 오디오를 전송하여 추론 결과를 받는다.
 */
class ServerEngine(
    baseUrl: String,
    private val fallback: InferenceEngine? = null,
) : InferenceEngine {

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    private val api: DetectionApi = Retrofit.Builder()
        .baseUrl(baseUrl)
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(DetectionApi::class.java)

    private var ready = true

    override suspend fun detect(audio: FloatArray): DetectionResult {
        val startTime = System.nanoTime()

        try {
            val wavBytes = floatArrayToWavBytes(audio)
            val requestBody = wavBytes.toRequestBody("audio/wav".toMediaType())
            val part = MultipartBody.Part.createFormData("audio", "segment.wav", requestBody)

            val response = withContext(Dispatchers.IO) {
                api.detect(part)
            }

            val elapsedMs = (System.nanoTime() - startTime) / 1_000_000
            return DetectionResult(
                fakeScore = response.fake_score,
                realScore = response.real_score,
                confidence = response.confidence,
                latencyMs = elapsedMs,
            )
        } catch (e: Exception) {
            // 서버 실패 시 On-Device 폴백
            if (fallback != null && fallback.isReady()) {
                return fallback.detect(audio)
            }
            throw e
        }
    }

    override fun isReady(): Boolean = ready

    override fun getModelInfo(): ModelInfo = ModelInfo(
        name = "AASIST-Ensemble",
        version = "1.0",
        type = "server",
    )

    override fun close() {
        ready = false
    }

    /** Float 배열을 16-bit PCM WAV 바이트로 변환. */
    private fun floatArrayToWavBytes(audio: FloatArray): ByteArray {
        val sampleRate = 16000
        val bitsPerSample = 16
        val numChannels = 1
        val dataSize = audio.size * 2

        val baos = ByteArrayOutputStream()
        val dos = DataOutputStream(baos)

        // WAV header
        dos.writeBytes("RIFF")
        dos.writeIntLE(36 + dataSize)
        dos.writeBytes("WAVE")
        dos.writeBytes("fmt ")
        dos.writeIntLE(16)           // subchunk size
        dos.writeShortLE(1)          // PCM
        dos.writeShortLE(numChannels)
        dos.writeIntLE(sampleRate)
        dos.writeIntLE(sampleRate * numChannels * bitsPerSample / 8)
        dos.writeShortLE(numChannels * bitsPerSample / 8)
        dos.writeShortLE(bitsPerSample)
        dos.writeBytes("data")
        dos.writeIntLE(dataSize)

        // PCM data
        for (sample in audio) {
            val clamped = sample.coerceIn(-1f, 1f)
            val intVal = (clamped * 32767).toInt().toShort()
            dos.writeShortLE(intVal.toInt())
        }

        dos.flush()
        return baos.toByteArray()
    }

    private fun DataOutputStream.writeIntLE(value: Int) {
        write(value and 0xFF)
        write((value shr 8) and 0xFF)
        write((value shr 16) and 0xFF)
        write((value shr 24) and 0xFF)
    }

    private fun DataOutputStream.writeShortLE(value: Int) {
        write(value and 0xFF)
        write((value shr 8) and 0xFF)
    }
}
