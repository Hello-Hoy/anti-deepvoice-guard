package com.deepvoiceguard.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * On-Device AASIST 추론 엔진.
 * ONNX Runtime으로 AASIST-L 모델을 실행한다.
 */
class OnDeviceEngine(
    context: Context,
    modelFileName: String = "aasist_l.onnx",
) : InferenceEngine {

    companion object {
        private const val NB_SAMP = 80_000  // 5초 @ 16kHz
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var ready = false

    init {
        val modelBytes = context.assets.open(modelFileName).readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, opts)
        ready = true
    }

    override suspend fun detect(audio: FloatArray): DetectionResult {
        val padded = padOrCrop(audio, NB_SAMP)

        val startTime = System.nanoTime()

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(padded),
            longArrayOf(1, NB_SAMP.toLong()),
        )

        val results = session.run(mapOf("audio" to inputTensor))
        val probabilities = (results[0].value as Array<FloatArray>)[0]

        inputTensor.close()
        results.close()

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

        val realScore = probabilities[0]
        val fakeScore = probabilities[1]

        return DetectionResult(
            fakeScore = fakeScore,
            realScore = realScore,
            confidence = maxOf(fakeScore, realScore),
            latencyMs = elapsedMs,
        )
    }

    override fun isReady(): Boolean = ready

    override fun getModelInfo(): ModelInfo = ModelInfo(
        name = "AASIST-L",
        version = "1.0",
        type = "on_device",
    )

    override fun close() {
        ready = false
        session.close()
    }

    private fun padOrCrop(audio: FloatArray, targetLen: Int): FloatArray {
        if (audio.size >= targetLen) {
            return audio.copyOfRange(0, targetLen)
        }
        val result = FloatArray(targetLen)
        var written = 0
        while (written < targetLen) {
            val toCopy = minOf(audio.size, targetLen - written)
            audio.copyInto(result, written, 0, toCopy)
            written += toCopy
        }
        return result
    }
}
