package com.deepvoiceguard.app.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * Silero VAD (Voice Activity Detection) 엔진.
 * ONNX Runtime으로 512 samples (32ms @ 16kHz) 단위로 음성 구간을 감지한다.
 */
class VadEngine(context: Context, modelFileName: String = "silero_vad.onnx") {

    companion object {
        private const val SAMPLE_RATE = 16000L
        private const val WINDOW_SIZE = 512
        private const val SPEECH_THRESHOLD = 0.5f
        private const val SILENCE_THRESHOLD = 0.3f
        private const val SPEECH_FRAMES_REQUIRED = 8    // ~256ms
        private const val SILENCE_FRAMES_REQUIRED = 15  // ~480ms
    }

    enum class State { SILENCE, SPEECH }

    interface Listener {
        fun onSpeechStart(timestampMs: Long)
        fun onSpeechEnd(timestampMs: Long, durationMs: Long)
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var state = State.SILENCE
    private var frameCount = 0
    private var speechStartMs = 0L
    private var totalFrames = 0L

    // Silero VAD 내부 상태 (h, c)
    private var h = FloatArray(2 * 1 * 64)  // (2, 1, 64)
    private var c = FloatArray(2 * 1 * 64)

    var listener: Listener? = null

    init {
        val modelBytes = context.assets.open(modelFileName).readBytes()
        session = env.createSession(modelBytes)
    }

    /**
     * 512 samples 프레임을 처리하여 음성 확률을 반환한다.
     * 상태 머신을 업데이트하고, 상태 전환 시 리스너를 호출한다.
     */
    fun process(frame: FloatArray): Float {
        require(frame.size == WINDOW_SIZE) { "Frame must be $WINDOW_SIZE samples" }

        val probability = runInference(frame)
        totalFrames++
        val currentMs = (totalFrames * WINDOW_SIZE * 1000) / SAMPLE_RATE

        when (state) {
            State.SILENCE -> {
                if (probability > SPEECH_THRESHOLD) {
                    frameCount++
                    if (frameCount >= SPEECH_FRAMES_REQUIRED) {
                        state = State.SPEECH
                        frameCount = 0
                        speechStartMs = currentMs
                        listener?.onSpeechStart(currentMs)
                    }
                } else {
                    frameCount = 0
                }
            }
            State.SPEECH -> {
                if (probability < SILENCE_THRESHOLD) {
                    frameCount++
                    if (frameCount >= SILENCE_FRAMES_REQUIRED) {
                        state = State.SILENCE
                        frameCount = 0
                        val duration = currentMs - speechStartMs
                        listener?.onSpeechEnd(currentMs, duration)
                    }
                } else {
                    frameCount = 0
                }
            }
        }

        return probability
    }

    private fun runInference(frame: FloatArray): Float {
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(frame),
            longArrayOf(1, WINDOW_SIZE.toLong())
        )
        val srTensor = OnnxTensor.createTensor(env, longArrayOf(SAMPLE_RATE))
        val hTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(h),
            longArrayOf(2, 1, 64)
        )
        val cTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(c),
            longArrayOf(2, 1, 64)
        )

        val inputs = mapOf(
            "input" to inputTensor,
            "sr" to srTensor,
            "h" to hTensor,
            "c" to cTensor,
        )

        val results = session.run(inputs)

        val output = (results[0].value as Array<*>)[0] as FloatArray
        val probability = output[0]

        // 내부 상태 업데이트
        val hn = results[1].value
        val cn = results[2].value
        if (hn is Array<*>) {
            // flatten to h array
            var idx = 0
            for (layer in hn) {
                if (layer is Array<*>) {
                    for (batch in layer) {
                        if (batch is FloatArray) {
                            batch.copyInto(h, idx)
                            idx += batch.size
                        }
                    }
                }
            }
        }
        if (cn is Array<*>) {
            var idx = 0
            for (layer in cn) {
                if (layer is Array<*>) {
                    for (batch in layer) {
                        if (batch is FloatArray) {
                            batch.copyInto(c, idx)
                            idx += batch.size
                        }
                    }
                }
            }
        }

        results.close()
        inputTensor.close()
        srTensor.close()
        hTensor.close()
        cTensor.close()

        return probability
    }

    fun reset() {
        state = State.SILENCE
        frameCount = 0
        totalFrames = 0
        h.fill(0f)
        c.fill(0f)
    }

    fun currentState(): State = state

    fun close() {
        session.close()
    }
}
