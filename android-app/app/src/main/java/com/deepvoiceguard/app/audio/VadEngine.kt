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

    // Silero VAD v5 내부 상태 — (2, 1, 128) 단일 텐서로 통합.
    private var state0 = FloatArray(2 * 1 * 128)

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
        // Silero VAD v5 signature: input(1, 512), state(2, 1, 128), sr(scalar).
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(frame),
            longArrayOf(1, WINDOW_SIZE.toLong())
        )
        val srTensor = OnnxTensor.createTensor(env, longArrayOf(SAMPLE_RATE))
        val stateTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(state0),
            longArrayOf(2, 1, 128)
        )

        val inputs = mapOf(
            "input" to inputTensor,
            "state" to stateTensor,
            "sr" to srTensor,
        )

        val results = session.run(inputs)

        val output = (results[0].value as Array<*>)[0] as FloatArray
        val probability = output[0]

        // stateN (2, 1, 128) → state0 flatten.
        val stateN = results[1].value
        if (stateN is Array<*>) {
            var idx = 0
            for (layer in stateN) {
                if (layer is Array<*>) {
                    for (batch in layer) {
                        if (batch is FloatArray) {
                            batch.copyInto(state0, idx)
                            idx += batch.size
                        }
                    }
                }
            }
        }

        results.close()
        inputTensor.close()
        srTensor.close()
        stateTensor.close()

        return probability
    }

    fun reset() {
        state = State.SILENCE
        frameCount = 0
        totalFrames = 0
        state0.fill(0f)
    }

    fun currentState(): State = state

    fun close() {
        session.close()
    }
}
