package com.deepvoiceguard.app.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import java.nio.LongBuffer

/**
 * Silero VAD (Voice Activity Detection) 엔진.
 * ONNX Runtime으로 512 samples (32ms @ 16kHz) 단위로 음성 구간을 감지한다.
 */
class VadEngine(context: Context, modelFileName: String = "silero_vad.onnx") {

    companion object {
        private const val SAMPLE_RATE = 16000L
        private const val WINDOW_SIZE = 512
        // **Silero VAD v5는 이전 프레임 context 64 샘플을 앞에 붙인 (1, 64+512=576) 을 입력으로 요구.**
        // 공식 Python wrapper (silero_vad/utils_vad.py OnnxWrapper.__call__)에서 확인됨.
        // context 없이 raw 512 samples만 넣으면 모델 prob이 모든 입력에 대해 ~0.001로 고정.
        private const val CONTEXT_SIZE = 64
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

    // Silero v5 context buffer — 직전 프레임의 마지막 64 샘플. 다음 호출 앞에 concat한다.
    private var context = FloatArray(CONTEXT_SIZE)

    // [DIAG] 첫 inference 결과의 실제 구조를 1회 로깅하기 위한 플래그.
    private var inferenceDiagLogged = false

    var listener: Listener? = null

    init {
        val modelBytes = context.assets.open(modelFileName).readBytes()
        session = env.createSession(modelBytes)
        // [DIAG] 실제 모델 signature 로깅 — 진단 후 제거 예정.
        for ((name, info) in session.inputInfo) {
            val ti = info.info as? TensorInfo
            Log.i("VadEngine", "input  $name: shape=${ti?.shape?.toList()} dtype=${ti?.type}")
        }
        for ((name, info) in session.outputInfo) {
            val ti = info.info as? TensorInfo
            Log.i("VadEngine", "output $name: shape=${ti?.shape?.toList()} dtype=${ti?.type}")
        }
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
        // Silero VAD v5 signature: input(1, 64+512=576), state(2, 1, 128), sr(scalar).
        // 직전 프레임의 마지막 64 샘플을 context로 앞에 concat해서 넘겨야 모델이 제대로 동작.
        val concatenated = FloatArray(CONTEXT_SIZE + WINDOW_SIZE)
        context.copyInto(concatenated, 0)
        frame.copyInto(concatenated, CONTEXT_SIZE)

        val inputTensor = OnnxTensor.createTensor(env, arrayOf(concatenated))

        // sr은 rank-0 int64 scalar. Buffer API로 shape [] 명시.
        val srTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(longArrayOf(SAMPLE_RATE)),
            longArrayOf()
        )

        // state (2, 1, 128) → float[2][1][128] 형태의 3-D 배열로 구성.
        val state3D: Array<Array<FloatArray>> = Array(2) { layer ->
            arrayOf(state0.copyOfRange(layer * 128, (layer + 1) * 128))
        }
        val stateTensor = OnnxTensor.createTensor(env, state3D)

        // [DIAG] 첫 호출에 input tensor로 실제 들어간 값 역검증.
        if (!inferenceDiagLogged) {
            val readBack = (inputTensor.value as Array<*>)[0] as FloatArray
            var rbSum = 0.0
            for (v in readBack) rbSum += v * v
            val rbRms = kotlin.math.sqrt(rbSum / readBack.size)
            var frSum = 0.0
            for (v in frame) frSum += v * v
            val frRms = kotlin.math.sqrt(frSum / frame.size)
            Log.i(
                "VadEngine",
                "input readback rms=${"%.4f".format(rbRms)} vs frame rms=${"%.4f".format(frRms)} " +
                        "readback[0..4]=${readBack.take(5).joinToString { "%.4f".format(it) }} " +
                        "frame[0..4]=${frame.take(5).joinToString { "%.4f".format(it) }}"
            )
        }

        val inputs = mapOf(
            "input" to inputTensor,
            "state" to stateTensor,
            "sr" to srTensor,
        )

        val results = session.run(inputs)

        // [DIAG] 첫 호출 시 실제 output 구조 로깅.
        if (!inferenceDiagLogged) {
            inferenceDiagLogged = true
            val names = session.outputInfo.keys.toList()
            for (i in 0 until results.size()) {
                val r = results[i]
                val ti = r.info as? TensorInfo
                Log.i(
                    "VadEngine",
                    "result[$i] name=${names.getOrNull(i)} shape=${ti?.shape?.toList()} dtype=${ti?.type} valueClass=${r.value::class.java.name}"
                )
            }
        }

        // 출력 이름 기반 접근 — 인덱스 순서에 의존하지 않도록.
        val outputValue = results["output"].orElse(null)?.value
            ?: (results[0].value)
        val output = (outputValue as Array<*>)[0] as FloatArray
        val probability = output[0]

        // stateN (2, 1, 128) → state0 flatten.
        val stateName = session.outputInfo.keys.firstOrNull { it != "output" } ?: "stateN"
        val stateN = results[stateName].orElse(null)?.value ?: results[1].value
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

        // 다음 호출 context: 현재 프레임의 마지막 64 샘플을 저장 (concat 경계 맞춤).
        frame.copyInto(context, 0, frame.size - CONTEXT_SIZE, frame.size)

        return probability
    }

    fun reset() {
        state = State.SILENCE
        frameCount = 0
        totalFrames = 0
        state0.fill(0f)
        context.fill(0f)
    }

    fun currentState(): State = state

    fun close() {
        session.close()
    }
}
