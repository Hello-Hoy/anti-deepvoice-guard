# 안드로이드 데모 실시간 스트리밍 + GPT-SoVITS 음성 추가 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPT-SoVITS 전현무 클론 음성 2개를 안드로이드 데모에 추가하고, 재생과 동기되어 VAD·딥보이스·STT·피싱·통합위협 5지표가 실시간 점진 갱신되도록 데모 화면을 개편한다.

**Architecture:** 시나리오 시작 시 `DemoAnalysisPipeline.buildTimeline()`이 VAD(프레임별)·딥보이스(1초 슬라이딩 윈도우)·구간별 피싱/통합위협을 한 번에 계산해 100ms 격자 `DemoTimeline`을 만든다(A안). 재생 중 ticker가 현재 위치의 `DemoFrame`을 선택해 5개 카드를 갱신한다. 무거운 계산은 사전 1회, UI는 결정적 replay.

**Tech Stack:** Kotlin, Jetpack Compose, ONNX Runtime(AASIST `aasist.onnx`, Silero VAD `silero_vad.onnx`), JUnit. Python(ffmpeg/scipy/onnxruntime)으로 데모 자산 전처리·검증.

**Spec:** `docs/superpowers/specs/2026-05-23-android-demo-realtime-streaming-design.md`

---

## 사전 컨텍스트 (실행자 필독)

- 작업 루트: `/Users/hyohee/Documents/Claude_project/anti-deepvoice-guard`. 안드로이드 모듈: `android-app/`. Python venv: `.venv/bin/python` (루트).
- 패키지: `com.deepvoiceguard.app`. 데모 화면=`android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/DemoScreen.kt`, 파이프라인=`.../inference/DemoAnalysisPipeline.kt`.
- 기존 타입(이미 존재, 그대로 사용):
  - `inference/InferenceEngine.kt`: `interface InferenceEngine { suspend fun detect(audio: FloatArray): DetectionResult; fun isReady(): Boolean }`, `data class DetectionResult(fakeScore, realScore, confidence, latencyMs)`.
  - `inference/OnDeviceEngine.kt`: `class OnDeviceEngine(context): InferenceEngine` (aasist.onnx 로드), `close()`.
  - `inference/DetectionAggregator.kt`: `enum ThreatLevel { SAFE, CAUTION, WARNING, DANGER }`, `data class AggregatedResult(threatLevel, averageFakeScore, latestResult, consecutiveHighCount)`, `class DetectionAggregator(windowSize=5).add(DetectionResult): AggregatedResult`, `clear()`.
  - `inference/CombinedThreatResult.kt`: `enum CombinedThreatLevel { SAFE, CAUTION, WARNING, DANGER, CRITICAL }`, `data class CombinedThreatResult(combinedThreatLevel, ..., phishingScore, matchedKeywords, ...)`.
  - `inference/CombinedThreatAggregator.kt`: `class CombinedThreatAggregator().combine(deepfakeResult: AggregatedResult?, phishingResult: PhishingResult?, sttStatus: SttStatus, transcription: String): CombinedThreatResult`.
  - `phishing/PhishingKeywordDetector.kt`: `class PhishingKeywordDetector(context).analyze(text: String): PhishingResult`.
  - `phishing/model/PhrasePattern.kt`: `data class PhishingResult(score, matchedKeywords, matchedPhrases, threatLevel, transcription)`.
  - `stt/SttStatus.kt`: `enum SttStatus { ... LISTENING, UNAVAILABLE ... }`.
  - `audio/VadEngine.kt`: `class VadEngine(context, modelFileName="silero_vad.onnx") { fun process(frame: FloatArray): Float /*512 samples→speechProb*/; fun reset(); fun close() }`.
  - `inference/DemoAnalysisPipeline.kt`: `class DemoAnalysisPipeline(context, inferenceEngine, phishingDetector)`, 기존 `analyze()` + private `loadWavFromAssets`, `segmentAudio`, `parseWav`, `prepareSegment`, `loadTextFromAssets`. `data class DemoResult(deepfakeResult, phishingResult, combinedResult, transcript, audioLoaded)`, `data class DemoScenario(id,title,description,audioAsset,transcriptAsset,expectedResult)`, `DemoAnalysisException`/`DemoFailureReason`.
- 단위테스트 위치: `android-app/app/src/test/java/com/deepvoiceguard/app/...`. Gradle: `cd android-app && ./gradlew :app:testDebugUnitTest`(단위), `./gradlew :app:assembleDebug`(빌드). (실행자 환경에 Android SDK 필요. SDK 미설치면 빌드는 사용자에게 위임하고 코드/단위테스트까지만 진행.)
- 커밋 메시지 끝에:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- 브랜치: 현재 브랜치 그대로(`feature/jeonhyunmoo-voice-extraction`).

## File Structure

| 파일 | 책임 |
|---|---|
| `tools/make_demo_jhm_wav.py` [생성] | GPT-SoVITS 32k wav → 16k 협대역 demo_08/09.wav + aasist 점수 검증 |
| `android-app/app/src/main/assets/demo/demo_08.wav`, `demo_09.wav` [생성] | 전처리된 데모 오디오(16k mono PCM16) |
| `.../assets/demo/demo_08_transcript.txt`, `demo_09_transcript.txt` [생성] | 사전 전사 |
| `.../inference/DemoTimeline.kt` [생성] | `DemoFrame`/`DemoTimeline` + 순수 헬퍼(vadActiveAt/stepHoldIndexAt/transcriptCharsAt/windowEndingAt) |
| `.../inference/DemoAnalysisPipeline.kt` [수정] | `buildTimeline()` 추가(+ VadEngine 주입), 기존 analyze 보존 |
| `.../ui/screens/DemoScreen.kt` [수정] | 타임라인 기반 5지표 실시간 렌더링으로 개편 + demoScenarios 2줄 추가 |
| `android-app/app/src/test/java/com/deepvoiceguard/app/inference/DemoTimelineTest.kt` [생성] | 순수 헬퍼 단위테스트 |

---

## Task 1: 데모 자산 생성 + 딥보이스 점수 사전 검증 (empirical-first)

**Files:**
- Create: `tools/make_demo_jhm_wav.py`
- Create: `android-app/app/src/main/assets/demo/demo_08.wav`, `demo_09.wav`, `demo_08_transcript.txt`, `demo_09_transcript.txt`

- [ ] **Step 1: 변환+검증 스크립트 작성**

`tools/make_demo_jhm_wav.py`:
```python
#!/usr/bin/env python3
"""GPT-SoVITS 전현무 클론 데모(32k) → 16k 협대역 전처리 → assets/demo/demo_08,09.wav.

기존 tools/preprocess_demo_wav.py의 전처리(100~3000Hz band-pass + RMS -30dBFS)를 재사용해
전화통화 대역을 에뮬레이션한다. 변환 후 aasist.onnx로 fakeScore를 출력해 데모 적합성을 검증한다.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from preprocess_demo_wav import decode_to_16k_mono, preprocess, to_int16_wav  # noqa: E402

SRC = ROOT / "GPT-SoVITS/outputs/jhm/demo"
DST = ROOT / "android-app/app/src/main/assets/demo"
MAPPING = [("01_call.wav", "demo_08.wav"), ("02_menu.wav", "demo_09.wav")]


def aasist_fake_score(wav_path: Path) -> float:
    import onnxruntime as ort
    import soundfile as sf
    model = ROOT / "android-app/app/src/main/assets/aasist.onnx"
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    audio, _ = sf.read(str(wav_path), dtype="float32")
    target = 64600
    if len(audio) >= target:
        audio = audio[:target]
    else:
        audio = np.pad(audio, (0, target - len(audio)))
    inp = sess.get_inputs()[0]
    x = audio.reshape(1, -1).astype(np.float32)
    out = sess.run(None, {inp.name: x})[0]
    e = np.exp(out - out.max())
    prob = (e / e.sum()).ravel()
    return float(prob[0])  # index 0 = spoof/fake


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in MAPPING:
        src = SRC / src_name
        dst = DST / dst_name
        raw = decode_to_16k_mono(src)
        clean = preprocess(raw)
        to_int16_wav(clean, dst)
        score = aasist_fake_score(dst)
        flag = "OK(fake)" if score >= 0.7 else "LOW — 데모 부적합 가능"
        print(f"[{dst_name}] {len(clean)/16000:.1f}s  fakeScore={score:.3f}  {flag}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 → 변환 + 점수 확인**

Run: `.venv/bin/python tools/make_demo_jhm_wav.py`
Expected: 두 줄 출력, 각 `fakeScore=` 표시. **fakeScore ≥ 0.7 (OK)** 이면 진행. 만약 LOW면 STOP하고 보고 — 데모 핵심(AI음성 탐지)이 깨지므로 사용자와 전처리 강도/대상 재논의.

- [ ] **Step 3: 전사 파일 작성**

`android-app/app/src/main/assets/demo/demo_08_transcript.txt`:
```
안녕하세요. 저 전현무입니다. 다름이 아니라 저희가 지금 전현무계획 촬영 중이거든요. 지금 스태프들이랑 다 같이 이동 중인데 촬영 가능할까요?
```
`android-app/app/src/main/assets/demo/demo_09_transcript.txt`:
```
저희 스태프들까지 다 해서 총 25명이고요. 방송 나가시기도 하고, 저희가 찐맛집만 골라 다니잖아요. 메뉴 중에 제일 유명한 고기 세트로 바로 세팅 좀 부탁드릴게요.
```

- [ ] **Step 4: 검증**

Run:
```bash
cd /Users/hyohee/Documents/Claude_project/anti-deepvoice-guard
.venv/bin/python -c "import soundfile as sf; [print(p, sf.info('android-app/app/src/main/assets/demo/'+p).samplerate, sf.info('android-app/app/src/main/assets/demo/'+p).channels) for p in ['demo_08.wav','demo_09.wav']]"
wc -m android-app/app/src/main/assets/demo/demo_08_transcript.txt android-app/app/src/main/assets/demo/demo_09_transcript.txt
```
Expected: 두 wav 모두 `16000 1`, 전사 파일 비어있지 않음.

- [ ] **Step 5: 커밋**

```bash
git add tools/make_demo_jhm_wav.py android-app/app/src/main/assets/demo/demo_08.wav android-app/app/src/main/assets/demo/demo_09.wav android-app/app/src/main/assets/demo/demo_08_transcript.txt android-app/app/src/main/assets/demo/demo_09_transcript.txt
git commit -m "feat(android): GPT-SoVITS 전현무 데모 음성 2개(demo_08/09) 협대역 전처리 + 전사"
```

> 주: `.gitignore`가 `*.wav`를 무시할 수 있음. assets의 데모 wav는 앱에 필요하므로 `git add -f`로 강제 추가. 기존 demo_01..07.wav가 git에 추적되는지 `git ls-files android-app/app/src/main/assets/demo/` 로 확인하고 동일 방식 적용. (추적돼 있으면 `-f` 불필요.)

---

## Task 2: DemoTimeline 데이터 모델 + 순수 헬퍼 (TDD)

**Files:**
- Create: `android-app/app/src/main/java/com/deepvoiceguard/app/inference/DemoTimeline.kt`
- Test: `android-app/app/src/test/java/com/deepvoiceguard/app/inference/DemoTimelineTest.kt`

- [ ] **Step 1: 실패 테스트 작성**

`DemoTimelineTest.kt`:
```kotlin
package com.deepvoiceguard.app.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoTimelineTest {

    @Test
    fun vadActiveAt_picks_frame_by_offset_and_threshold() {
        val probs = floatArrayOf(0.1f, 0.9f, 0.2f) // 프레임 32ms 간격
        assertFalse(DemoTimelineMath.vadActiveAt(probs, frameMs = 32, offsetMs = 0, threshold = 0.5f))
        assertTrue(DemoTimelineMath.vadActiveAt(probs, frameMs = 32, offsetMs = 40, threshold = 0.5f)) // idx 1
        assertFalse(DemoTimelineMath.vadActiveAt(probs, frameMs = 32, offsetMs = 70, threshold = 0.5f)) // idx 2
    }

    @Test
    fun vadActiveAt_out_of_range_is_false() {
        val probs = floatArrayOf(0.9f)
        assertFalse(DemoTimelineMath.vadActiveAt(probs, 32, offsetMs = 9999, threshold = 0.5f))
        assertFalse(DemoTimelineMath.vadActiveAt(FloatArray(0), 32, 0, 0.5f))
    }

    @Test
    fun stepHoldIndexAt_returns_last_step_at_or_before_offset() {
        val stepTimesMs = intArrayOf(1000, 2000, 3000)
        assertEquals(-1, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 500))
        assertEquals(0, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 1000))
        assertEquals(0, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 1999))
        assertEquals(1, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 2000))
        assertEquals(2, DemoTimelineMath.stepHoldIndexAt(stepTimesMs, 99999))
    }

    @Test
    fun transcriptCharsAt_is_proportional_and_clamped() {
        assertEquals(0, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 0, durationMs = 1000))
        assertEquals(5, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 500, durationMs = 1000))
        assertEquals(10, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 5000, durationMs = 1000))
        assertEquals(0, DemoTimelineMath.transcriptCharsAt(10, offsetMs = 100, durationMs = 0)) // dur 0 가드
    }

    @Test
    fun windowEndingAt_left_pads_when_short() {
        val audio = floatArrayOf(1f, 2f, 3f, 4f)
        val w = DemoTimelineMath.windowEndingAt(audio, endSample = 2, length = 4)
        // 마지막 endSample(2) 샘플 [1,2]를 오른쪽 끝에, 앞은 0 패딩
        assertEquals(4, w.size)
        assertEquals(0f, w[0], 0f); assertEquals(0f, w[1], 0f)
        assertEquals(1f, w[2], 0f); assertEquals(2f, w[3], 0f)
    }

    @Test
    fun windowEndingAt_takes_tail_when_long() {
        val audio = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val w = DemoTimelineMath.windowEndingAt(audio, endSample = 5, length = 3)
        assertEquals(3, w.size)
        assertEquals(3f, w[0], 0f); assertEquals(4f, w[1], 0f); assertEquals(5f, w[2], 0f)
    }
}
```

- [ ] **Step 2: 실패 확인**

Run: `cd android-app && ./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.inference.DemoTimelineTest"`
Expected: 컴파일 실패(`DemoTimelineMath` 미정의).

- [ ] **Step 3: 구현**

`DemoTimeline.kt`:
```kotlin
package com.deepvoiceguard.app.inference

/** 재생 시점별 5지표 스냅샷 (100ms 격자). */
data class DemoFrame(
    val offsetMs: Int,
    val vadActive: Boolean,
    val fakeScore: Float,
    val deepfakeLevel: ThreatLevel,
    val phishingScore: Float,
    val threatLevel: CombinedThreatLevel,
    val transcriptChars: Int,
)

/** 데모 타임라인: 재생 동기 replay용. */
data class DemoTimeline(
    val durationMs: Int,
    val frames: List<DemoFrame>,
    val transcript: String,
    val vadAvailable: Boolean,
    val finalResult: DemoResult,
) {
    /** 재생 위치(ms)에 해당하는 프레임(없으면 첫 프레임/널). */
    fun frameAt(offsetMs: Int): DemoFrame? {
        if (frames.isEmpty()) return null
        val idx = (offsetMs / DemoTimelineMath.FRAME_MS).coerceIn(0, frames.size - 1)
        return frames[idx]
    }
}

/** 순수 파생 로직 — Android 의존 없이 단위테스트 가능. */
object DemoTimelineMath {
    const val FRAME_MS = 100

    fun vadActiveAt(probs: FloatArray, frameMs: Int, offsetMs: Int, threshold: Float): Boolean {
        if (probs.isEmpty() || frameMs <= 0) return false
        val idx = offsetMs / frameMs
        return idx in probs.indices && probs[idx] > threshold
    }

    /** time<=offset 중 가장 마지막 step의 인덱스. 없으면 -1. (stepTimesMs는 오름차순) */
    fun stepHoldIndexAt(stepTimesMs: IntArray, offsetMs: Int): Int {
        var result = -1
        for (i in stepTimesMs.indices) {
            if (stepTimesMs[i] <= offsetMs) result = i else break
        }
        return result
    }

    fun transcriptCharsAt(length: Int, offsetMs: Int, durationMs: Int): Int {
        if (durationMs <= 0) return 0
        val ratio = (offsetMs.toFloat() / durationMs).coerceIn(0f, 1f)
        return (length * ratio).toInt().coerceIn(0, length)
    }

    /** audio에서 endSample에 끝나는 length 윈도우. 모자라면 왼쪽 0 패딩. */
    fun windowEndingAt(audio: FloatArray, endSample: Int, length: Int): FloatArray {
        val out = FloatArray(length)
        val end = endSample.coerceIn(0, audio.size)
        val start = (end - length).coerceAtLeast(0)
        val count = end - start
        if (count > 0) audio.copyInto(out, destinationOffset = length - count, startIndex = start, endIndex = end)
        return out
    }
}
```

- [ ] **Step 4: 통과 확인**

Run: `cd android-app && ./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.inference.DemoTimelineTest"`
Expected: BUILD SUCCESSFUL, 6 tests passed.

- [ ] **Step 5: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/inference/DemoTimeline.kt android-app/app/src/test/java/com/deepvoiceguard/app/inference/DemoTimelineTest.kt
git commit -m "feat(android): DemoTimeline 모델 + 순수 파생 헬퍼 (TDD)"
```

---

## Task 3: DemoAnalysisPipeline.buildTimeline() — VAD/딥보이스/피싱/통합 타임라인

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/inference/DemoAnalysisPipeline.kt`

- [ ] **Step 1: 생성자에 VadEngine(옵션) 추가**

`DemoAnalysisPipeline` 클래스 선언을 다음으로 교체(필드 추가):
```kotlin
class DemoAnalysisPipeline(
    private val context: Context,
    private val inferenceEngine: InferenceEngine,
    private val phishingDetector: PhishingKeywordDetector,
    private val vadEngine: com.deepvoiceguard.app.audio.VadEngine? = null,
) {
```
(기존 `analyze()`·private 헬퍼는 그대로 둔다.)

- [ ] **Step 2: import 추가**

파일 상단 import 블록에 추가:
```kotlin
import com.deepvoiceguard.app.stt.SttStatus
```
(이미 있으면 중복 추가하지 말 것. `SttStatus`는 기존 analyze에서 사용 중이므로 보통 이미 존재.)

- [ ] **Step 3: buildTimeline() + 보조 메서드 추가**

`analyze()` 메서드 아래에 추가:
```kotlin
    /**
     * 재생 동기 replay용 타임라인을 만든다.
     * VAD(32ms 프레임) + 딥보이스(1초 슬라이딩 윈도우 누적) + 100ms 격자별 피싱/통합위협.
     * fail-closed는 analyze와 동일(오디오/전사/엔진 누락·추론 실패 시 예외).
     */
    @Throws(DemoAnalysisException::class)
    suspend fun buildTimeline(
        audioAssetPath: String,
        transcriptAssetPath: String? = null,
    ): DemoTimeline {
        val audio = loadWavFromAssets(audioAssetPath)
            ?: throw DemoAnalysisException("오디오 asset을 찾을 수 없습니다: $audioAssetPath", DemoFailureReason.AUDIO_MISSING)
        if (!inferenceEngine.isReady()) {
            throw DemoAnalysisException("추론 엔진이 준비되지 않았습니다", DemoFailureReason.ENGINE_NOT_READY)
        }
        val sr = 16000
        val durationMs = (audio.size * 1000L / sr).toInt().coerceAtLeast(DemoTimelineMath.FRAME_MS)

        // 1) 딥보이스 1초 슬라이딩 윈도우 누적 step
        val stepMs = 1000
        val win = 64_600
        val steps = ArrayList<Pair<Int, AggregatedResult>>()
        val aggregator = DetectionAggregator()
        try {
            var endSample = stepMs * sr / 1000
            while (true) {
                val e = minOf(endSample, audio.size)
                val window = DemoTimelineMath.windowEndingAt(audio, e, win)
                val agg = aggregator.add(inferenceEngine.detect(window))
                steps.add((e * 1000L / sr).toInt() to agg)
                if (e >= audio.size) break
                endSample += stepMs * sr / 1000
            }
        } catch (ex: Exception) {
            throw DemoAnalysisException("AASIST 추론 실패: ${ex.message}", DemoFailureReason.INFERENCE_FAILED, ex)
        }
        val stepTimes = IntArray(steps.size) { steps[it].first }

        // 2) VAD 프레임별 speechProb (엔진 실패는 비치명 — vadAvailable=false)
        var vadAvailable = false
        val vadProbs: FloatArray = try {
            vadEngine?.let { engine ->
                engine.reset()
                val frameLen = 512
                val n = audio.size / frameLen
                FloatArray(n) { i ->
                    engine.process(audio.copyOfRange(i * frameLen, i * frameLen + frameLen))
                }.also { vadAvailable = true }
            } ?: FloatArray(0)
        } catch (_: Exception) {
            vadAvailable = false
            FloatArray(0)
        }
        val vadFrameMs = 512 * 1000 / sr // 32

        // 3) 전사 (fail-closed)
        val transcript = if (transcriptAssetPath != null) {
            val loaded = loadTextFromAssets(transcriptAssetPath)
            if (loaded.isBlank()) throw DemoAnalysisException("전사본을 읽을 수 없거나 비어 있습니다: $transcriptAssetPath", DemoFailureReason.TRANSCRIPT_MISSING)
            loaded
        } else ""

        // 4) 100ms 격자 frame 생성
        val frames = ArrayList<DemoFrame>()
        var t = 0
        while (t <= durationMs) {
            val vadActive = DemoTimelineMath.vadActiveAt(vadProbs, vadFrameMs, t, 0.5f)
            val stepIdx = DemoTimelineMath.stepHoldIndexAt(stepTimes, t)
            val agg = if (stepIdx >= 0) steps[stepIdx].second else null
            val chars = DemoTimelineMath.transcriptCharsAt(transcript.length, t, durationMs)
            val revealed = transcript.substring(0, chars)
            val phishing = if (revealed.isNotBlank()) phishingDetector.analyze(revealed) else null
            val combined = combinedAggregator.combine(
                deepfakeResult = agg,
                phishingResult = phishing,
                sttStatus = if (revealed.isNotBlank()) SttStatus.LISTENING else SttStatus.UNAVAILABLE,
                transcription = revealed,
            )
            frames.add(
                DemoFrame(
                    offsetMs = t,
                    vadActive = vadActive,
                    fakeScore = agg?.averageFakeScore ?: 0f,
                    deepfakeLevel = agg?.threatLevel ?: ThreatLevel.SAFE,
                    phishingScore = combined.phishingScore,
                    threatLevel = combined.combinedThreatLevel,
                    transcriptChars = chars,
                )
            )
            t += DemoTimelineMath.FRAME_MS
        }

        // 5) 최종 요약 (전체 전사 기준)
        val lastAgg = steps.lastOrNull()?.second
        val finalPhishing = if (transcript.isNotBlank()) phishingDetector.analyze(transcript) else null
        val finalCombined = combinedAggregator.combine(
            deepfakeResult = lastAgg,
            phishingResult = finalPhishing,
            sttStatus = if (transcript.isNotBlank()) SttStatus.LISTENING else SttStatus.UNAVAILABLE,
            transcription = transcript,
        )
        val finalResult = DemoResult(
            deepfakeResult = lastAgg,
            phishingResult = finalPhishing,
            combinedResult = finalCombined,
            transcript = transcript,
            audioLoaded = true,
        )

        return DemoTimeline(durationMs, frames, transcript, vadAvailable, finalResult)
    }
```

- [ ] **Step 4: 컴파일 확인**

Run: `cd android-app && ./gradlew :app:compileDebugKotlin`
Expected: BUILD SUCCESSFUL. (실패 시 import/타입 오류 수정. `combinedAggregator`는 기존 클래스 필드로 이미 존재.)

- [ ] **Step 5: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/inference/DemoAnalysisPipeline.kt
git commit -m "feat(android): DemoAnalysisPipeline.buildTimeline() — VAD/딥보이스/피싱/통합 타임라인"
```

---

## Task 4: DemoScreen 개편 — 5지표 실시간 스트리밍 렌더링

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/DemoScreen.kt`

기존 파일의 재생/생명주기 인프라(`startPlayback`, `loadWaveform`, `WaveformView`, `threatColor`, `highlightKeywords`, `phishing*` 헬퍼, Mutex/generation/DisposableEffect)는 **그대로 유지**한다. 변경점은: ① 파이프라인이 `buildTimeline()`을 쓰도록, ② ticker가 현재 위치로 `DemoFrame`을 선택해 라이브 카드 갱신, ③ VadEngine 생성/주입/해제, ④ 라이브 5지표 카드 추가.

- [ ] **Step 1: import 추가**

상단 import에 추가:
```kotlin
import com.deepvoiceguard.app.audio.VadEngine
import com.deepvoiceguard.app.inference.DemoFrame
import com.deepvoiceguard.app.inference.DemoTimeline
import com.deepvoiceguard.app.inference.ThreatLevel
```

- [ ] **Step 2: VadEngine 생성/주입**

엔진 생성 블록(`val engine = remember(context) { ... OnDeviceEngine ... }` 부근)에 VadEngine 추가하고 파이프라인에 주입:
```kotlin
    val vad = remember(context) { runCatching { VadEngine(context) }.getOrNull() }
    val pipeline = remember(context, engine, detector, vad) {
        engine?.let { DemoAnalysisPipeline(context, it, detector, vad) }
    }
```
그리고 `DisposableEffect(engine)`의 정리 블록에서 엔진 close 옆에 `vad?.let { runCatching { it.close() } }`를 추가(엔진 close 경로와 동일 위치 두 곳 모두).

- [ ] **Step 3: 타임라인/현재프레임 상태 추가**

`DemoScreen()` 상단 상태 선언부에 추가:
```kotlin
    var timeline by remember { mutableStateOf<DemoTimeline?>(null) }
    var currentFrame by remember { mutableStateOf<DemoFrame?>(null) }
```
`resetScreen()`에 추가: `timeline = null; currentFrame = null`.

- [ ] **Step 4: startScenario를 타임라인 기반으로 교체**

`startScenario`의 분석 부분을 다음으로 변경 — `analysisDeferred`가 `buildTimeline`을 호출하고, ticker `onTick`에서 frame을 선택한다. 기존 `startScenario` 본문에서 `pipeline.analyze(...)` 호출을 `pipeline.buildTimeline(...)`로 바꾸고 결과 타입을 `DemoTimeline`으로:
```kotlin
            val analysisDeferred: Deferred<Result<DemoTimeline>> = async(Dispatchers.IO) {
                runCatching { pipeline.buildTimeline(scenario.audioAsset, scenario.transcriptAsset) }
            }
```
`onTick` 콜백을 다음으로 교체(progress + frame 동시 갱신):
```kotlin
                    onTick = { ratio ->
                        if (stillMine()) {
                            playbackProgress = ratio
                            val tl = timeline
                            if (tl != null) {
                                val ms = (ratio * tl.durationMs).toInt()
                                currentFrame = tl.frameAt(ms)
                                transcriptShown = tl.transcript.substring(0, currentFrame?.transcriptChars ?: 0)
                            } else {
                                transcriptShown = revealTranscriptAtProgress(transcriptSource, ratio)
                            }
                        }
                    },
```
분석 결과 처리부에서 `analysisResult`(DemoResult) 대신 `timeline`을 세팅하고, 최종 요약은 `timeline.finalResult` 사용:
```kotlin
            val analysisOutcome = analysisDeferred.await()
            ensureActive()
            if (analysisOutcome.isFailure) {
                playbackDeferred.cancel(); waveformDeferred.cancel()
                val e = analysisOutcome.exceptionOrNull()
                if (stillMine()) {
                    val reason = (e as? DemoAnalysisException)?.reason
                    errorMessage = if (reason != null) "분석 실패($reason): ${e.message}" else "분석 실패: ${e?.message ?: "unknown"}"
                    phase = DemoPhase.IDLE; selectedScenario = null
                }
                return@launch
            }
            val tl = analysisOutcome.getOrThrow()
            ensureActive(); if (!stillMine()) return@launch
            timeline = tl
            analysisResult = tl.finalResult
```
(이후 waveform/playback await·phase 전이 로직은 기존 유지. `phase = DemoPhase.DONE` 시 `currentFrame = tl.frames.lastOrNull()`로 마지막 상태 고정하는 한 줄을 DONE 직전에 추가.)

- [ ] **Step 5: 라이브 5지표 카드 렌더링 추가**

`DemoPlaybackSection(...)` 호출 다음(재생 섹션 아래)에 라이브 카드 블록을 추가. 기존 `if (transcriptShown.isNotBlank() && phase == DemoPhase.PLAYING) { DemoLiveTranscriptCard(...) }` 자리를 다음으로 확장:
```kotlin
            val frame = currentFrame
            if (phase == DemoPhase.PLAYING && frame != null) {
                DemoLiveMetrics(
                    frame = frame,
                    vadAvailable = timeline?.vadAvailable == true,
                    transcriptShown = transcriptShown,
                    result = analysisResult,
                )
                Spacer(modifier = Modifier.height(12.dp))
            }
```
그리고 파일 하단(다른 private @Composable 옆)에 새 컴포저블 추가:
```kotlin
@Composable
private fun DemoLiveMetrics(
    frame: DemoFrame,
    vadAvailable: Boolean,
    transcriptShown: String,
    result: DemoResult?,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // 1) 통합 위협 배너 (실시간)
            Text(
                text = "통합 위협: ${frame.threatLevel}",
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium,
                color = threatColor(frame.threatLevel),
            )
            Spacer(modifier = Modifier.height(8.dp))

            // 2) VAD
            Text(
                text = if (!vadAvailable) "🎙 VAD: N/A"
                       else if (frame.vadActive) "🎙 발화 감지 중" else "⏸ 무음",
                style = MaterialTheme.typography.bodyMedium,
                color = if (frame.vadActive) MaterialTheme.colorScheme.primary
                        else MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(8.dp))

            // 3) 딥보이스 점수 (실시간)
            Text("딥보이스 점수: ${(frame.fakeScore * 100).toInt()}% (${frame.deepfakeLevel})")
            LinearProgressIndicator(
                progress = { frame.fakeScore.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
                color = if (frame.fakeScore > 0.7f) Color.Red else MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(8.dp))

            // 4) 피싱 점수 (실시간)
            Text("피싱 점수: ${(frame.phishingScore * 100).toInt()}%")
            LinearProgressIndicator(
                progress = { frame.phishingScore.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
                color = if (frame.phishingScore > 0.3f) Color(0xFFF57F17) else MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(8.dp))

            // 5) 실시간 STT
            Text("실시간 STT", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary)
            val highlightTerms = result?.let { phishingHighlightTerms(it) }.orEmpty()
            Text(text = highlightKeywords(transcriptShown, highlightTerms), style = MaterialTheme.typography.bodyMedium)
        }
    }
}
```

- [ ] **Step 6: 컴파일 확인**

Run: `cd android-app && ./gradlew :app:compileDebugKotlin`
Expected: BUILD SUCCESSFUL. (오류 시 import/시그니처 정렬.)

- [ ] **Step 7: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/DemoScreen.kt
git commit -m "feat(android): 데모 화면 5지표 실시간 스트리밍 렌더링(타임라인 기반) + VAD 통합"
```

---

## Task 5: 시나리오 등록 + 빌드/테스트 검증

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/DemoScreen.kt` (`demoScenarios`)

- [ ] **Step 1: demoScenarios에 2줄 추가**

`private val demoScenarios = listOf(` 의 마지막 항목 뒤(닫는 `)` 앞)에 추가:
```kotlin
    DemoScenario(8, "전현무 사칭 촬영요청(AI)", "GPT-SoVITS 합성 음성 — 사칭 통화",
        "demo/demo_08.wav", "demo/demo_08_transcript.txt", "DANGER"),
    DemoScenario(9, "전현무 사칭 메뉴요청(AI)", "GPT-SoVITS 합성 음성 — 사칭 통화",
        "demo/demo_09.wav", "demo/demo_09_transcript.txt", "DANGER"),
```

- [ ] **Step 2: 단위테스트 전체 회귀**

Run: `cd android-app && ./gradlew :app:testDebugUnitTest`
Expected: BUILD SUCCESSFUL, 신규 DemoTimelineTest 포함 모든 테스트 통과.

- [ ] **Step 3: 디버그 APK 빌드**

Run: `cd android-app && ./gradlew :app:assembleDebug`
Expected: BUILD SUCCESSFUL. (Android SDK 미설치로 실패하면 STOP하고 사용자에게 빌드/실기기 확인 위임 — 코드/단위테스트는 완료 상태로 보고.)

- [ ] **Step 4: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/DemoScreen.kt
git commit -m "feat(android): demo_08/09(전현무 클론) 시나리오 등록"
```

- [ ] **Step 5: 사용자 검증 안내**

데모 탭 → #8/#9 실행 시 재생과 함께 VAD/딥보이스/피싱/통합위협이 실시간 갱신되는지, 통합위협이 SAFE→DANGER로 상승하는지 실기기에서 확인하도록 보고.

---

## Self-Review 결과 (계획 작성자 점검)

- **Spec 커버리지:** §1 5지표→Task3(타임라인)+Task4(렌더링), §4.1 모델→Task2, §4.2 buildTimeline→Task3, §4.3 화면→Task4, §4.4 자산→Task1+Task5, §5 딥보이스 사전검증→Task1 Step2, §6 VAD 비치명/fail-closed→Task3(vadAvailable)+Task4(N/A), §7 테스트→Task2. 누락 없음.
- **플레이스홀더:** 없음. 모든 코드 스텝에 실제 코드. (fakeScore 임계 0.7, frame 100ms, VAD thr 0.5, step 1s 명시.)
- **타입 일관성:** `DemoTimelineMath.{vadActiveAt,stepHoldIndexAt,transcriptCharsAt,windowEndingAt,FRAME_MS}`, `DemoFrame`/`DemoTimeline`/`frameAt`, `buildTimeline(): DemoTimeline`, `DemoLiveMetrics(frame,vadAvailable,transcriptShown,result)` — Task 간 시그니처 일치. 기존 `threatColor`/`highlightKeywords`/`phishingHighlightTerms`/`combinedAggregator`/`AggregatedResult`/`ThreatLevel`/`CombinedThreatLevel` 재사용 정확.
- **리스크:** Android SDK 빌드 환경 의존(Task5 Step3 가드 포함). 딥보이스 사전검증 LOW 시 Task1에서 STOP·재논의 게이트 존재.
