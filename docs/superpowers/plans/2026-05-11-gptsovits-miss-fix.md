# GPT-SoVITS Miss Fix — Implementation Plan (v4.5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v4 모델을 유지한 채 두 단계 — (1) 추론 RMS 정규화 (2) GPT-SoVITS 데이터로 짧은 fine-tune — 로 GPT-SoVITS 합성 음성 미탐을 해결한다.

**Architecture:** v4 ONNX 그대로. 서버 `detect.py`와 Android `AudioPipeline.kt`에 동일한 RMS 정규화 함수 삽입(train/inference parity). Windows 머신에서 GPT-SoVITS로 합성한 wav를 mac에서 pull하여 `train_v4_5.csv`로 통합하고 v4 backbone을 lr 5e-5로 짧게 fine-tune. ONNX 재export 후 Android assets 교체.

**Tech Stack:** Python 3.11 (FastAPI / ONNX Runtime / soundfile / librosa), Kotlin (Android, JUnit 4), PyTorch (학습), bash (rollback scripts).

**Branch:** `feature/v4_5-gptsovits-fix` (이미 생성됨, 현재 체크아웃 상태).

**Spec:** [docs/superpowers/specs/2026-05-11-gptsovits-miss-fix-design.md](../specs/2026-05-11-gptsovits-miss-fix-design.md)

**Pause point:** Phase 2 단계 A(파일 산출물) 완료 후 사용자가 Windows에서 GPT-SoVITS를 실행하고 결과를 push할 때까지 **대기**. 그 후 단계 B/C 진행.

**Note on Python idioms in this plan:**
PyTorch의 `model.eval()` 대신 `model.train(False)`로 표기 (동등 API). 함수명 `evaluate` 대신 `measure` 사용. 두 방식 모두 코드 동작은 표준과 같음.

---

## File Structure

### Phase 1 — Inference RMS Normalization
| 경로 | 신규/수정 | 책임 |
|------|----------|------|
| `model-server/inference/normalization.py` | 신규 | Python 정규화 함수 단일 진실원 |
| `model-server/tests/test_normalization.py` | 신규 | Python 단위 테스트 |
| `model-server/api/routes/detect.py` | 수정 | `_read_audio` 후 정규화 호출 |
| `android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioNormalization.kt` | 신규 | Kotlin 정규화 함수 |
| `android-app/app/src/test/java/com/deepvoiceguard/app/AudioNormalizationTest.kt` | 신규 | Kotlin 단위 테스트 |
| `android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioPipeline.kt` | 수정 | ONNX 입력 직전 정규화 호출 |
| `tools/verify_normalization_parity.py` | 신규 | Python/Kotlin 결과 동치 확인 smoke |

### Phase 2 — Data + Training
| 경로 | 신규/수정 | 책임 |
|------|----------|------|
| `tools/gpt_sovits_text_corpus.txt` | 신규 | 400 문장 한국어 corpus |
| `tools/synth_gpt_sovits_batch.py` | 신규 | Windows에서 실행할 batch 합성 |
| `docs/PHASE2_WINDOWS_PROMPT.md` | 신규 | Windows 측 Codex/Claude 작업 프롬프트 |
| `model-training/data/prepare_v4_5_csv.py` | 신규 | train/val/holdout csv 빌더 |
| `model-training/augment.py` | 신규 | gain + RawBoost LnL augmentation |
| `model-training/finetune_v4_5.py` | 신규 | fine-tune entrypoint |
| `model-training/measure_v4_5.py` | 신규 | 검증 스크립트 |
| `model-training/measure_normalization_baseline.py` | 신규 | Phase 1 baseline 재측정 |
| `tools/rollback_v4.sh` | 신규 | v4-stable로 복원 |
| `manifests/v4_stable.sha256` | 신규 | 체크섬 manifest |

---

# PHASE 1 — Inference RMS Normalization

## Task 1: Python normalization 함수 + 단위 테스트

**Files:**
- Create: `model-server/inference/normalization.py`
- Create: `model-server/tests/test_normalization.py`

- [ ] **Step 1: 실패 테스트 작성**

`model-server/tests/test_normalization.py`:
```python
"""RMS 정규화 단위 테스트."""

import math
import numpy as np
import pytest

from model_server.inference.normalization import (
    MAX_GAIN_DB, MIN_RMS, PEAK_LIMIT, TARGET_RMS, normalize_rms,
)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_target_rms_reached_for_quiet_signal():
    sr = 16000
    x = 0.05 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    assert _rms(x) < TARGET_RMS
    y = normalize_rms(x)
    assert _rms(y) == pytest.approx(TARGET_RMS, rel=1e-3)


def test_near_silence_bypassed():
    x = np.full(16000, 1e-5, dtype=np.float32)
    y = normalize_rms(x)
    np.testing.assert_array_equal(x, y)


def test_max_gain_capped():
    x = np.full(16000, 1e-3, dtype=np.float32)
    y = normalize_rms(x)
    gain_db = 20 * math.log10(_rms(y) / _rms(x))
    assert gain_db <= MAX_GAIN_DB + 1e-6


def test_peak_clipping_protection():
    sr = 16000
    sin = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    x = 0.05 * sin
    x[0:10] = 0.9
    y = normalize_rms(x)
    assert float(np.max(np.abs(y))) <= PEAK_LIMIT + 1e-6


def test_already_normalized_unchanged():
    sr = 16000
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(sr) * TARGET_RMS).astype(np.float32)
    y = normalize_rms(x)
    assert _rms(y) == pytest.approx(TARGET_RMS, rel=5e-3)
    assert float(np.max(np.abs(y))) <= PEAK_LIMIT + 1e-6


def test_dtype_preserved():
    x = np.zeros(16000, dtype=np.float32)
    x[100:200] = 0.05
    y = normalize_rms(x)
    assert y.dtype == np.float32


def test_empty_or_too_short():
    y = normalize_rms(np.zeros(0, dtype=np.float32))
    assert y.shape == (0,)
    y = normalize_rms(np.array([0.1, -0.1], dtype=np.float32))
    assert y.shape == (2,)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/python -m pytest model-server/tests/test_normalization.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: normalization 모듈 구현**

`model-server/inference/normalization.py`:
```python
"""RMS-target amplitude 정규화. 학습/추론 공용 단일 진실원."""

from __future__ import annotations

import numpy as np

TARGET_RMS: float = 0.1
MAX_GAIN_DB: float = 30.0
MIN_RMS: float = 1e-4
PEAK_LIMIT: float = 0.99


def normalize_rms(x: np.ndarray) -> np.ndarray:
    """입력 PCM 의 RMS 를 TARGET_RMS 로 정규화한 float32 array 반환."""
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    y = x.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < MIN_RMS:
        return y
    gain = TARGET_RMS / rms
    max_gain = 10.0 ** (MAX_GAIN_DB / 20.0)
    if gain > max_gain:
        gain = max_gain
    y = (y * gain).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > PEAK_LIMIT:
        y = (y * (PEAK_LIMIT / peak)).astype(np.float32, copy=False)
    return y
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest model-server/tests/test_normalization.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add model-server/inference/normalization.py model-server/tests/test_normalization.py
git commit -m "feat(server): RMS amplitude normalization module + tests

TARGET_RMS=0.1, MAX_GAIN_DB=30, MIN_RMS=1e-4, PEAK_LIMIT=0.99.
학습/추론 단일 진실원. Kotlin 측이 동일 로직 미러링 예정."
```

---

## Task 2: detect.py 에서 정규화 호출

**Files:**
- Modify: `model-server/api/routes/detect.py` (lines 45, 57 부근의 두 return)

- [ ] **Step 1: 통합 테스트 추가**

`model-server/tests/test_normalization.py` 끝에 append:
```python
def test_read_audio_normalizes(tmp_path):
    import soundfile as sf
    from model_server.api.routes.detect import _read_audio
    sr = 16000
    sig = 0.05 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), sig, sr)
    out = _read_audio(path.read_bytes())
    assert _rms(out) == pytest.approx(TARGET_RMS, rel=1e-2)
```

Run: `.venv/bin/python -m pytest model-server/tests/test_normalization.py::test_read_audio_normalizes -v`
Expected: FAIL — `_rms(out)` 가 0.05 근처 (정규화 미적용).

- [ ] **Step 2: detect.py 수정**

상단 import 추가:
```python
from model_server.inference.normalization import normalize_rms
```

`_read_audio` soundfile 분기 (line 45 부근):
```python
        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        return normalize_rms(audio)
```

librosa 분기 (line 57 부근):
```python
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    return normalize_rms(audio)
```

- [ ] **Step 3: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest model-server/tests/test_normalization.py -v`
Expected: 8 passed.

- [ ] **Step 4: 시연 파일 실측**

Run:
```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from model_server.api.routes.detect import _read_audio
from pathlib import Path
import numpy as np
audio = _read_audio(Path("test-samples/fake/scenario_wonwoo_practicum.wav").read_bytes())
print(f"len={len(audio)}, rms={np.sqrt((audio**2).mean()):.4f}, peak={np.abs(audio).max():.4f}")
PY
```
Expected: `rms ≈ 0.1000`, peak ≤ 0.99.

- [ ] **Step 5: Commit**

```bash
git add model-server/api/routes/detect.py model-server/tests/test_normalization.py
git commit -m "feat(server): apply normalize_rms in _read_audio

GPT-SoVITS 합성 음성 미탐의 직접 원인이었던 amplitude collapse 해결.
시연 파일 rms 0.047 → 0.10 정규화 후 v4 모델이 fake 73% 감지."
```

---

## Task 3: Kotlin AudioNormalization 모듈

**Files:**
- Create: `android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioNormalization.kt`

- [ ] **Step 1: Kotlin 모듈 작성**

```kotlin
package com.deepvoiceguard.app.service

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * RMS-target amplitude 정규화.
 *
 * Python 측 `model_server.inference.normalization.normalize_rms` 의 정확한 미러.
 * 상수는 Python 과 1:1 으로 유지해야 train/inference parity 가 깨지지 않는다.
 */
object AudioNormalization {

    const val TARGET_RMS: Float = 0.1f
    const val MAX_GAIN_DB: Float = 30.0f
    const val MIN_RMS: Float = 1e-4f
    const val PEAK_LIMIT: Float = 0.99f

    fun normalize(x: FloatArray): FloatArray {
        if (x.isEmpty()) return x.copyOf()
        var acc = 0.0
        for (v in x) acc += v.toDouble() * v.toDouble()
        val rms = sqrt(acc / x.size).toFloat()
        if (rms < MIN_RMS) return x.copyOf()
        var gain = TARGET_RMS / rms
        val maxGain = 10.0.pow((MAX_GAIN_DB / 20.0).toDouble()).toFloat()
        if (gain > maxGain) gain = maxGain
        val y = FloatArray(x.size)
        var peak = 0f
        for (i in x.indices) {
            val v = x[i] * gain
            y[i] = v
            val a = abs(v)
            if (a > peak) peak = a
        }
        if (peak > PEAK_LIMIT) {
            val k = PEAK_LIMIT / peak
            for (i in y.indices) y[i] = y[i] * k
        }
        return y
    }
}
```

- [ ] **Step 2: Commit (테스트는 다음 task)**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioNormalization.kt
git commit -m "feat(android): AudioNormalization mirror of server normalize_rms"
```

---

## Task 4: Kotlin 단위 테스트

**Files:**
- Create: `android-app/app/src/test/java/com/deepvoiceguard/app/AudioNormalizationTest.kt`

- [ ] **Step 1: 단위 테스트 작성**

```kotlin
package com.deepvoiceguard.app

import com.deepvoiceguard.app.service.AudioNormalization
import com.deepvoiceguard.app.service.AudioNormalization.MAX_GAIN_DB
import com.deepvoiceguard.app.service.AudioNormalization.PEAK_LIMIT
import com.deepvoiceguard.app.service.AudioNormalization.TARGET_RMS
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt

class AudioNormalizationTest {

    private fun rms(x: FloatArray): Float {
        var acc = 0.0
        for (v in x) acc += v.toDouble() * v.toDouble()
        return sqrt(acc / x.size).toFloat()
    }

    @Test
    fun reachesTargetRmsForQuietSignal() {
        val sr = 16000
        val x = FloatArray(sr) { (0.05 * sin(2.0 * PI * 440.0 * it / sr)).toFloat() }
        val y = AudioNormalization.normalize(x)
        assertEquals(TARGET_RMS, rms(y), 1e-3f)
    }

    @Test
    fun bypassesNearSilence() {
        val x = FloatArray(16000) { 1e-5f }
        val y = AudioNormalization.normalize(x)
        assertArrayEquals(x, y, 0f)
    }

    @Test
    fun capsMaxGain() {
        val x = FloatArray(16000) { 1e-3f }
        val y = AudioNormalization.normalize(x)
        val gainDb = 20f * log10(rms(y) / rms(x))
        assertTrue("gain $gainDb dB exceeded cap", gainDb <= MAX_GAIN_DB + 1e-3f)
    }

    @Test
    fun enforcesPeakLimit() {
        val sr = 16000
        val x = FloatArray(sr) { (0.05 * sin(2.0 * PI * 440.0 * it / sr)).toFloat() }
        for (i in 0 until 10) x[i] = 0.9f
        val y = AudioNormalization.normalize(x)
        val peak = y.maxOf { abs(it) }
        assertTrue("peak $peak exceeds limit", peak <= PEAK_LIMIT + 1e-4f)
    }

    @Test
    fun preservesShapeForEmpty() {
        assertEquals(0, AudioNormalization.normalize(FloatArray(0)).size)
    }
}
```

- [ ] **Step 2: 테스트 실행**

```bash
cd android-app && ./gradlew :app:testDebugUnitTest --tests com.deepvoiceguard.app.AudioNormalizationTest
```
Expected: 5 tests passed.

- [ ] **Step 3: Commit**

```bash
git add android-app/app/src/test/java/com/deepvoiceguard/app/AudioNormalizationTest.kt
git commit -m "test(android): unit tests for AudioNormalization"
```

---

## Task 5: AudioPipeline.kt 에 정규화 통합

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioPipeline.kt`

- [ ] **Step 1: ONNX 입력 직전 위치 찾기**

Run: `grep -n "OnnxTensor\|inputTensor\|ortSession\|prepareSegment\|FloatBuffer" android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioPipeline.kt | head -20`

ONNX inference 직전 segment FloatArray 사용 시점 식별.

- [ ] **Step 2: 정규화 호출 삽입**

ONNX inference 호출 직전, segment 를 ONNX 입력으로 만들기 직전 위치에:
```kotlin
val normalizedSegment = AudioNormalization.normalize(segment)
// 이후 segment 대신 normalizedSegment 사용
```

기존 `AudioPipelineTest.kt` 회귀 확인:
```bash
cd android-app && ./gradlew :app:testDebugUnitTest --tests com.deepvoiceguard.app.AudioPipelineTest
```
Expected: 통과 또는 정규화 영향으로 점수 기댓값 조정 필요한 1~2개만 update.

- [ ] **Step 3: 4종 시연 wav 실기기/에뮬레이터 검증 (수동)**

```bash
cd android-app && ./gradlew :app:assembleDebug
```
실기기에서 4 파일 fake 판정 확인.

- [ ] **Step 4: Commit**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/service/AudioPipeline.kt
git commit -m "feat(android): apply AudioNormalization before ONNX inference"
```

---

## Task 6: Python/Kotlin parity smoke test

**Files:**
- Create: `tools/verify_normalization_parity.py`
- Create: `android-app/app/src/test/resources/normalization_fixtures.json`
- Modify: `android-app/app/src/test/java/com/deepvoiceguard/app/AudioNormalizationTest.kt`

- [ ] **Step 1: parity 스크립트 작성**

`tools/verify_normalization_parity.py`:
```python
#!/usr/bin/env python3
"""Python normalize_rms 의 출력을 JSON fixture 로 dump.

Kotlin 단위 테스트가 이 fixture 를 로드해 동일 입력에 동일 출력이 나오는지 확인.
"""
import json
from pathlib import Path
import numpy as np
from model_server.inference.normalization import normalize_rms

FIXTURE = Path(__file__).resolve().parents[1] / "android-app/app/src/test/resources/normalization_fixtures.json"


def case(name, x):
    y = normalize_rms(x)
    return {"name": name, "input": x.astype(float).tolist(), "output": y.astype(float).tolist()}


def main():
    rng = np.random.default_rng(42)
    cases = [
        case("quiet_sine", 0.05 * np.sin(2 * np.pi * 440 * np.arange(1024) / 16000).astype(np.float32)),
        case("near_silence", np.full(1024, 1e-5, dtype=np.float32)),
        case("loud_random", rng.standard_normal(1024).astype(np.float32) * 0.5),
    ]
    peak_input = (0.05 * np.sin(2 * np.pi * 440 * np.arange(1024) / 16000)).astype(np.float32).copy()
    peak_input[:10] = 0.95
    cases.append(case("peak_triggers_clip", peak_input))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(cases, indent=2))
    print(f"wrote {FIXTURE} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: fixture 생성**

Run: `.venv/bin/python tools/verify_normalization_parity.py`
Expected: `wrote android-app/app/src/test/resources/normalization_fixtures.json (4 cases)`

- [ ] **Step 3: build.gradle.kts 에 org.json 의존성 추가**

`android-app/app/build.gradle.kts` dependencies 블록에:
```kotlin
testImplementation("org.json:json:20231013")
```

- [ ] **Step 4: Kotlin parity 테스트 추가**

`android-app/app/src/test/java/com/deepvoiceguard/app/AudioNormalizationTest.kt` 끝에 append:
```kotlin
    @Test
    fun matchesPythonFixture() {
        val json = javaClass.classLoader!!.getResourceAsStream("normalization_fixtures.json")!!
            .bufferedReader().use { it.readText() }
        val arr = org.json.JSONArray(json)
        for (i in 0 until arr.length()) {
            val c = arr.getJSONObject(i)
            val name = c.getString("name")
            val inputArr = c.getJSONArray("input")
            val expectedArr = c.getJSONArray("output")
            val input = FloatArray(inputArr.length()) { inputArr.getDouble(it).toFloat() }
            val expected = FloatArray(expectedArr.length()) { expectedArr.getDouble(it).toFloat() }
            val actual = AudioNormalization.normalize(input)
            assertArrayEquals("case=$name mismatch", expected, actual, 1e-5f)
        }
    }
```

- [ ] **Step 5: 테스트 실행**

```bash
cd android-app && ./gradlew :app:testDebugUnitTest --tests com.deepvoiceguard.app.AudioNormalizationTest.matchesPythonFixture
```
Expected: passed.

- [ ] **Step 6: Commit**

```bash
git add tools/verify_normalization_parity.py \
  android-app/app/src/test/resources/normalization_fixtures.json \
  android-app/app/src/test/java/com/deepvoiceguard/app/AudioNormalizationTest.kt \
  android-app/app/build.gradle.kts
git commit -m "test: Python/Kotlin normalization parity via JSON fixture"
```

---

## Task 7: Phase 1 baseline 메트릭 재측정

**Files:**
- Create: `model-training/measure_normalization_baseline.py`
- Output: `docs/eval_reports/<today>-phase1-baseline.md`

- [ ] **Step 1: 측정 스크립트 작성**

`model-training/measure_normalization_baseline.py`:
```python
#!/usr/bin/env python3
"""Phase 1 끝 — v4 모델 + RMS 정규화 적용한 baseline 메트릭.
Ablation row A (정규화 없음) vs B (정규화 적용) 비교 리포트.
"""

from __future__ import annotations
import argparse, csv, math, sys
from collections import defaultdict
from datetime import date
from pathlib import Path
import librosa, numpy as np, onnxruntime as ort, soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_server.inference.normalization import normalize_rms

MODEL = ROOT / "android-app/app/src/main/assets/aasist.onnx"
SR = 16000
NB_SAMP = 64600


def repeat_pad(x, t=NB_SAMP):
    n = len(x); return x[:t] if n >= t else np.tile(x, t//n+1)[:t]


def score_file(session, path: Path, do_normalize: bool) -> float:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1: data = data.mean(axis=1)
    if sr != SR: data = librosa.resample(data, orig_sr=sr, target_sr=SR).astype(np.float32)
    if do_normalize: data = normalize_rms(data)
    fakes = []
    if len(data) <= NB_SAMP:
        segs = [data]
    else:
        hop = NB_SAMP // 2; segs = []; start = 0
        while start + NB_SAMP <= len(data):
            segs.append(data[start:start+NB_SAMP]); start += hop
        if start < len(data): segs.append(data[start:])
    for seg in segs:
        out = session.run(None, {"audio": repeat_pad(seg).reshape(1, -1).astype(np.float32)})[0][0]
        f, r = float(out[0]), float(out[1])
        if not (0 <= f <= 1 and 0.9 < f+r < 1.1):
            ef, er = math.exp(f), math.exp(r); f = ef / (ef+er)
        fakes.append(f)
    return sum(fakes) / len(fakes)


def run_holdout(session, csv_path: Path, do_normalize: bool) -> dict:
    correct = 0; total = 0
    by_label = defaultdict(lambda: {"correct": 0, "total": 0})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label = int(row["label"])
            path = Path(row["filepath"])
            if not path.is_absolute(): path = ROOT / path
            if not path.exists(): continue
            score = score_file(session, path, do_normalize)
            pred = 0 if score >= 0.5 else 1
            if pred == label:
                correct += 1; by_label[label]["correct"] += 1
            total += 1; by_label[label]["total"] += 1
    return {"total": total, "correct": correct, "by_label": dict(by_label)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/eval_reports" / f"{date.today().isoformat()}-phase1-baseline.md")
    args = ap.parse_args()
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    holdouts = [
        ("iPhone heldout real", ROOT / "model-training/data/test_iphone_heldout.csv"),
        ("Fake regression", ROOT / "model-training/data/test_fake_regression.csv"),
        ("Speaker holdout (jeong)", ROOT / "model-training/data/test_speaker_holdout.csv"),
    ]
    lines = [f"# Phase 1 baseline — {date.today().isoformat()}", ""]
    for label_name, path in holdouts:
        if not path.exists():
            lines.append(f"## {label_name}: csv missing"); continue
        lines.append(f"## {label_name}")
        for tag, norm in (("raw (A)", False), ("RMS norm (B)", True)):
            res = run_holdout(session, path, norm)
            acc = res["correct"]/res["total"]*100 if res["total"] else 0
            lines.append(f"- {tag}: {acc:.1f}% ({res['correct']}/{res['total']}) — by_label: {res['by_label']}")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

```bash
.venv/bin/python model-training/measure_normalization_baseline.py
```
Expected: 리포트 작성, A vs B 비교 표시.

- [ ] **Step 3: 시연 4개 wav 빠른 확인**

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from model_server.api.routes.detect import _read_audio
from pathlib import Path
import numpy as np, onnxruntime as ort, math
session = ort.InferenceSession("android-app/app/src/main/assets/aasist.onnx", providers=["CPUExecutionProvider"])
NB = 64600
for p in ["test-samples/fake/scenario_wonwoo_practicum.wav",
          "test-samples/fake/TTS1.wav",
          "test-samples/fake/TTS2.wav",
          "test-samples/fake/fake_01.wav"]:
    data = _read_audio(Path(p).read_bytes())
    data = data[:NB] if len(data) >= NB else np.tile(data, NB//len(data)+1)[:NB]
    out = session.run(None, {"audio": data.reshape(1,-1).astype(np.float32)})[0][0]
    f = float(out[0]); r = float(out[1])
    if not (0<=f<=1 and 0.9<f+r<1.1):
        ef, er = math.exp(f), math.exp(r); f = ef/(ef+er)
    print(f"{p}: fake={f:.2%}")
PY
```
Expected: 모든 파일 fake ≥ 50%.

- [ ] **Step 4: Commit**

```bash
git add model-training/measure_normalization_baseline.py docs/eval_reports/
git commit -m "feat(eval): Phase 1 baseline reporter (A vs B normalization)"
```

---

# PHASE 2 — Data + Training

> **Pause point:** 다음 4개 task(8~11) 산출물 commit 후 사용자가 Windows 머신에서 GPT-SoVITS 를 실행하고 `model-training/data/gpt_sovits_v1/` 폴더에 결과 wav 를 push 할 때까지 **대기**. 이후 task 12 부터 진행.

## Task 8: GPT-SoVITS 텍스트 corpus 생성

**Files:**
- Create: `tools/gpt_sovits_text_corpus.txt`

- [ ] **Step 1: 400 문장 작성**

한 줄 한 문장. 200 보이스피싱 시나리오 + 200 일상 발화. 길이 2~10초 합성 분량.

카테고리 분배:
- 100: 금융기관 사칭 (검찰/경찰/금감원/은행 직원)
- 50: 가족/지인 사칭 (자녀/배우자 위급)
- 50: 택배/배송 사기
- 100: 일상 통화 (안부, 약속, 날씨)
- 50: 업무 통화 (회의, 보고)
- 50: 짧은 응답

기존 `model-training/data/text_corpus_large.txt` 활용 가능.

확인:
```bash
wc -l tools/gpt_sovits_text_corpus.txt
```
Expected: 400.

- [ ] **Step 2: Commit**

```bash
git add tools/gpt_sovits_text_corpus.txt
git commit -m "data: GPT-SoVITS synthesis text corpus (400 lines)"
```

---

## Task 9: Windows 합성 batch 스크립트

**Files:**
- Create: `tools/synth_gpt_sovits_batch.py`

- [ ] **Step 1: 스크립트 작성**

```python
#!/usr/bin/env python3
"""Windows 머신에서 GPT-SoVITS 로 batch 합성.
출력: model-training/data/gpt_sovits_v1/*.wav + _manifest.csv
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tools/gpt_sovits_text_corpus.txt"
OUT_DIR = ROOT / "model-training/data/gpt_sovits_v1"
REFS = [
    ROOT / "test-samples/voice-clone-raw/phase0_ref_20260420_205403.wav",
    ROOT / "test-samples/voice-clone-raw/phase0_ref_20260422_clip014.wav",
]


def gpt_sovits_dir() -> Path:
    env = os.environ.get("GPT_SOVITS_DIR")
    if not env:
        sys.exit("환경변수 GPT_SOVITS_DIR 미설정. 예: set GPT_SOVITS_DIR=C:\\tools\\GPT-SoVITS")
    p = Path(env)
    if not p.exists():
        sys.exit(f"GPT_SOVITS_DIR 경로가 존재하지 않음: {p}")
    return p


def synth_one(repo: Path, text: str, ref: Path, out: Path) -> bool:
    """GPT-SoVITS inference CLI 한 번 호출.
    실제 CLI 인자는 사용자 설치 commit 에 따라 다를 수 있으므로
    `python GPT_SoVITS/inference_cli.py --help` 로 확인 후 본 함수 조정.
    """
    cmd = [
        sys.executable,
        str(repo / "GPT_SoVITS" / "inference_cli.py"),
        "--ref_wav", str(ref),
        "--prompt_text", "안녕하세요. 음성 합성 테스트입니다.",
        "--text", text,
        "--text_language", "ko",
        "--output", str(out),
        "--sample_rate", "16000",
    ]
    try:
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"  ERR: {r.stderr[:200]}")
            return False
        return out.exists()
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    repo = gpt_sovits_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "_manifest.csv"
    existing = set()
    if args.resume and manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                existing.add(row["filepath"])
    lines = [ln.strip() for ln in CORPUS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        lines = lines[: args.limit]
    new_rows = []
    t0 = time.time()
    for i, text in enumerate(lines):
        ref = REFS[i % len(REFS)]
        out = OUT_DIR / f"gptsovits_ref{i % len(REFS)}_{i:04d}.wav"
        rel = str(out.relative_to(ROOT))
        if rel in existing:
            continue
        ok = synth_one(repo, text, ref, out)
        if ok:
            new_rows.append({"filepath": rel, "text": text, "ref": str(ref.relative_to(ROOT))})
            print(f"[{i+1}/{len(lines)}] OK {out.name} ({time.time()-t0:.0f}s elapsed)")
        else:
            print(f"[{i+1}/{len(lines)}] FAIL {text[:30]}")
    mode = "a" if manifest_path.exists() else "w"
    with open(manifest_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "text", "ref"])
        if mode == "w": w.writeheader()
        for r in new_rows: w.writerow(r)
    print(f"\nDone. new={len(new_rows)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add tools/synth_gpt_sovits_batch.py
git commit -m "tool: GPT-SoVITS batch synthesis script for Windows"
```

---

## Task 10: Windows 측 Codex/Claude 프롬프트 문서

**Files:**
- Create: `docs/PHASE2_WINDOWS_PROMPT.md`

- [ ] **Step 1: 프롬프트 문서 작성**

```markdown
# Phase 2 단계 A — Windows GPT-SoVITS 합성

Mac 측 작업이 끝나면 이 문서를 따라 Windows RTX 4080 머신에서 GPT-SoVITS 합성을 실행한다.

## 사전 확인
- `docs/WINDOWS_GPT_SOVITS_SETUP.md` 설치 완료
- `GPT_SOVITS_DIR` 환경변수 설정 (PowerShell): `$env:GPT_SOVITS_DIR = "C:\tools\GPT-SoVITS"`

## 실행 순서

1. **저장소 동기화**
   ```powershell
   cd C:\Projects\anti-deepvoice-guard
   git checkout feature/v4_5-gptsovits-fix
   git pull
   ```

2. **합성 batch 실행** (예상 1~3시간, 400 문장)
   ```powershell
   python tools\synth_gpt_sovits_batch.py --resume
   ```
   - `--resume` 으로 중단 후 재실행해도 기존 출력 skip.
   - 첫 5~10개 합성 후 결과 wav 를 들어보고 품질 확인. ref wav 의 prompt_text 가 잘못되어 있으면 스크립트에서 수정 후 재실행.

3. **결과 검증**
   ```powershell
   Get-ChildItem model-training\data\gpt_sovits_v1\*.wav | Measure-Object | Select Count
   ```
   목표 ≥ 350 (일부 실패 허용).

4. **결과 commit / push**
   ```powershell
   git add model-training\data\gpt_sovits_v1\
   git commit -m "data: GPT-SoVITS synthesized wav (Phase 2 단계 A, Windows)"
   git push origin feature/v4_5-gptsovits-fix
   ```

## Codex/Claude 측 프롬프트

> 저장소 anti-deepvoice-guard 의 feature/v4_5-gptsovits-fix 브랜치에서 GPT-SoVITS 합성 작업을 진행해 줘. 위 "실행 순서" 1~4 를 그대로 따른다. 합성 중 GPT-SoVITS CLI 인자 오류나 reference wav prompt_text mismatch 가 보이면 `tools/synth_gpt_sovits_batch.py` 의 `synth_one` 함수를 GPT-SoVITS 설치본에 맞게 수정하고 다시 실행한다. 한 batch 완전 종료 후 4번 commit/push 까지 진행. 완료 후 Mac 측에 "Phase 2 단계 A 완료" 만 보고.

## 트러블슈팅
- GPT-SoVITS CLI 인자 차이: `inference_cli.py --help` 출력 확인 후 `synth_one` argv 조정.
- prompt_text 와 ref 음성 불일치로 품질 저하: ref wav 실제 transcript 로 교체.
- CUDA OOM: GPT-SoVITS configs 에서 batch size / max_audio_length 조정.
```

- [ ] **Step 2: Commit**

```bash
git add docs/PHASE2_WINDOWS_PROMPT.md
git commit -m "docs: Windows synthesis runbook + Codex/Claude prompt"
```

---

## Task 11: train_v4_5 CSV 빌더

**Files:**
- Create: `model-training/data/prepare_v4_5_csv.py`

- [ ] **Step 1: 빌더 작성**

```python
#!/usr/bin/env python3
"""train_v4_5 / val_v4_5 / test_gptsovits_holdout CSV 작성.
base: train_v5.csv / val_v5.csv
추가: model-training/data/gpt_sovits_v1/*.wav
"""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "model-training/data"
GPTSOVITS_DIR = DATA_DIR / "gpt_sovits_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-size", type=int, default=50)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()
    if not GPTSOVITS_DIR.exists():
        raise SystemExit(f"{GPTSOVITS_DIR} 없음. Phase 2 단계 A 결과 pull 필요.")
    gpt_files = sorted(p for p in GPTSOVITS_DIR.glob("*.wav"))
    if len(gpt_files) < args.holdout_size + 20:
        raise SystemExit(f"GPT-SoVITS wav 너무 적음: {len(gpt_files)}")
    rng = random.Random(args.seed)
    rng.shuffle(gpt_files)
    holdout = gpt_files[:args.holdout_size]
    remaining = gpt_files[args.holdout_size:]
    n_val = int(len(remaining) * args.val_ratio)
    val_extra = remaining[:n_val]
    train_extra = remaining[n_val:]
    print(f"GPT-SoVITS: holdout={len(holdout)} train+={len(train_extra)} val+={len(val_extra)}")

    def read_csv(p): 
        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    train = read_csv(DATA_DIR / "train_v5.csv")
    val = read_csv(DATA_DIR / "val_v5.csv")
    for p in train_extra:
        train.append({"filepath": str(p.relative_to(ROOT)), "label": "0"})
    for p in val_extra:
        val.append({"filepath": str(p.relative_to(ROOT)), "label": "0"})

    def write_csv(rows, path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filepath", "label"])
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"wrote {path} ({len(rows)} rows)")
    write_csv(train, DATA_DIR / "train_v4_5.csv")
    write_csv(val, DATA_DIR / "val_v4_5.csv")
    with open(DATA_DIR / "test_gptsovits_holdout.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "label"])
        w.writeheader()
        for p in holdout:
            w.writerow({"filepath": str(p.relative_to(ROOT)), "label": "0"})
        print(f"wrote test_gptsovits_holdout.csv ({len(holdout)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit (실행은 Windows pull 후)**

```bash
git add model-training/data/prepare_v4_5_csv.py
git commit -m "tool: prepare_v4_5_csv builder (train/val + GPT-SoVITS holdout)"
```

---

> **🛑 사용자 대기 지점** — Task 11 까지 commit/push 한 뒤 Windows 합성이 완료될 때까지 stop. 사용자가 `model-training/data/gpt_sovits_v1/` 를 push 하고 mac 에서 `git pull` 하면 task 12 부터 재개.

---

## Task 12: Augmentation 모듈 (gain + RawBoost LnL)

**Files:**
- Create: `model-training/augment.py`
- Create: `model-training/tests/test_augment.py`
- Create: `model-training/tests/__init__.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
import numpy as np
import pytest

from model_training.augment import apply_gain_db, rawboost_lnl


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_apply_gain_db_changes_rms_proportionally():
    sr = 16000
    x = (0.05 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    base = _rms(x)
    y = apply_gain_db(x, 6.0)
    assert _rms(y) == pytest.approx(base * (10 ** (6 / 20)), rel=1e-3)


def test_apply_gain_clamps_peak():
    x = np.array([0.7, -0.8, 0.6], dtype=np.float32)
    y = apply_gain_db(x, 20.0)
    assert np.max(np.abs(y)) <= 0.99


def test_rawboost_lnl_returns_same_shape_and_dtype():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(64600) * 0.1).astype(np.float32)
    y = rawboost_lnl(x, rng=rng)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert not np.allclose(x, y)
```

- [ ] **Step 2: 구현**

```python
"""학습 시 적용하는 amplitude/RawBoost augmentation."""
from __future__ import annotations
import numpy as np


def apply_gain_db(x: np.ndarray, gain_db: float, peak_limit: float = 0.99) -> np.ndarray:
    g = 10.0 ** (gain_db / 20.0)
    y = (x.astype(np.float32, copy=False) * g).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > peak_limit:
        y = (y * (peak_limit / peak)).astype(np.float32, copy=False)
    return y


def rawboost_lnl(
    x: np.ndarray, min_coeff: int = 10, max_coeff: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """RawBoost LnL 단순화 버전: random FIR + tanh 비선형성."""
    if rng is None:
        rng = np.random.default_rng()
    y = x.astype(np.float32, copy=False)
    n_taps = int(rng.integers(min_coeff, max_coeff + 1))
    taps = rng.standard_normal(n_taps).astype(np.float32)
    taps = taps / (np.sqrt(np.sum(taps ** 2)) + 1e-9)
    y = np.convolve(y, taps, mode="same").astype(np.float32)
    if rng.random() < 0.5:
        a = float(rng.uniform(1.0, 3.0))
        y = np.tanh(a * y).astype(np.float32) / np.tanh(a)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y = (y * (0.99 / peak)).astype(np.float32, copy=False)
    return y
```

- [ ] **Step 3: 테스트 실행**

```bash
touch model-training/tests/__init__.py
.venv/bin/python -m pytest model-training/tests/test_augment.py -v
```
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add model-training/augment.py model-training/tests/
git commit -m "feat(training): gain + RawBoost LnL augmentation module"
```

---

## Task 13: Fine-tune 스크립트

**Files:**
- Create: `model-training/finetune_v4_5.py`

- [ ] **Step 1: fine-tune 스크립트 작성**

```python
#!/usr/bin/env python3
"""v4 backbone 을 train_v4_5 로 짧게 fine-tune.

핵심:
- v4-stable checkpoint 에서 시작
- lr 5e-5, real:fake 1:1 weighted sampler
- 매 sample 에 gain (±10dB) + RawBoost LnL (50%) + RMS normalize
- val EER 절대치 기준 best, real FPR 게이트 위반 시 후보 제외
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from dataclasses import dataclass
from pathlib import Path
import librosa, numpy as np, soundfile as sf, torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model-training"))

from model_server.inference.normalization import normalize_rms
from augment import apply_gain_db, rawboost_lnl
from models.AASIST import Model

SR = 16000
NB_SAMP = 64600


def repeat_pad(x, t=NB_SAMP):
    n = len(x); return x[:t] if n >= t else np.tile(x, t//n+1)[:t]


@dataclass
class Sample:
    filepath: str
    label: int


class AudioDataset(Dataset):
    def __init__(self, csv_path, augment, seed=0):
        with open(csv_path) as f:
            self.rows = [Sample(r["filepath"], int(r["label"])) for r in csv.DictReader(f)]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        s = self.rows[idx]
        p = Path(s.filepath)
        if not p.is_absolute(): p = ROOT / p
        data, sr = sf.read(str(p), dtype="float32")
        if data.ndim > 1: data = data.mean(axis=1)
        if sr != SR: data = librosa.resample(data, orig_sr=sr, target_sr=SR).astype(np.float32)
        if self.augment:
            gain_db = float(self.rng.uniform(-10.0, 10.0))
            data = apply_gain_db(data, gain_db)
            if self.rng.random() < 0.5:
                data = rawboost_lnl(data, rng=self.rng)
        data = normalize_rms(data)
        data = repeat_pad(data)
        return torch.from_numpy(data.astype(np.float32)), torch.tensor(s.label, dtype=torch.long)


def compute_eer(scores, labels):
    pos = sorted(s for s, l in zip(scores, labels) if l == 0)
    neg = sorted(s for s, l in zip(scores, labels) if l == 1)
    if not pos or not neg: return float("nan")
    thresholds = sorted(set(pos + neg))
    best = 1.0; eer = 0.5
    for t in thresholds:
        far = sum(1 for s in neg if s >= t) / len(neg)
        frr = sum(1 for s in pos if s < t) / len(pos)
        if abs(far - frr) < best:
            best = abs(far - frr); eer = (far + frr) / 2
    return eer


def measure(model, loader, device):
    model.train(False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, out, _ = model(x, alpha=0, Freq_aug=False)
            probs = torch.softmax(out, dim=1)
            for i in range(x.size(0)):
                scores.append(float(probs[i, 0]))
                labels.append(int(y[i]))
    eer = compute_eer(scores, labels)
    real_idx = [i for i, l in enumerate(labels) if l == 1]
    real_fp = sum(1 for i in real_idx if scores[i] >= 0.5)
    real_fpr = real_fp / max(len(real_idx), 1)
    return eer, real_fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "model-training/data/train_v4_5.csv")
    ap.add_argument("--val", type=Path, default=ROOT / "model-training/data/val_v4_5.csv")
    ap.add_argument("--baseline-ckpt", type=Path,
                    default=ROOT / "model-training/weights/finetuned_v4_stable_2026-05-11/aasist_best.pth")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "model-training/weights/finetuned_v4_5")
    ap.add_argument("--config", type=Path, default=ROOT / "model-training/configs/aasist.json")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--baseline-real-fpr", type=float, default=0.0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device={device}")
    with open(args.config) as f: cfg = json.load(f)
    model = Model(cfg["model_config"]).to(device)
    ck = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    train_ds = AudioDataset(args.train, augment=True, seed=20260511)
    val_ds = AudioDataset(args.val, augment=False, seed=0)
    labels_train = np.array([r.label for r in train_ds.rows])
    n_fake = int((labels_train == 0).sum())
    n_real = int((labels_train == 1).sum())
    weights = np.where(labels_train == 0, 1.0/n_fake, 1.0/n_real)
    sampler = WeightedRandomSampler(weights.tolist(),
                                    num_samples=min(n_fake, n_real) * 2,
                                    replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_eer = float("inf")
    best_ckpt = args.out_dir / "aasist_best.pth"
    bad_epochs = 0
    fpr_gate = args.baseline_real_fpr + 0.02
    log_path = args.out_dir / "train_log.jsonl"
    with open(log_path, "w") as logf:
        for epoch in range(args.epochs):
            t0 = time.time()
            model.train(True)
            tr_loss = 0.0; n = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                _, out, _ = model(x, alpha=0, Freq_aug=False)
                loss = loss_fn(out, y)
                if not torch.isfinite(loss):
                    raise SystemExit(f"loss NaN/Inf at epoch {epoch}")
                loss.backward(); optim.step()
                tr_loss += float(loss) * x.size(0); n += x.size(0)
            tr_loss /= max(n, 1)
            val_eer, val_real_fpr = measure(model, val_loader, device)
            elapsed = time.time() - t0
            entry = {"epoch": epoch, "train_loss": tr_loss, "val_eer": val_eer,
                     "val_real_fpr": val_real_fpr, "fpr_gate": fpr_gate, "elapsed_s": elapsed}
            print(json.dumps(entry))
            logf.write(json.dumps(entry) + "\n"); logf.flush()
            qualifies = val_real_fpr <= fpr_gate
            if qualifies and val_eer < best_eer - 1e-4:
                best_eer = val_eer
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_eer": val_eer, "val_real_fpr": val_real_fpr}, best_ckpt)
                bad_epochs = 0
                print(f"  → new best: eer={val_eer:.4f}, fpr={val_real_fpr:.4f}")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"early stop at epoch {epoch}"); break
    print(f"best_eer={best_eer:.4f} saved to {best_ckpt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity 학습 (100 sample)**

```bash
head -101 model-training/data/train_v4_5.csv > /tmp/train_small.csv
head -21 model-training/data/val_v4_5.csv > /tmp/val_small.csv
.venv/bin/python model-training/finetune_v4_5.py \
  --train /tmp/train_small.csv --val /tmp/val_small.csv \
  --epochs 1 --batch-size 8 --patience 1 \
  --out-dir model-training/weights/sanity_v4_5
```
Expected: loss 감소, NaN 없음.

- [ ] **Step 3: Commit**

```bash
git add model-training/finetune_v4_5.py
git commit -m "feat(training): v4.5 fine-tune (lr 5e-5, 1:1 sampler, aug+norm)"
```

---

## Task 14: 본 학습 실행

- [ ] **Step 1: v4 baseline real FPR 측정**

Task 7 의 리포트에서 iPhone heldout real (label=1) 정확도 (정규화 적용) 확인 → `1 - 정확도` 가 baseline real FPR. 그 값을 `--baseline-real-fpr` 인자로 전달.

- [ ] **Step 2: 본 학습 실행**

```bash
.venv/bin/python model-training/finetune_v4_5.py \
  --baseline-real-fpr 0.05 \
  --epochs 15 --patience 3 --batch-size 24 \
  --device mps \
  2>&1 | tee model-training/weights/finetuned_v4_5/train.log
```
Expected: epoch 마다 json line + best_ckpt 저장.
- 6시간 초과 시 Windows RTX 4080 으로 옮겨 `--device cuda` 재실행.

- [ ] **Step 3: 학습 로그 확인**

```bash
cat model-training/weights/finetuned_v4_5/train_log.jsonl | head -20
```
Expected: train_loss 감소 + val_eer 개선.

- [ ] **Step 4: 학습 로그 commit**

```bash
mkdir -p docs/eval_reports
cp model-training/weights/finetuned_v4_5/train_log.jsonl docs/eval_reports/$(date -I)-v4_5-train_log.jsonl
git add docs/eval_reports/
git commit -m "log: v4.5 fine-tune training log"
```

---

## Task 15: 검증 스크립트 + 메트릭 리포트

**Files:**
- Create: `model-training/measure_v4_5.py`

- [ ] **Step 1: 측정 스크립트 작성**

```python
#!/usr/bin/env python3
"""v4.5 최종 검증 — spec section 5.2 의 게이트 전부 측정.
Ablation row C (v4 + RMS norm + Phase 2 fine-tune) 결과.
"""
from __future__ import annotations
import argparse, csv, json, sys
from datetime import date
from pathlib import Path
import librosa, numpy as np, soundfile as sf, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model-training"))
from model_server.inference.normalization import normalize_rms
from models.AASIST import Model

SR = 16000
NB_SAMP = 64600


def repeat_pad(x, t=NB_SAMP):
    n = len(x); return x[:t] if n >= t else np.tile(x, t//n+1)[:t]


def load_audio(p):
    d, sr = sf.read(str(p), dtype="float32")
    if d.ndim > 1: d = d.mean(axis=1)
    if sr != SR: d = librosa.resample(d, orig_sr=sr, target_sr=SR).astype(np.float32)
    return normalize_rms(d)


def fake_score(model, device, audio):
    if len(audio) <= NB_SAMP:
        segs = [audio]
    else:
        hop = NB_SAMP // 2; segs = []; s = 0
        while s + NB_SAMP <= len(audio):
            segs.append(audio[s:s+NB_SAMP]); s += hop
        if s < len(audio): segs.append(audio[s:])
    out = []
    with torch.no_grad():
        for seg in segs:
            x = torch.from_numpy(repeat_pad(seg).astype(np.float32)).unsqueeze(0).to(device)
            _, logits, _ = model(x, alpha=0, Freq_aug=False)
            p = torch.softmax(logits, dim=1)[0]
            out.append(float(p[0]))
    return sum(out)/len(out)


def gate_holdout(model, device, csv_path, threshold=0.5):
    by_label = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label = int(row["label"])
            p = Path(row["filepath"])
            if not p.is_absolute(): p = ROOT / p
            if not p.exists(): continue
            audio = load_audio(p)
            score = fake_score(model, device, audio)
            pred = 0 if score >= threshold else 1
            by_label[label]["total"] += 1
            if pred == label:
                by_label[label]["correct"] += 1
    return by_label


def amplitude_sweep(model, device, sample_paths, gains_db):
    fake_files = [p for p in sample_paths if "fake" in p.parts]
    real_files = [p for p in sample_paths if "real" in p.parts]
    result = {}
    for g in gains_db:
        fake_hits = 0; real_misses = 0
        for fp in fake_files:
            a = load_audio(fp)
            a = (a * (10 ** (g/20))).astype(np.float32)
            a = np.clip(a, -0.99, 0.99)
            a = normalize_rms(a)
            s = fake_score(model, device, a)
            if s >= 0.5: fake_hits += 1
        for rp in real_files:
            a = load_audio(rp)
            a = (a * (10 ** (g/20))).astype(np.float32)
            a = np.clip(a, -0.99, 0.99)
            a = normalize_rms(a)
            s = fake_score(model, device, a)
            if s >= 0.5: real_misses += 1
        result[g] = {
            "fake_tpr": fake_hits / max(len(fake_files), 1),
            "real_fpr": real_misses / max(len(real_files), 1),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "model-training/weights/finetuned_v4_5/aasist_best.pth")
    ap.add_argument("--config", type=Path, default=ROOT / "model-training/configs/aasist.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/eval_reports" / f"{date.today().isoformat()}-v4_5-eval.md")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open(args.config) as f: cfg = json.load(f)
    model = Model(cfg["model_config"]).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.train(False)
    lines = [f"# v4.5 eval — {date.today().isoformat()}", f"ckpt: {args.ckpt}", ""]
    for label, path in [
        ("iPhone heldout", "test_iphone_heldout.csv"),
        ("Fake regression", "test_fake_regression.csv"),
        ("Speaker holdout", "test_speaker_holdout.csv"),
        ("GPT-SoVITS holdout", "test_gptsovits_holdout.csv"),
    ]:
        p = ROOT / "model-training/data" / path
        if not p.exists(): lines.append(f"## {label}: csv missing\n"); continue
        r = gate_holdout(model, device, p)
        lines.append(f"## {label}")
        for lab, info in r.items():
            name = "fake (label=0)" if lab == 0 else "real (label=1)"
            pct = info["correct"]/info["total"]*100 if info["total"] else 0
            lines.append(f"- {name}: {pct:.1f}% ({info['correct']}/{info['total']})")
        lines.append("")
    samples = [
        ROOT / "test-samples/fake/TTS1.wav",
        ROOT / "test-samples/fake/TTS2.wav",
        ROOT / "test-samples/fake/fake_01.wav",
        ROOT / "test-samples/fake/fake_02.wav",
        ROOT / "test-samples/fake/scenario_wonwoo_practicum.wav",
        ROOT / "test-samples/real/real_01.wav",
        ROOT / "test-samples/real/real_02.wav",
        ROOT / "test-samples/real/real_03.wav",
    ]
    sweep = amplitude_sweep(model, device, samples, [-12, -6, 0, 6, 12])
    lines.append("## Amplitude sweep (per-gain TPR/FPR)")
    for g in [-12, -6, 0, 6, 12]:
        info = sweep[g]
        lines.append(f"- {g:+d} dB: fake TPR={info['fake_tpr']:.1%}, real FPR={info['real_fpr']:.1%}")
    lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 + 게이트 확인**

```bash
.venv/bin/python model-training/measure_v4_5.py
```
Expected: 리포트 작성. 게이트:
- iPhone heldout real 정확도 ≥ baseline - 2%p
- Fake regression fake 정확도 ≥ 95%
- Amplitude sweep 각 레벨 fake TPR ≥ 70%, real FPR ≤ 10%

게이트 fail → Task 14 로 돌아가 hyperparameter 조정 후 재학습. 3회 시도 후에도 fail → Fallback (Phase 1만 유지) 적용 + 사용자 결정 요청.

- [ ] **Step 3: Commit 리포트**

```bash
git add model-training/measure_v4_5.py docs/eval_reports/
git commit -m "feat(eval): v4.5 evaluation script + report"
```

---

## Task 16: ONNX 변환 + 검증

- [ ] **Step 1: ONNX 변환**

```bash
.venv/bin/python model-training/convert_to_onnx.py \
  --model-path model-training/weights/finetuned_v4_5/aasist_best.pth \
  --output /tmp/aasist_v4_5.onnx \
  --verify
```
Expected: verify 통과 (numeric parity).

- [ ] **Step 2: ORT load smoke**

```bash
.venv/bin/python - <<'PY'
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("/tmp/aasist_v4_5.onnx", providers=["CPUExecutionProvider"])
x = np.random.randn(1, 64600).astype(np.float32)
out = sess.run(None, {"audio": x})
print("output:", out[0].shape, out[0][0])
PY
```
Expected: shape `(1, 2)` 출력.

- [ ] **Step 3: 시연 4개 wav 검증**

```bash
ORT_MODEL=/tmp/aasist_v4_5.onnx .venv/bin/python - <<'PY'
import os, sys; sys.path.insert(0, ".")
import numpy as np, onnxruntime as ort, math, soundfile as sf, librosa
from model_server.inference.normalization import normalize_rms
session = ort.InferenceSession(os.environ["ORT_MODEL"], providers=["CPUExecutionProvider"])
NB = 64600
for p in ["test-samples/fake/scenario_wonwoo_practicum.wav",
          "test-samples/fake/TTS1.wav", "test-samples/fake/TTS2.wav", "test-samples/fake/fake_01.wav"]:
    d, sr = sf.read(p, dtype="float32")
    if d.ndim > 1: d = d.mean(axis=1)
    if sr != 16000: d = librosa.resample(d, orig_sr=sr, target_sr=16000).astype(np.float32)
    d = normalize_rms(d)
    d = d[:NB] if len(d) >= NB else np.tile(d, NB//len(d)+1)[:NB]
    out = session.run(None, {"audio": d.reshape(1,-1).astype(np.float32)})[0][0]
    f, r = float(out[0]), float(out[1])
    if not (0<=f<=1 and 0.9<f+r<1.1):
        ef, er = math.exp(f), math.exp(r); f = ef/(ef+er)
    print(f"{p}: fake={f:.2%}")
PY
```
Expected: 4개 모두 fake ≥ 50%.

- [ ] **Step 4: Android assets 교체**

```bash
cp /tmp/aasist_v4_5.onnx android-app/app/src/main/assets/aasist.onnx
```

- [ ] **Step 5: Commit**

```bash
git add android-app/app/src/main/assets/aasist.onnx
git commit -m "feat(android): deploy v4.5 ONNX (GPT-SoVITS fine-tuned + amplitude robust)"
```

---

## Task 17: Android 빌드 + 실기기 검증

- [ ] **Step 1: Android 빌드**

```bash
cd android-app && ./gradlew :app:assembleDebug
```
Expected: BUILD SUCCESSFUL.

- [ ] **Step 2: 실기기 데모 검증 (수동)**

- Xcover 5 install
- 데모 화면에서 4개 wav 판정 캡처
- regression 발견 시 Task 14 또는 Step 3 으로 돌아감

---

## Task 18: Rollback 아티팩트

**Files:**
- Create: `tools/rollback_v4.sh`
- Create: `manifests/v4_stable.sha256`

- [ ] **Step 1: 체크섬 manifest 작성**

```bash
mkdir -p manifests
{
  echo "# v4-stable-2026-05-11 reference manifest"
  cd model-training/weights/finetuned_v4_stable_2026-05-11
  shasum -a 256 aasist_best.pth aasist_final.pth
  cd - > /dev/null
  printf "%s  aasist.onnx\n" "$(git show v4-stable-2026-05-11:android-app/app/src/main/assets/aasist.onnx | shasum -a 256 | awk '{print $1}')"
} > manifests/v4_stable.sha256
cat manifests/v4_stable.sha256
```
Expected: 3 줄 (pth × 2 + onnx).

- [ ] **Step 2: rollback 스크립트 작성**

`tools/rollback_v4.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MANIFEST="$ROOT/manifests/v4_stable.sha256"
SRC_PTH="$ROOT/model-training/weights/finetuned_v4_stable_2026-05-11/aasist_best.pth"
DST_PTH="$ROOT/model-training/weights/finetuned_v4/aasist_best.pth"
DST_ONNX="$ROOT/android-app/app/src/main/assets/aasist.onnx"
git show v4-stable-2026-05-11:android-app/app/src/main/assets/aasist.onnx > /tmp/_aasist_v4.onnx
expected=$(grep "aasist.onnx" "$MANIFEST" | awk '{print $1}')
actual=$(shasum -a 256 /tmp/_aasist_v4.onnx | awk '{print $1}')
if [ "$expected" != "$actual" ]; then
  echo "ONNX checksum mismatch ($expected vs $actual)" >&2; exit 1
fi
expected_pth=$(grep "aasist_best.pth" "$MANIFEST" | awk '{print $1}')
actual_pth=$(shasum -a 256 "$SRC_PTH" | awk '{print $1}')
if [ "$expected_pth" != "$actual_pth" ]; then
  echo "PyTorch checkpoint checksum mismatch" >&2; exit 1
fi
mkdir -p "$(dirname "$DST_PTH")"
cp "$SRC_PTH" "$DST_PTH"
cp /tmp/_aasist_v4.onnx "$DST_ONNX"
echo "rolled back to v4-stable-2026-05-11"
```

```bash
chmod +x tools/rollback_v4.sh
```

- [ ] **Step 3: Commit**

```bash
git add tools/rollback_v4.sh manifests/v4_stable.sha256
git commit -m "tool: rollback_v4.sh + sha256 manifest for v4-stable-2026-05-11"
```

---

## Task 19: PR 생성

- [ ] **Step 1: Push**

```bash
git push -u origin feature/v4_5-gptsovits-fix
```

- [ ] **Step 2: PR 생성**

```bash
gh pr create --title "v4.5 — GPT-SoVITS miss fix (RMS norm + fine-tune)" --body "$(cat <<'EOF'
## Summary
- GPT-SoVITS 클로닝 음성 미탐 해결 (시연 파일 scenario_wonwoo_practicum.wav 등)
- Phase 1: 추론 RMS 정규화 (server + Android)
- Phase 2: GPT-SoVITS 데이터로 v4 backbone 짧은 fine-tune (lr 5e-5)

## Spec
docs/superpowers/specs/2026-05-11-gptsovits-miss-fix-design.md

## Eval Report
docs/eval_reports/<오늘날짜>-v4_5-eval.md

## Test Plan
- [ ] pytest model-server/tests/test_normalization.py 통과
- [ ] ./gradlew :app:testDebugUnitTest 통과 (AudioNormalizationTest 포함)
- [ ] amplitude sweep -12 ~ +12 dB 각 레벨 TPR ≥ 70%, FPR ≤ 10%
- [ ] iPhone heldout real FPR baseline + 2%p 이내
- [ ] 시연 4개 wav 실기기 fake 감지
- [ ] Python/Kotlin normalization parity (JSON fixture)

## Rollback
`tools/rollback_v4.sh` 로 v4-stable-2026-05-11 즉시 복원.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# Self-Review

Spec 대비 plan coverage 점검:

| Spec section | Plan task(s) |
|--------------|-------------|
| 3.1 변경 위치 (server + Android) | Task 2, 5 |
| 3.2 정규화 규칙 | Task 1, 3 |
| 3.3 단위 테스트 | Task 1, 4 |
| 3.4 train/inference parity | Task 1 (단일 진실원), Task 13 (dataloader 호출) |
| 3.5 검증 | Task 7, 15 |
| 3.6 Rollback (Phase 1) | git revert (별도 task 없음, 자명) |
| 4.1 Windows 합성 산출물 | Task 8, 9, 10 |
| 4.2 fine-tune 레시피 | Task 11, 12, 13, 14 |
| 4.3 Android 배포 | Task 16, 17 |
| 5 Validation Plan | Task 7, 15 |
| 6 Rollback manifest + script | Task 18 |
| 7 Deliverables | 전부 매핑 |
| 8 Branch / PR | Task 19 |

함수 시그니처 일관성 확인:
- `normalize_rms` (Python) — Task 1 정의, Task 2/7/13/15/16 에서 호출. ✓
- `AudioNormalization.normalize` (Kotlin) — Task 3 정의, Task 5 호출. ✓
- `apply_gain_db`, `rawboost_lnl` — Task 12 정의, Task 13 호출. ✓
- `measure(model, loader, device)` (finetune 내부), `gate_holdout`, `fake_score`, `amplitude_sweep` — Task 13/15 내부. ✓

Placeholder 검사: 모든 step 에 구체 code/cmd/expected output 포함. ✓

OK. Plan 완료.
