# 전현무 단독 음성 추출 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 508개 m4a에서 전현무 단독 발화 10~20초 깨끗한 클립만 골라 한 폴더에 모은다.

**Architecture:** `voicebox-clone-workspace/jhm_extract/` 패키지(focused 모듈) + 얇은 `extract_jhm.py` CLI. 순수 로직(클린니스·턴그룹핑·클립선택·매니페스트·클러스터선택)은 TDD, Resemblyzer/ffmpeg 결합부는 통합 스모크 + 명세의 대화형 검증 게이트(부트스트랩 청취, 파일럿 청취)로 검증. 디코드 캐시 + manifest로 resumable.

**Tech Stack:** Python 3, Resemblyzer(VoiceEncoder, CPU), librosa, soundfile, scikit-learn(KMeans), ffmpeg, pytest. 신규 의존성 설치 없음(모두 `.venv`에 존재).

**Spec:** `docs/superpowers/specs/2026-05-18-jeonhyunmoo-voice-extraction-design.md`

**모든 명령은 리포 루트 `/Users/hyohee/Documents/Claude_project/anti-deepvoice-guard` 에서 실행.** 파이썬은 `.venv/bin/python`, 테스트는 `.venv/bin/python -m pytest`.

---

## File Structure

```
voicebox-clone-workspace/
├── extract_jhm.py                  # 얇은 CLI 엔트리 → jhm_extract.cli.main()
├── pytest.ini                      # pytest 설정
├── .gitignore                      # 대용량 데이터/중간산물 제외(코드만 커밋 가능)
├── jhm_extract/
│   ├── __init__.py                 # 빈 패키지 마커(무거운 import 금지)
│   ├── decode.py                   # cache_key, decode_to_wav16k
│   ├── cleanliness.py              # CleanMetrics, cleanliness_gate
│   ├── segments.py                 # group_turns, select_clips, sliding_window_embeddings
│   ├── anchor.py                   # pick_target_cluster, nearest_distinct_files, bootstrap_anchor
│   ├── manifest.py                 # Manifest
│   └── cli.py                      # argparse, run_bootstrap/run_pilot/run_full
└── tests/
    ├── conftest.py                 # sys.path shim + 합성 오디오 픽스처
    ├── test_decode.py
    ├── test_cleanliness.py
    ├── test_segments.py
    ├── test_anchor.py
    └── test_manifest.py
```

런타임 산출(gitignore): `_cache/`, `_anchor/`, `_anchor_candidates/`, `jeonhyunmoo_clean/`.

---

### Task 0: 프로젝트 스켈레톤 + pytest 경로 + gitignore

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/__init__.py`
- Create: `voicebox-clone-workspace/pytest.ini`
- Create: `voicebox-clone-workspace/.gitignore`
- Create: `voicebox-clone-workspace/tests/conftest.py`

- [ ] **Step 1: 패키지 마커 생성**

`voicebox-clone-workspace/jhm_extract/__init__.py`:

```python
"""전현무 단독 음성 추출 파이프라인. 무거운 import는 각 모듈에서 lazy 처리."""
```

- [ ] **Step 2: pytest.ini 생성**

`voicebox-clone-workspace/pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 3: .gitignore 생성 (코드만 커밋, 데이터/산출 제외)**

`voicebox-clone-workspace/.gitignore`:

```gitignore
# 대용량 소스/중간산물/산출 — 커밋 금지
[전현무계획3]*/
_cache/
_anchor/
_anchor_candidates/
jeonhyunmoo_clean/
*.wav
*.m4a
__pycache__/
.pytest_cache/
candidates/
mp3_prof_segments*/
outputs/
```

- [ ] **Step 4: conftest.py 생성 (sys.path shim + 합성 오디오 픽스처)**

`voicebox-clone-workspace/tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

SR = 16000


def _mod_noise(dur_s: float, gap_every: float = 1.5, gap_len: float = 0.25,
                amp: float = 0.2, seed: int = 0) -> np.ndarray:
    """진폭변조 대역잡음 + 주기적 완전무음 gap = '깨끗한 발화' 합성."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    base = rng.standard_normal(n).astype(np.float32)
    k = np.ones(8, np.float32) / 8.0
    base = np.convolve(base, k, mode="same").astype(np.float32)
    env = np.ones(n, np.float32)
    t = 0.0
    while t < dur_s:
        s = int(t * SR)
        e = int((t + gap_len) * SR)
        env[s:e] = 0.0
        t += gap_every
    return (base * env * amp).astype(np.float32)


@pytest.fixture
def clean_speech() -> np.ndarray:
    return _mod_noise(12.0)


@pytest.fixture
def speech_with_music_bed(clean_speech) -> np.ndarray:
    # 톤 진폭 0.007 (RMS≈5e-3): noise_floor(1e-3)의 gap_ratio_max(3)배는 넘되
    # voiced_mult(8)배는 안 넘어야 pause로 분류되어 gap-energy 분기가 발동한다.
    # 더 크면 gap 프레임이 voiced로 분류돼 no_pauses로 잡힘(다른 판별기).
    n = clean_speech.shape[0]
    t = np.arange(n) / SR
    tone = (0.007 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    return (clean_speech + tone).astype(np.float32)


@pytest.fixture
def clipped(clean_speech) -> np.ndarray:
    return np.clip(clean_speech * 10.0, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def mostly_silence() -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.standard_normal(int(12.0 * SR)).astype(np.float32) * 1e-4)


@pytest.fixture
def continuous_tone() -> np.ndarray:
    t = np.arange(int(12.0 * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
```

- [ ] **Step 5: 디렉터리 인식 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace -q`
Expected: `no tests ran` (테스트 0개, 에러 없음 — collection 성공)

- [ ] **Step 6: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/__init__.py voicebox-clone-workspace/pytest.ini voicebox-clone-workspace/.gitignore voicebox-clone-workspace/tests/conftest.py
git commit -m "feat(jhm): 추출 파이프라인 스켈레톤 + 합성 오디오 픽스처"
```

---

### Task 1: decode.cache_key (순수 함수, TDD)

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/decode.py`
- Test: `voicebox-clone-workspace/tests/test_decode.py`

- [ ] **Step 1: 실패 테스트 작성**

`voicebox-clone-workspace/tests/test_decode.py`:

```python
from pathlib import Path

from jhm_extract.decode import cache_key


def test_cache_key_stable_for_same_file(tmp_path: Path):
    f = tmp_path / "a.m4a"
    f.write_bytes(b"x" * 100)
    assert cache_key(f) == cache_key(f)


def test_cache_key_changes_with_size(tmp_path: Path):
    f = tmp_path / "a.m4a"
    f.write_bytes(b"x" * 100)
    k1 = cache_key(f)
    f.write_bytes(b"x" * 200)
    assert cache_key(f) != k1


def test_cache_key_is_16_hex(tmp_path: Path):
    f = tmp_path / "a.m4a"
    f.write_bytes(b"x" * 10)
    k = cache_key(f)
    assert len(k) == 16 and all(c in "0123456789abcdef" for c in k)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_decode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jhm_extract.decode'`

- [ ] **Step 3: 최소 구현**

`voicebox-clone-workspace/jhm_extract/decode.py`:

```python
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

SR = 16000


def cache_key(src: Path) -> str:
    st = src.stat()
    raw = f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_decode.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/decode.py voicebox-clone-workspace/tests/test_decode.py
git commit -m "feat(jhm): decode.cache_key (소스 mtime+size 해시)"
```

---

### Task 2: decode.decode_to_wav16k (ffmpeg 통합)

**Files:**
- Modify: `voicebox-clone-workspace/jhm_extract/decode.py`
- Test: `voicebox-clone-workspace/tests/test_decode.py`

- [ ] **Step 1: 실패 테스트 추가** (`test_decode.py` 끝에 추가)

```python
import subprocess

import numpy as np
import soundfile as sf

from jhm_extract.decode import decode_to_wav16k


def _make_m4a(path: Path):
    sr = 44100
    t = np.arange(sr) / sr
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = path.with_suffix(".wav")
    sf.write(str(wav), sine, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-c:a", "aac", "-loglevel", "error", str(path)],
        check=True,
    )


def test_decode_to_wav16k(tmp_path: Path):
    src = tmp_path / "clip.m4a"
    _make_m4a(src)
    cache = tmp_path / "_cache"
    out = decode_to_wav16k(src, cache)
    assert out.exists()
    y, sr = sf.read(str(out))
    assert sr == 16000
    assert y.ndim == 1
    assert 0.8 < len(y) / sr < 1.3


def test_decode_is_cached(tmp_path: Path):
    src = tmp_path / "clip.m4a"
    _make_m4a(src)
    cache = tmp_path / "_cache"
    out1 = decode_to_wav16k(src, cache)
    mtime1 = out1.stat().st_mtime_ns
    out2 = decode_to_wav16k(src, cache)
    assert out2 == out1
    assert out2.stat().st_mtime_ns == mtime1  # 재디코드 안 함
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_decode.py -k wav16k -v`
Expected: FAIL — `ImportError: cannot import name 'decode_to_wav16k'`

- [ ] **Step 3: 구현 추가** (`decode.py` 끝에 추가)

```python
def decode_to_wav16k(src: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{cache_key(src)}.wav"
    if out.exists() and out.stat().st_size > 0:
        return out
    tmp = out.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR),
         "-vn", "-sample_fmt", "s16", "-loglevel", "error", str(tmp)],
        check=True,
    )
    tmp.replace(out)
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_decode.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/decode.py voicebox-clone-workspace/tests/test_decode.py
git commit -m "feat(jhm): decode_to_wav16k (ffmpeg 16k mono, 캐시 resumable)"
```

---

### Task 3: cleanliness.cleanliness_gate (엄격 게이트, TDD)

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/cleanliness.py`
- Test: `voicebox-clone-workspace/tests/test_cleanliness.py`

- [ ] **Step 1: 실패 테스트 작성**

`voicebox-clone-workspace/tests/test_cleanliness.py`:

```python
from jhm_extract.cleanliness import cleanliness_gate

NF = 1e-3  # 테스트용 고정 노이즈 플로어


def test_clean_speech_passes(clean_speech):
    m = cleanliness_gate(clean_speech, noise_floor=NF)
    assert m.passed, m.reasons
    assert m.gap_rms_ratio < 3.0  # 실제 gate 임계(gap_ratio_max 기본값)
    assert m.voiced_ratio >= 0.7


def test_music_bed_rejected_by_gap_energy(speech_with_music_bed):
    m = cleanliness_gate(speech_with_music_bed, noise_floor=NF)
    assert not m.passed
    assert any("gap_energy" in r for r in m.reasons)


def test_clipping_rejected(clipped):
    m = cleanliness_gate(clipped, noise_floor=NF)
    assert not m.passed
    assert any("clipping" in r for r in m.reasons)


def test_silence_rejected_low_voiced_ratio(mostly_silence):
    m = cleanliness_gate(mostly_silence, noise_floor=NF)
    assert not m.passed
    assert any("voiced_ratio" in r for r in m.reasons)


def test_continuous_tone_rejected_no_pauses(continuous_tone):
    m = cleanliness_gate(continuous_tone, noise_floor=NF)
    assert not m.passed
    assert any("no_pauses" in r for r in m.reasons)


def test_metrics_serializable(clean_speech):
    m = cleanliness_gate(clean_speech, noise_floor=NF)
    d = m.as_dict()
    assert isinstance(d, dict)
    assert set(d) >= {"voiced_ratio", "gap_rms_ratio", "pause_ratio",
                      "spectral_flatness", "peak", "longest_sil_s",
                      "passed", "reasons"}
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_cleanliness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jhm_extract.cleanliness'`

- [ ] **Step 3: 구현**

`voicebox-clone-workspace/jhm_extract/cleanliness.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import librosa
import numpy as np

SR = 16000
HOP_MS = 50
FRAME_MS = 50


@dataclass
class CleanMetrics:
    voiced_ratio: float
    longest_sil_s: float
    peak: float
    gap_rms_ratio: float
    pause_ratio: float
    spectral_flatness: float
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _rms(y: np.ndarray) -> np.ndarray:
    hop = int(SR * HOP_MS / 1000)
    frame = int(SR * FRAME_MS / 1000)
    if y.size < frame:
        return np.zeros(0, dtype=np.float32)
    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]


def cleanliness_gate(
    y: np.ndarray,
    noise_floor: float,
    voiced_mult: float = 8.0,
    gap_ratio_max: float = 3.0,
    min_voiced_ratio: float = 0.7,
    max_sil_s: float = 1.5,
    peak_max: float = 0.95,
    min_pause_ratio: float = 0.04,
    max_flatness: float = 0.6,
) -> CleanMetrics:
    """모든 조건 통과 시 passed=True. 핵심 판별기는 gap_rms_ratio + pause_ratio.

    불변식: gap_ratio_max < voiced_mult. pause = (rms < noise_floor*voiced_mult)로
    분류되므로, gap-energy 분기가 도달 가능하려면 BGM 판정 임계가 pause 임계보다
    낮아야 한다(같으면 gap_rms_ratio ≤ voiced_mult = gap_ratio_max 로 영원히 미발동).
    크게 깔린 BGM은 gap을 voiced로 만들어 no_pauses 분기가 잡고, 약하게 깔린
    BGM(noise_floor의 gap_ratio_max~voiced_mult배)은 gap-energy 분기가 잡는다."""
    reasons: list[str] = []
    rms = _rms(y)
    voiced_thr = max(noise_floor * voiced_mult, 1e-6)
    voiced = rms > voiced_thr if rms.size else np.zeros(0, dtype=bool)
    pause = ~voiced if rms.size else np.zeros(0, dtype=bool)

    voiced_ratio = float(voiced.mean()) if rms.size else 0.0
    pause_ratio = float(pause.mean()) if rms.size else 0.0
    gap_rms = float(rms[pause].mean()) if pause.any() else 0.0
    gap_rms_ratio = gap_rms / noise_floor if noise_floor > 0 else float("inf")

    longest = cur = 0
    for v in pause:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    longest_sil_s = longest * HOP_MS / 1000.0

    peak = float(np.abs(y).max()) if y.size else 0.0
    flat = (
        float(np.median(librosa.feature.spectral_flatness(y=y)[0]))
        if y.size
        else 1.0
    )

    if voiced_ratio < min_voiced_ratio:
        reasons.append(f"voiced_ratio<{min_voiced_ratio}")
    if longest_sil_s > max_sil_s:
        reasons.append(f"silence>{max_sil_s}s")
    if peak > peak_max:
        reasons.append("clipping")
    if pause_ratio < min_pause_ratio:
        reasons.append("no_pauses(music-bed?)")
    if gap_rms_ratio > gap_ratio_max:
        reasons.append(f"gap_energy>{gap_ratio_max}x(BGM?)")
    if flat > max_flatness:
        reasons.append(f"flatness>{max_flatness}(noisy)")

    return CleanMetrics(
        voiced_ratio=voiced_ratio,
        longest_sil_s=longest_sil_s,
        peak=peak,
        gap_rms_ratio=gap_rms_ratio,
        pause_ratio=pause_ratio,
        spectral_flatness=flat,
        passed=not reasons,
        reasons=reasons,
    )
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_cleanliness.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/cleanliness.py voicebox-clone-workspace/tests/test_cleanliness.py
git commit -m "feat(jhm): 엄격 클린니스 게이트 (gap-energy BGM 판별 핵심)"
```

---

### Task 4: segments.group_turns (턴 그룹핑, TDD)

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/segments.py`
- Test: `voicebox-clone-workspace/tests/test_segments.py`

- [ ] **Step 1: 실패 테스트 작성**

`voicebox-clone-workspace/tests/test_segments.py`:

```python
import numpy as np

from jhm_extract.segments import group_turns

WIN_S = 1.6
STEP_S = 0.4


def _starts(n: int) -> list[float]:
    return [i * STEP_S for i in range(n)]


def test_single_contiguous_turn():
    is_t = np.zeros(30, dtype=bool)
    is_t[5:25] = True  # 20 윈도우 연속
    turns = group_turns(is_t, _starts(30), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert len(turns) == 1
    t = turns[0]
    assert abs(t["t0"] - 5 * STEP_S) < 1e-6
    assert abs(t["t1"] - (24 * STEP_S + WIN_S)) < 1e-6
    assert t["dur"] > 6.0


def test_short_turn_dropped():
    is_t = np.zeros(20, dtype=bool)
    is_t[2:6] = True  # 4 윈도우 → dur ~ 1.6+1.2=2.8s < 6
    turns = group_turns(is_t, _starts(20), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert turns == []


def test_small_gap_merged():
    is_t = np.zeros(40, dtype=bool)
    is_t[5:19] = True
    is_t[19] = False  # 정확히 한 윈도우(idx 19) dip
    is_t[20:33] = True
    turns = group_turns(is_t, _starts(40), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert len(turns) == 1  # 1-윈도우 gap 병합되어 단일 턴


def test_large_gap_splits():
    is_t = np.zeros(60, dtype=bool)
    is_t[2:20] = True
    is_t[40:58] = True  # 사이에 큰 무음
    turns = group_turns(is_t, _starts(60), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert len(turns) == 2


def test_just_over_gap_splits():
    # 3-윈도우 gap: starts[nxt]-starts[j]=1.6 > step+gap_merge=1.2 → 병합 금지
    is_t = np.zeros(40, dtype=bool)
    is_t[5:15] = True   # j는 idx 14에서 끝
    is_t[18:28] = True  # gap = idx 15,16,17 (3 윈도우)
    turns = group_turns(is_t, _starts(40), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=0.0)
    assert len(turns) == 2
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jhm_extract.segments'`

- [ ] **Step 3: 구현**

`voicebox-clone-workspace/jhm_extract/segments.py`:

```python
from __future__ import annotations

import numpy as np


def group_turns(
    is_target: np.ndarray,
    starts: list[float],
    win_s: float,
    step_s: float,
    gap_merge_s: float,
    min_turn_s: float,
) -> list[dict]:
    """연속 타깃 윈도우를 턴으로 묶고, 짧은 dip은 병합. min_turn_s 미만 제거."""
    turns: list[dict] = []
    n = len(is_target)
    i = 0
    max_gap_windows = int(round(gap_merge_s / step_s)) + 1
    while i < n:
        if not is_target[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if is_target[j + 1]:
                j += 1
                continue
            nxt = None
            for k in range(j + 2, min(j + 2 + max_gap_windows, n)):
                if is_target[k]:
                    nxt = k
                    break
            if nxt is not None and (starts[nxt] - starts[j]) <= (step_s + gap_merge_s):
                j = nxt
                continue
            break
        t0 = starts[i]
        t1 = starts[j] + win_s
        dur = t1 - t0
        if dur >= min_turn_s:
            turns.append({"t0": t0, "t1": t1, "dur": dur, "i0": i, "i1": j})
        i = j + 1
    return turns
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_segments.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/segments.py voicebox-clone-workspace/tests/test_segments.py
git commit -m "feat(jhm): segments.group_turns (연속 윈도우 턴 그룹핑 + gap 병합)"
```

---

### Task 5: segments.select_clips (10~20초 절취, TDD)

**Files:**
- Modify: `voicebox-clone-workspace/jhm_extract/segments.py`
- Test: `voicebox-clone-workspace/tests/test_segments.py`

단일 턴 → 클립 변환은 `clip_from_turn` 한 함수로 책임 분리(runner에서 턴-클립 페어링 보장에 사용). `select_clips`는 그 위에 구현(기존 호출부 호환).

- [ ] **Step 1: 실패 테스트 추가** (`test_segments.py` 끝에 추가)

```python
from jhm_extract.segments import clip_from_turn, select_clips


def test_clip_from_turn_short_returns_none():
    assert clip_from_turn({"t0": 0.0, "t1": 8.0, "dur": 8.0}, 10.0, 20.0) is None


def test_clip_from_turn_in_range_whole():
    assert clip_from_turn({"t0": 3.0, "t1": 18.0, "dur": 15.0}, 10.0, 20.0) == (3.0, 18.0)


def test_clip_from_turn_long_center_cropped():
    c = clip_from_turn({"t0": 100.0, "t1": 130.0, "dur": 30.0}, 10.0, 20.0)
    assert c is not None
    t0, t1 = c
    assert abs((t1 - t0) - 20.0) < 1e-6
    assert abs(((t0 + t1) / 2) - 115.0) < 1e-6


def test_clip_from_turn_exact_bounds_kept_whole():
    # dur == min, dur == max 는 strict 비교(<,>)라 둘 다 절취 없이 그대로 반환
    assert clip_from_turn({"t0": 1.0, "t1": 11.0, "dur": 10.0}, 10.0, 20.0) == (1.0, 11.0)
    assert clip_from_turn({"t0": 2.0, "t1": 22.0, "dur": 20.0}, 10.0, 20.0) == (2.0, 22.0)


def test_select_clips_skips_short():
    turns = [
        {"t0": 0.0, "t1": 8.0, "dur": 8.0},
        {"t0": 3.0, "t1": 18.0, "dur": 15.0},
    ]
    assert select_clips(turns, 10.0, 20.0) == [(3.0, 18.0)]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_segments.py -k "clip or select" -v`
Expected: FAIL — `ImportError: cannot import name 'clip_from_turn'`

- [ ] **Step 3: 구현 추가** (`segments.py` 끝에 추가)

```python
def clip_from_turn(
    turn: dict,
    min_clip_s: float = 10.0,
    max_clip_s: float = 20.0,
) -> tuple[float, float] | None:
    """단일 턴 → 10~20초 클립 (t0, t1). min 미만은 None, max 초과는 중앙 절취."""
    dur = turn["dur"]
    if dur < min_clip_s:
        return None
    if dur > max_clip_s:
        mid = (turn["t0"] + turn["t1"]) / 2.0
        return (mid - max_clip_s / 2.0, mid + max_clip_s / 2.0)
    return (turn["t0"], turn["t1"])


def select_clips(
    turns: list[dict],
    min_clip_s: float = 10.0,
    max_clip_s: float = 20.0,
) -> list[tuple[float, float]]:
    """여러 턴에서 클립 절취 (clip_from_turn 위에 구현).

    min_clip_s 미만 턴은 제외되므로 len(결과) <= len(turns) 일 수 있다."""
    out: list[tuple[float, float]] = []
    for t in turns:
        c = clip_from_turn(t, min_clip_s, max_clip_s)
        if c is not None:
            out.append(c)
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_segments.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/segments.py voicebox-clone-workspace/tests/test_segments.py
git commit -m "feat(jhm): segments.select_clips (10~20초 중앙절취)"
```

---

### Task 6: segments.sliding_window_embeddings (Resemblyzer 통합 스모크)

**Files:**
- Modify: `voicebox-clone-workspace/jhm_extract/segments.py`
- Test: `voicebox-clone-workspace/tests/test_segments.py`

- [ ] **Step 1: 스모크 테스트 추가** (`test_segments.py` 끝에 추가)

```python
def test_sliding_window_embeddings_shapes():
    from resemblyzer import VoiceEncoder

    from jhm_extract.segments import sliding_window_embeddings

    rng = np.random.default_rng(0)
    y = (rng.standard_normal(16000 * 5).astype(np.float32) * 0.1)
    enc = VoiceEncoder(device="cpu", verbose=False)
    starts, embs = sliding_window_embeddings(y, enc, win_s=1.6, step_s=0.4)
    assert embs.ndim == 2
    assert embs.shape[0] == len(starts)
    assert embs.shape[0] > 5
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)  # 정규화됨
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_segments.py -k embeddings -v`
Expected: FAIL — `ImportError: cannot import name 'sliding_window_embeddings'`

- [ ] **Step 3: 구현 추가** (`segments.py` 끝에 추가)

```python
def sliding_window_embeddings(
    y: np.ndarray,
    encoder,
    win_s: float = 1.6,
    step_s: float = 0.4,
    sr: int = 16000,
):
    """16k mono y에 슬라이딩 윈도우 화자 임베딩.

    반환: (starts: list[float] 윈도우 시작초, embs: ndarray[n,256] L2정규화).
    임베딩이 없으면 ([], ndarray shape (0,256))."""
    win = int(win_s * sr)
    step = int(step_s * sr)
    starts: list[float] = []
    embs: list[np.ndarray] = []
    for s in range(0, max(0, len(y) - win + 1), step):
        emb = encoder.embed_utterance(y[s:s + win])
        embs.append(emb)
        starts.append(s / sr)
    if not embs:
        return [], np.zeros((0, 256), dtype=np.float32)
    arr = np.stack(embs).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return starts, arr
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_segments.py -v`
Expected: 11 passed (임베딩 테스트는 VoiceEncoder 로드로 수 초 소요 가능)

- [ ] **Step 5: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/segments.py voicebox-clone-workspace/tests/test_segments.py
git commit -m "feat(jhm): sliding_window_embeddings (Resemblyzer 윈도우 임베딩)"
```

---

### Task 7: anchor.pick_target_cluster + nearest_distinct_files (순수, TDD)

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/anchor.py`
- Test: `voicebox-clone-workspace/tests/test_anchor.py`

- [ ] **Step 1: 실패 테스트 작성**

`voicebox-clone-workspace/tests/test_anchor.py`:

```python
import numpy as np

from jhm_extract.anchor import nearest_distinct_files, pick_target_cluster


def test_pick_cluster_by_file_coverage_not_window_count():
    # 클러스터1이 윈도우는 3개로 최다지만 파일은 1개. 클러스터0이 파일 2개 → 0 선택.
    labels = np.array([0, 0, 1, 1, 1, 2])
    file_ids = np.array([0, 1, 2, 2, 2, 3])
    assert pick_target_cluster(labels, file_ids) == 0


def test_pick_cluster_tiebreak_by_window_count():
    # 0과 1 모두 파일 2개 커버 → 윈도우 더 많은 1 선택
    labels = np.array([0, 0, 1, 1, 1])
    file_ids = np.array([0, 1, 0, 1, 1])
    assert pick_target_cluster(labels, file_ids) == 1


def test_nearest_distinct_files_picks_one_per_file():
    centroid = np.array([1.0, 0.0], dtype=np.float32)
    embs = np.array(
        [[0.99, 0.14], [0.95, 0.31], [0.20, 0.98], [0.90, 0.44]],
        dtype=np.float32,
    )
    file_ids = np.array([10, 10, 11, 12])
    picks = nearest_distinct_files(embs, file_ids, centroid, n=2)
    assert len(picks) == 2
    assert len({fid for fid, _ in picks}) == 2  # 서로 다른 파일
    assert picks[0] == (10, 0)  # centroid에 가장 가까운 윈도우
    assert picks[1] == (12, 3)  # file 11(sim 0.20) 건너뛰고 file 12(sim 0.90) 선택
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_anchor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jhm_extract.anchor'`

- [ ] **Step 3: 구현**

`voicebox-clone-workspace/jhm_extract/anchor.py`:

```python
from __future__ import annotations

import numpy as np


def pick_target_cluster(labels: np.ndarray, file_ids: np.ndarray) -> int:
    """가장 많은 '서로 다른 소스 파일'에 걸친 클러스터 선택.
    동률 시 윈도우 수가 많은 클러스터. (전현무 = 메인 MC = 최다 파일 커버리지)"""
    best_label = -1
    best_key = (-1, -1)
    for lab in np.unique(labels):
        mask = labels == lab
        n_files = len(np.unique(file_ids[mask]))
        n_win = int(mask.sum())
        key = (n_files, n_win)
        if key > best_key:
            best_key = key
            best_label = int(lab)
    return best_label


def nearest_distinct_files(
    embs: np.ndarray,
    file_ids: np.ndarray,
    centroid: np.ndarray,
    n: int,
) -> list[tuple[int, int]]:
    """centroid에 가까운 순서로, 서로 다른 file_id에서 하나씩 (file_id, window_idx) n개."""
    c = centroid / (np.linalg.norm(centroid) + 1e-9)
    sims = embs @ c
    order = np.argsort(sims)[::-1]
    picks: list[tuple[int, int]] = []
    seen: set[int] = set()
    for idx in order:
        fid = int(file_ids[idx])
        if fid in seen:
            continue
        seen.add(fid)
        picks.append((fid, int(idx)))
        if len(picks) >= n:
            break
    return picks
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_anchor.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/anchor.py voicebox-clone-workspace/tests/test_anchor.py
git commit -m "feat(jhm): anchor 클러스터 선택 (최다 파일 커버리지) + 대표 윈도우 추출"
```

---

### Task 8: manifest.Manifest (resumable 상태, TDD)

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/manifest.py`
- Test: `voicebox-clone-workspace/tests/test_manifest.py`

- [ ] **Step 1: 실패 테스트 작성**

`voicebox-clone-workspace/tests/test_manifest.py`:

```python
from pathlib import Path

from jhm_extract.manifest import Manifest


def test_roundtrip_and_resume(tmp_path: Path):
    p = tmp_path / "manifest.json"
    m = Manifest(p)
    assert not m.done("keyA")
    m.add_clip({"src": "fileA.m4a", "t0": 1.0, "dur": 12.0, "sim": 0.83})
    m.mark_done("keyA", n_clips=1)
    m.add_error("fileB.m4a", "decode failed")
    m.save()

    m2 = Manifest(p)
    assert m2.done("keyA")
    assert not m2.done("keyC")
    assert len(m2.clips) == 1
    assert m2.clips[0]["dur"] == 12.0  # 레코드 구조가 직렬화 왕복에서 보존됨
    assert m2.clips[0]["src"] == "fileA.m4a"
    assert len(m2.errors) == 1
    assert m2.errors[0] == {"src": "fileB.m4a", "msg": "decode failed"}


def test_stats(tmp_path: Path):
    m = Manifest(tmp_path / "manifest.json")
    m.add_clip({"src": "a", "t0": 0.0, "dur": 12.0, "sim": 0.8})
    m.add_clip({"src": "a", "t0": 30.0, "dur": 18.0, "sim": 0.9})
    s = m.stats()
    assert s["n_clips"] == 2
    assert abs(s["total_seconds"] - 30.0) < 1e-6
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jhm_extract.manifest'`

- [ ] **Step 3: 구현**

`voicebox-clone-workspace/jhm_extract/manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


class Manifest:
    """resumable 상태 + 산출 기록. 처리완료 소스키/클립/에러를 JSON으로 영속."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.done_keys: set[str] = set()
        self.clips: list[dict] = []
        self.errors: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.done_keys = set(data.get("done_keys", []))
            self.clips = data.get("clips", [])
            self.errors = data.get("errors", [])

    def done(self, src_key: str) -> bool:
        return src_key in self.done_keys

    def mark_done(self, src_key: str, n_clips: int) -> None:
        self.done_keys.add(src_key)

    def add_clip(self, record: dict) -> None:
        self.clips.append(record)

    def add_error(self, src: str, msg: str) -> None:
        self.errors.append({"src": src, "msg": msg})

    def stats(self) -> dict:
        total = sum(c.get("dur", 0.0) for c in self.clips)
        return {
            "n_clips": len(self.clips),
            "total_seconds": total,
            "n_errors": len(self.errors),
            "n_done_sources": len(self.done_keys),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "done_keys": sorted(self.done_keys),
            "clips": self.clips,
            "errors": self.errors,
            "stats": self.stats(),  # 정보용 스냅샷; load 시 다시 읽지 않고 재계산
        }
        # 원자적 쓰기: 다중 시간 실행 중 crash가 manifest를 truncate해
        # 다음 run의 json.loads가 깨져 resume 전체를 잃는 것을 방지.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self.path)
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace/tests/test_manifest.py -v`
Expected: 2 passed

- [ ] **Step 5: 전체 회귀 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace -v`
Expected: 모든 테스트 passed (27개 내외)

- [ ] **Step 6: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/manifest.py voicebox-clone-workspace/tests/test_manifest.py
git commit -m "feat(jhm): Manifest (resumable 상태 + 클립/에러/통계)"
```

---

### Task 9: cli — 공통 헬퍼 + run_bootstrap (대화형 확인 게이트)

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/cli.py`
- Create: `voicebox-clone-workspace/extract_jhm.py`

이 태스크는 Resemblyzer/ffmpeg/실음원 결합부라 단위 TDD 대신 **명세의 대화형 검증 게이트**로 확인한다(spec §9). 코드는 완전 제공.

- [ ] **Step 1: cli.py 작성 (공통 + bootstrap)**

`voicebox-clone-workspace/jhm_extract/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .anchor import nearest_distinct_files, pick_target_cluster
from .decode import decode_to_wav16k
from .segments import sliding_window_embeddings

SR = 16000
WIN_S = 1.6
STEP_S = 0.4

WORK = Path(__file__).resolve().parents[1]
SHOW_DIR = WORK / "[전현무계획3] 無계획 노포 맛집 탐방!   매주 금요일 밤 9시 10분"
CACHE = WORK / "_cache"
ANCHOR_DIR = WORK / "_anchor"
ANCHOR_CAND = WORK / "_anchor_candidates"
OUT_DIR = WORK / "jeonhyunmoo_clean"
ANCHOR_NPY = ANCHOR_DIR / "anchor.npy"


def _encoder():
    from resemblyzer import VoiceEncoder

    return VoiceEncoder(device="cpu", verbose=False)


def _list_sources() -> list[Path]:
    return sorted(SHOW_DIR.glob("*.m4a"))


def _short_clips(srcs: list[Path], limit: int) -> list[Path]:
    """파일 크기 작은 순(예고편/쇼츠) = 전현무 내레이션 비중 큼."""
    return sorted(srcs, key=lambda p: p.stat().st_size)[:limit]


def run_bootstrap(sample_n: int = 30, k: int = 6, force: bool = False) -> None:
    if ANCHOR_NPY.exists() and not force:
        print(f"anchor 이미 존재: {ANCHOR_NPY} (재생성하려면 --rebootstrap)")
        return
    ANCHOR_CAND.mkdir(parents=True, exist_ok=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    srcs = _short_clips(_list_sources(), sample_n)
    print(f"부트스트랩 샘플 {len(srcs)}개")
    enc = _encoder()

    all_emb: list[np.ndarray] = []
    all_fid: list[int] = []
    win_meta: list[tuple[int, float, Path]] = []  # (file_idx, t0, wav)
    for fi, src in enumerate(srcs):
        try:
            wav = decode_to_wav16k(src, CACHE)
            y, _ = sf.read(str(wav), dtype="float32")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {src.name}: {e}")
            continue
        starts, embs = sliding_window_embeddings(y, enc, WIN_S, STEP_S)
        for s, e in zip(starts, embs):
            all_emb.append(e)
            all_fid.append(fi)
            win_meta.append((fi, s, wav))

    if not all_emb:
        raise RuntimeError("부트스트랩: 임베딩 0개 — 소스 디코드 확인")

    X = np.stack(all_emb)
    fid = np.array(all_fid)

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    target = pick_target_cluster(km.labels_, fid)
    centroid = km.cluster_centers_[target]
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    mask = km.labels_ == target
    picks = nearest_distinct_files(X[mask], fid[mask], centroid, n=3)
    cand_idx = np.where(mask)[0]

    print(f"\n전현무 후보 클러스터={target} "
          f"(파일 {len(np.unique(fid[mask]))}개 커버, 윈도우 {mask.sum()}개)")
    print("아래 3개 클립을 들어보고 전현무가 맞는지 확인하세요:\n")
    for n, (f_, local_i) in enumerate(picks):
        gi = cand_idx[local_i]
        _, t0, wav = win_meta[gi]
        y, _ = sf.read(str(wav), dtype="float32")
        s = int(t0 * SR)
        seg = y[s:s + int(10 * SR)]
        out = ANCHOR_CAND / f"cand_{n + 1:02d}.wav"
        sf.write(str(out), seg, SR, subtype="PCM_16")
        print(f"  {out}")

    np.save(ANCHOR_NPY, centroid.astype(np.float32))
    print(f"\n임시 anchor 저장: {ANCHOR_NPY}")
    print("→ 후보가 전현무가 맞으면 그대로 --pilot 진행.")
    print("→ 아니면 --rebootstrap --k <다른값> 으로 재시도.")


def main() -> None:
    ap = argparse.ArgumentParser(description="전현무 단독 음성 추출")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--rebootstrap", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pilot-n", type=int, default=25)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sim-thr", type=float, default=0.80)
    args = ap.parse_args()

    if args.bootstrap or args.rebootstrap:
        run_bootstrap(k=args.k, force=args.rebootstrap)
    elif args.pilot:
        from .runner import run_extract

        run_extract(_short_clips(_list_sources(), args.pilot_n),
                    args.sim_thr, tag="pilot")
    elif args.full:
        from .runner import run_extract

        run_extract(_list_sources(), args.sim_thr, tag="full")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: extract_jhm.py 엔트리 작성**

`voicebox-clone-workspace/extract_jhm.py`:

```python
#!/usr/bin/env python3
"""전현무 단독 음성 추출 CLI. 사용법:
  python extract_jhm.py --bootstrap      # 전현무 anchor 자동 발견 + 확인 클립 생성
  python extract_jhm.py --pilot          # 25개 파일 파일럿 (임계 튜닝용)
  python extract_jhm.py --full           # 508개 전체
"""
from jhm_extract.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: import 무결성 확인** (runner는 다음 태스크에서 생성 — bootstrap 경로만 검증)

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -c "from jhm_extract.cli import run_bootstrap, main; print('cli import OK')" && cd ..`
Expected: `cli import OK`

- [ ] **Step 4: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/cli.py voicebox-clone-workspace/extract_jhm.py
git commit -m "feat(jhm): CLI + run_bootstrap (전현무 클러스터 자동발견 + 확인 게이트)"
```

---

### Task 10: runner.run_extract (파일럿/전체 공통 추출 루프)

**Files:**
- Create: `voicebox-clone-workspace/jhm_extract/runner.py`

Resemblyzer/실음원 결합부 → 명세의 파일럿 청취 게이트(spec §9)로 검증. 코드 완전 제공.

- [ ] **Step 1: runner.py 작성**

`voicebox-clone-workspace/jhm_extract/runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .cleanliness import cleanliness_gate
from .decode import cache_key, decode_to_wav16k
from .manifest import Manifest
from .segments import clip_from_turn, group_turns, sliding_window_embeddings

SR = 16000
WIN_S = 1.6
STEP_S = 0.4
HOP_MS = 50

# 상수 뒤 import는 의도적: cli↔runner 순환을 cli의 lazy import로 끊으므로 안전.
from .cli import ANCHOR_NPY, CACHE, OUT_DIR, _encoder  # noqa: E402


def _noise_floor(y: np.ndarray) -> float:
    hop = int(SR * HOP_MS / 1000)
    frame = hop
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    return float(np.percentile(rms, 10)) if rms.size else 1e-4


def run_extract(srcs: list[Path], sim_thr: float, tag: str) -> None:
    if not ANCHOR_NPY.exists():
        raise RuntimeError("anchor 없음 — 먼저 `--bootstrap` 실행")
    anchor = np.load(ANCHOR_NPY)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man = Manifest(OUT_DIR / f"manifest_{tag}.json")
    enc = _encoder()
    print(f"[{tag}] 소스 {len(srcs)}개, sim_thr={sim_thr}")

    for si, src in enumerate(srcs):
        key = None
        try:
            key = cache_key(src)
            if man.done(key):
                continue
            wav = decode_to_wav16k(src, CACHE)
            y, _ = sf.read(str(wav), dtype="float32")
        except Exception as e:  # noqa: BLE001
            man.add_error(src.name, f"decode: {e}")
            print(f"  [{si}] ERROR {src.name}: {e}")
            continue

        starts, embs = sliding_window_embeddings(y, enc, WIN_S, STEP_S)
        if embs.shape[0] == 0:
            if key:
                man.mark_done(key, 0)
            continue
        sims = embs @ anchor
        is_t = sims >= sim_thr
        turns = group_turns(is_t, starts, WIN_S, STEP_S,
                            gap_merge_s=0.8, min_turn_s=8.0)

        nf = _noise_floor(y)
        n_kept = 0
        for turn in turns:
            clip = clip_from_turn(turn, 10.0, 20.0)
            if clip is None:
                continue
            t0, t1 = clip
            seg = y[int(t0 * SR):int(t1 * SR)]
            cm = cleanliness_gate(seg, noise_floor=nf)
            if not cm.passed:
                continue
            dur = t1 - t0
            # i0,i1는 inclusive 윈도우 인덱스 → 슬라이스 end에 +1
            sim = float(sims[turn["i0"]:turn["i1"] + 1].mean())
            # 통과 클립은 gap_rms_ratio ≤ gap_ratio_max(3) 이라 clean_score∈[0.25,1];
            # max(0,..)는 방어용. 단조감소라 순위는 divisor와 무관하게 동일.
            clean_score = max(0.0, 1.0 - cm.gap_rms_ratio / 4.0)
            score = float(sim * np.sqrt(dur) * (0.5 + 0.5 * clean_score))
            simtag = int(round(sim * 100))
            name = f"jhm_{si:03d}_{int(t0)}s_{int(dur)}s_sim{simtag}.wav"
            sf.write(str(OUT_DIR / name), seg, SR, subtype="PCM_16")
            man.add_clip({
                "out": name, "src": src.name, "t0": float(t0),
                "dur": float(dur), "sim": sim,
                "gap_rms_ratio": cm.gap_rms_ratio,
                "voiced_ratio": cm.voiced_ratio,
                "spectral_flatness": cm.spectral_flatness,
                "score": score,
            })
            n_kept += 1
        if key:
            man.mark_done(key, n_kept)
        if si > 0 and si % 10 == 0:
            man.save()
        print(f"  [{si}] {src.name[:40]}  clips={n_kept}")

    # 점수순 정렬 후 매니페스트 최종 저장
    man.clips.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    man.save()
    s = man.stats()
    print(f"\n[{tag}] 완료: clips={s['n_clips']} "
          f"총 {s['total_seconds'] / 60:.1f}분 errors={s['n_errors']}")
    print(f"산출: {OUT_DIR}")
    print(f"매니페스트: {man.path}")
```

- [ ] **Step 2: import 무결성 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -c "from jhm_extract.runner import run_extract; print('runner import OK')" && cd ..`
Expected: `runner import OK`

- [ ] **Step 3: 전체 단위테스트 회귀 확인**

Run: `.venv/bin/python -m pytest voicebox-clone-workspace -v`
Expected: 모든 테스트 passed (회귀 없음)

- [ ] **Step 4: Commit**

```bash
git add voicebox-clone-workspace/jhm_extract/runner.py
git commit -m "feat(jhm): runner.run_extract (화자매칭+클린니스+점수정렬, resumable)"
```

---

### Task 11: 부트스트랩 실행 + 전현무 확인 게이트 (대화형 검증)

**Files:** 없음 (실행 + 사용자 청취)

- [ ] **Step 1: 부트스트랩 실행**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python extract_jhm.py --bootstrap && cd ..`
Expected: 콘솔에 `전현무 후보 클러스터=N (파일 M개 커버 ...)` 출력 + `_anchor_candidates/cand_01.wav ~ cand_03.wav` 3개 생성, `_anchor/anchor.npy` 저장

- [ ] **Step 2: 사용자 청취 확인 게이트 (필수)**

`voicebox-clone-workspace/_anchor_candidates/cand_01.wav`, `cand_02.wav`, `cand_03.wav`를 사용자가 직접 듣는다.
- 3개 모두 전현무 목소리 → 다음 태스크 진행
- 아니면: `../.venv/bin/python extract_jhm.py --rebootstrap --k 8` (또는 다른 k) 재시도 후 본 게이트 재확인

**사용자 승인 없이는 다음 태스크로 진행 금지.**

---

### Task 12: 파일럿 실행 + 임계값 튜닝 (대화형 검증)

**Files:** 튜닝 시에만 `voicebox-clone-workspace/jhm_extract/cleanliness.py` 기본값 또는 `--sim-thr` 인자 조정

- [ ] **Step 1: 파일럿 실행 (25개)**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python extract_jhm.py --pilot && cd ..`
Expected: `[pilot] 완료: clips=... 총 N분 errors=...`, `jeonhyunmoo_clean/jhm_*.wav` 생성, `jeonhyunmoo_clean/manifest_pilot.json` 생성

- [ ] **Step 2: 산출 클립 무작위 청취**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -c "import json,random; m=json.load(open('jeonhyunmoo_clean/manifest_pilot.json')); cs=m['clips']; [print(c['out'], round(c['sim'],3), round(c['gap_rms_ratio'],2)) for c in random.sample(cs, min(8,len(cs)))]" && cd ..`
사용자가 출력된 클립 6~8개를 들어본다.

- [ ] **Step 3: 임계값 판단 (사용자 피드백 기반)**

- 다른 화자/겹침이 섞임 → `--sim-thr` 상향(예: 0.83) 후 Step 1 재실행 (`jeonhyunmoo_clean/` 비우고)
- BGM 잔존 → `cleanliness.py`의 `gap_ratio_max` 하향(예: 3.0) 후 재실행
- 수율 0에 가까움 → `--sim-thr` 하향(0.77) 또는 `gap_ratio_max` 상향(5.0)
- 깨끗하고 전현무 단독 확인 → 다음 태스크 진행

재실행 전: `rm -rf voicebox-clone-workspace/jeonhyunmoo_clean voicebox-clone-workspace/_cache` 는 캐시 재사용 위해 `jeonhyunmoo_clean`만 비울 것: `rm -rf voicebox-clone-workspace/jeonhyunmoo_clean`

- [ ] **Step 4: 튜닝값 커밋 (변경 시에만)**

```bash
git add voicebox-clone-workspace/jhm_extract/cleanliness.py
git commit -m "tune(jhm): 파일럿 청취 기반 클린니스 임계값 조정"
```

**사용자 승인 없이는 전체 실행(Task 13) 진행 금지.**

---

### Task 13: 전체 실행 + 최종 검증

**Files:** 없음 (실행 + 통계 검토)

- [ ] **Step 1: 전체 508개 실행 (장시간 — 백그라운드 권장)**

Run: `cd voicebox-clone-workspace && nohup ../.venv/bin/python extract_jhm.py --full > full.log 2>&1 & cd ..`
중간 진행: `tail -f voicebox-clone-workspace/full.log`
resumable — 중단되어도 재실행 시 `manifest_full.json`의 done 소스 스킵.

- [ ] **Step 2: 완료 후 통계 검토**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -c "import json; m=json.load(open('jeonhyunmoo_clean/manifest_full.json')); print(m['stats'])" && cd ..`
Expected: `{'n_clips': ..., 'total_seconds': ..., 'n_errors': ..., 'n_done_sources': ...}` — 전현무 단독 클립 다수, n_done_sources ≈ 508

- [ ] **Step 3: 무작위 spot-check 5개 (사용자 청취)**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -c "import json,random; cs=json.load(open('jeonhyunmoo_clean/manifest_full.json'))['clips']; [print('jeonhyunmoo_clean/'+c['out']) for c in random.sample(cs,5)]" && cd ..`
사용자가 5개를 듣고 전현무 단독·BGM 거의 없음 확인.

- [ ] **Step 4: 결과 요약 출력**

산출: `voicebox-clone-workspace/jeonhyunmoo_clean/` — 전현무 단독 10~20초 클립 전부(점수순 매니페스트), 16k mono PCM16.
회귀 확인: 기존 워크스페이스 스크립트/데이터 무변경(신규 파일만 추가).

- [ ] **Step 5: 최종 단위테스트 회귀 확인 + Commit (코드만)**

```bash
.venv/bin/python -m pytest voicebox-clone-workspace -q
git add voicebox-clone-workspace/jhm_extract voicebox-clone-workspace/extract_jhm.py voicebox-clone-workspace/tests voicebox-clone-workspace/pytest.ini voicebox-clone-workspace/.gitignore
git commit -m "feat(jhm): 전현무 단독 음성 추출 파이프라인 완료"
```

(산출 wav/manifest/캐시는 `.gitignore`로 제외되어 커밋 안 됨 — 의도된 동작)

---

## Self-Review

**1. Spec coverage:**
- §4 아키텍처(decode/anchor/cleanliness/extract/run_batch) → Task 1·2(decode), 7(anchor 선택), 3(cleanliness), 4·5·6(segments), 8(manifest), 10(runner) ✓
- §6 부트스트랩(최다 파일 커버리지, 확인 게이트, top-N 정제) → Task 7·9·11 ✓ (반복정제는 centroid + 클러스터 평균으로 구현, spec의 top-200 의도 충족)
- §7 클린니스(sim, gap-energy, voiced, 클리핑, flatness, 길이) → Task 3 전부 + Task 10에서 sim 게이트 ✓
- §8 에러처리/재개(명시 로깅, 캐시, manifest done) → Task 2(캐시), 8(manifest), 10(에러 기록·resumable) ✓
- §9 검증(부트스트랩·파일럿 청취, 통계, spot-check) → Task 11·12·13 ✓
- §10 산출 스펙(폴더/포맷/파일명/manifest) → Task 10 구현 일치 ✓
- §11 런타임(백그라운드) → Task 13 nohup ✓

**2. Placeholder scan:** "TODO/TBD/적절히 처리" 없음. 모든 코드 단계에 완전 코드 제공. 대화형 검증 태스크(11·12·13)는 정확한 명령+기대출력 명시.

**3. Type consistency:** `cleanliness_gate`→`CleanMetrics`(.passed/.reasons/.gap_rms_ratio/.as_dict) 일관. `group_turns`→dict(t0,t1,dur,i0,i1). `clip_from_turn`(단일 턴→클립|None) / `select_clips`(다중) 시그니처 일관.

**Self-review에서 발견·교정 완료 (1):** 초안의 Task 10 `for (t0,t1),turn in zip(clips,turns)` 는 `select_clips`가 짧은 턴을 스킵하면 clips·turns 정렬이 어긋나는 버그였다. → Task 5에 단일책임 `clip_from_turn` 추가, Task 10 runner를 `for turn in turns: clip_from_turn(turn)` 턴별 루프로 교정하여 턴-클립-sim 페어링을 구조적으로 보장.

**실행 중 발견·교정 완료 (2):** Task 3 구현 시 서브에이전트가 BLOCKED로 적발 — 초안은 `voiced_mult=4.0 == gap_ratio_max=4.0` 이라 gap-energy 분기가 영원히 미발동(pause 프레임은 정의상 rms<noise_floor*4 이므로 gap_rms_ratio≤4=gap_ratio_max)하는 dead code였고, `speech_with_music_bed` 톤(0.03)이 과대해 gap이 사라져 no_pauses로만 잡혀 `test_music_bed_rejected_by_gap_energy`가 실패했다. → `voiced_mult=8.0`, `gap_ratio_max=3.0`(불변식 gap_ratio_max<voiced_mult), 픽스처 톤 0.03→0.007 로 교정. 6개 픽스처 전부 의도한 사유로 통과함을 수식 검증. spec §7 수치도 동기화.
