# 전현무 GPT-SoVITS 학습 데이터셋 파이프라인 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `전현무음성/`의 m4a에서 전현무 단독 깨끗한 발화를 추출해 32kHz 개별 wav 세그먼트 + GPT-SoVITS `.list` 전사 파일로 패키징하고, Windows(CUDA)에서 재실행 가능한 이식형 CLI를 만든다.

**Architecture:** 기존 `jhm/` 패키지(검증된 decode→Demucs→Silero VAD→resemblyzer→anchor 임계)를 재사용하고, GPT-SoVITS 어댑테이션(개별 세그먼트 32k export, faster-whisper 전사, `.list` 빌더, montage 귀확인 반자동 승인)을 `jhm/gptsovits.py` + `build_gptsovits_dataset.py` CLI로 추가한다. 순도 최우선: 자동 sim≥T + montage 귀확인 후 reject 반영.

**Tech Stack:** Python 3.13, PyTorch(MPS/CUDA), demucs(htdemucs), resemblyzer, librosa, soundfile, faster-whisper, pytest.

**Spec:** `docs/superpowers/specs/2026-05-21-jeonhyunmoo-gptsovits-dataset-design.md`

---

## 사전 컨텍스트 (실행자 필독)

- **작업 디렉터리:** `voicebox-clone-workspace/` (이하 경로는 이 폴더 기준). 파이썬은 `../.venv/bin/python` (프로젝트 루트의 venv).
- **기존 테스트 깨짐:** `tests/test_anchor.py` 등 5개는 삭제된 `jhm_extract` 패키지를 import해 **collection 에러**가 난다. 이번 작업 범위 밖이므로 **고치지 말 것**. 신규 테스트는 **명시적 파일 경로로** 실행한다(`pytest tests/test_xxx.py`). 전체 `pytest`는 collection 에러로 중단되니 쓰지 말 것.
- `tests/conftest.py`가 워크스페이스 루트를 `sys.path`에 넣으므로 `from jhm.xxx import ...`가 동작한다.
- 모든 커밋 메시지 끝에:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- 현재 브랜치: `feature/jeonhyunmoo-voice-extraction`.

## File Structure

| 파일 | 책임 |
|---|---|
| `jhm/device.py` [생성] | `pick_device()` — cuda>mps>cpu 자동 선택 |
| `jhm/gptsovits.py` [생성] | 세그먼트 32k 슬라이스/저장, manifest IO, reject 파싱, montage 빌드, faster-whisper 전사, `.list` 빌더 |
| `jhm/core.py` [수정] | demucs 보컬 추출을 `demucs_vocal()`로 추출(리팩터), native SR 반환 |
| `build_gptsovits_dataset.py` [생성] | CLI: `probe` / `rebuild-anchor` / `extract` / `montage` / `finalize` |
| `tests/test_device.py` [생성] | `pick_device` 단위테스트 |
| `tests/test_gptsovits.py` [생성] | 순수 함수 단위테스트(remap/parse/build_list/manifest/relocate) |

산출 데이터(코드 아님, git 미커밋): `jhm_work/gptsovits/staging/…`, `jhm_work/gptsovits/dataset/…`

---

## Task 1: `pick_device()` 디바이스 자동 선택

**Files:**
- Create: `jhm/device.py`
- Test: `tests/test_device.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_device.py`:
```python
from jhm.device import pick_device


def test_prefers_cuda_when_available():
    avail = {"cuda": True, "mps": True, "cpu": True}
    assert pick_device(available=avail) == "cuda"


def test_falls_back_to_mps_without_cuda():
    avail = {"cuda": False, "mps": True, "cpu": True}
    assert pick_device(available=avail) == "mps"


def test_falls_back_to_cpu():
    avail = {"cuda": False, "mps": False, "cpu": True}
    assert pick_device(available=avail) == "cpu"


def test_prefer_honored_when_available():
    avail = {"cuda": True, "mps": True, "cpu": True}
    assert pick_device(prefer="mps", available=avail) == "mps"


def test_prefer_ignored_when_unavailable():
    avail = {"cuda": True, "mps": False, "cpu": True}
    assert pick_device(prefer="mps", available=avail) == "cuda"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jhm.device'`

- [ ] **Step 3: 최소 구현**

`jhm/device.py`:
```python
"""실행 디바이스 자동 선택 — Mac(MPS)/Windows·Linux(CUDA)/CPU 이식성."""
from __future__ import annotations


def pick_device(prefer: str | None = None, *, available: dict[str, bool] | None = None) -> str:
    """cuda > mps > cpu 순으로 사용 가능한 디바이스명 반환.

    prefer가 주어지고 사용 가능하면 그것을 우선한다.
    available을 주면(테스트용) torch 조회를 건너뛴다.
    """
    if available is None:
        import torch

        available = {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        }
    if prefer and available.get(prefer):
        return prefer
    for d in ("cuda", "mps", "cpu"):
        if available.get(d):
            return d
    return "cpu"
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_device.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add voicebox-clone-workspace/jhm/device.py voicebox-clone-workspace/tests/test_device.py
git commit -m "feat(jhm): pick_device() 디바이스 자동 선택 (cuda>mps>cpu)"
```

---

## Task 2: gptsovits 순수 헬퍼 (remap/parse/build_list/manifest/relocate)

**Files:**
- Create: `jhm/gptsovits.py` (순수 함수 부분)
- Test: `tests/test_gptsovits.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_gptsovits.py`:
```python
import json

from jhm.gptsovits import (
    build_list_lines,
    parse_reject,
    relocate_lines,
    remap_segments,
    seg_id,
    read_manifest,
    write_manifest,
)


def test_seg_id_zero_padded():
    assert seg_id(1) == "jhm_0001"
    assert seg_id(42) == "jhm_0042"


def test_remap_16k_to_32k_doubles_indices():
    segs = [{"start": 16000, "end": 24000, "sim": 0.9}]
    out = remap_segments(segs, sr_from=16000, sr_to=32000)
    assert out[0]["start"] == 32000
    assert out[0]["end"] == 48000
    assert out[0]["sim"] == 0.9  # 다른 키 보존


def test_parse_reject_ignores_comments_and_blanks():
    text = "# 헤더 주석\njhm_0003\n\n  jhm_0007  # 잡음\n#jhm_0009\n"
    assert parse_reject(text) == {"jhm_0003", "jhm_0007"}


def test_build_list_lines_format_and_skip_empty():
    entries = [
        {"relpath": "wavs/jhm_0001.wav", "text": "안녕하세요"},
        {"relpath": "wavs/jhm_0002.wav", "text": "   "},   # 빈 전사 → 제외
        {"relpath": "wavs/jhm_0003.wav", "text": " 반갑습니다 "},
    ]
    lines = build_list_lines(entries, speaker="jhm", lang="ko")
    assert lines == [
        "wavs/jhm_0001.wav|jhm|ko|안녕하세요",
        "wavs/jhm_0003.wav|jhm|ko|반갑습니다",
    ]


def test_relocate_lines_rewrites_first_field():
    lines = ["wavs/jhm_0001.wav|jhm|ko|안녕"]
    out = relocate_lines(lines, base="D:/gpt-sovits/jhm", sep="/")
    assert out == ["D:/gpt-sovits/jhm/wavs/jhm_0001.wav|jhm|ko|안녕"]


def test_manifest_roundtrip(tmp_path):
    entries = [{"id": "jhm_0001", "source": "a.m4a", "start_s": 1.0, "end_s": 3.0, "dur": 2.0, "sim": 0.88}]
    p = tmp_path / "segments.json"
    write_manifest(p, entries)
    assert read_manifest(p) == entries
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jhm.gptsovits'`

- [ ] **Step 3: 최소 구현 (순수 함수)**

`jhm/gptsovits.py`:
```python
"""GPT-SoVITS 학습 데이터셋 어댑테이션.

전현무 단독 발화 세그먼트를 32k 개별 wav로 export, faster-whisper(ko) 전사,
GPT-SoVITS `.list` 빌드, montage 귀확인용 미리듣기/ reject 반영.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def seg_id(index: int) -> str:
    """1-기반 정수 → 'jhm_0001' 형식 세그먼트 id."""
    return f"jhm_{index:04d}"


def remap_segments(segs: list[dict], sr_from: int = 16000, sr_to: int = 32000) -> list[dict]:
    """샘플 인덱스 세그먼트를 sr_from→sr_to 기준으로 변환(다른 키 보존)."""
    r = sr_to / sr_from
    out = []
    for s in segs:
        out.append({**s, "start": int(round(s["start"] * r)), "end": int(round(s["end"] * r))})
    return out


def parse_reject(text: str) -> set[str]:
    """reject.txt → 제외할 seg id 집합. '#' 주석·공백 무시."""
    ids: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def build_list_lines(entries: list[dict], speaker: str = "jhm", lang: str = "ko") -> list[str]:
    """[{relpath, text}] → GPT-SoVITS `.list` 라인. 빈 전사 제외."""
    lines: list[str] = []
    for e in entries:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{e['relpath']}|{speaker}|{lang}|{text}")
    return lines


def relocate_lines(lines: list[str], base: str, sep: str = "/") -> list[str]:
    """`.list` 첫 필드(상대경로)를 base 기준 절대경로로 재작성."""
    out: list[str] = []
    base = base.rstrip("/\\")
    for ln in lines:
        parts = ln.split("|", 1)
        out.append(f"{base}{sep}{parts[0]}|{parts[1]}" if len(parts) == 2 else ln)
    return out


def write_manifest(path: Path, entries: list[dict]) -> None:
    Path(path).write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add voicebox-clone-workspace/jhm/gptsovits.py voicebox-clone-workspace/tests/test_gptsovits.py
git commit -m "feat(jhm): gptsovits 순수 헬퍼 (remap/parse_reject/build_list/relocate/manifest)"
```

---

## Task 3: `core.py` — demucs 보컬 추출 리팩터 (native SR 반환)

기존 `separate_vocals`의 demucs 부분을 `demucs_vocal()`로 분리해, 32k export용으로 native(44.1k) 보컬을 재사용할 수 있게 한다. `separate_vocals`의 외부 동작(16k 반환, 캐시)은 그대로 유지(회귀 방지).

**Files:**
- Modify: `jhm/core.py:64-101` (`separate_vocals` 본문)

- [ ] **Step 1: `demucs_vocal()` 추가**

`jhm/core.py`의 `separate_vocals` 정의 바로 위에 추가:
```python
DEMUCS_SR = 44100  # htdemucs native sample rate


def demucs_vocal(src: Path, device: str = "mps") -> tuple[np.ndarray, int]:
    """htdemucs 보컬 분리 → (mono float32 보컬, native SR=44100). 캐시 안 함.

    호출측에서 1회 호출 후 원하는 SR로 리샘플(16k 분석 / 32k export)에 재사용.
    """
    import torch
    from demucs.apply import apply_model

    model = _demucs_model()
    stereo = decode(src, model.samplerate, mono=False)  # (2, N) float32
    mix = torch.from_numpy(stereo)
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = (mix - mean) / std
    log(f"  [demucs] {src.name} (device={device}) …")
    with torch.no_grad():
        out = apply_model(model, mix[None], device=device, split=True, overlap=0.25, progress=False)[0]
    out = out * std + mean
    voc_idx = model.sources.index("vocals")
    vocals = out[voc_idx].mean(0).cpu().numpy().astype(np.float32)  # mono @ 44100
    return vocals, int(model.samplerate)
```

- [ ] **Step 2: `separate_vocals`를 `demucs_vocal` 사용으로 교체**

`jhm/core.py`의 기존 `separate_vocals` 본문(캐시 체크 이후 demucs 직접 호출 부분)을 다음으로 교체:
```python
def separate_vocals(src: Path, device: str = "mps") -> np.ndarray:
    """Demucs htdemucs 보컬 분리 → 16k mono 보컬(BGM 제거). 캐시됨."""
    cached = _CACHE / f"vocals_{_key(src)}.wav"
    if cached.exists():
        wav, _ = sf.read(str(cached), dtype="float32")
        return wav

    import librosa

    vocals, sr = demucs_vocal(src, device=device)
    y = librosa.resample(vocals, orig_sr=sr, target_sr=SR).astype(np.float32)
    sf.write(str(cached), y, SR)
    return y
```

- [ ] **Step 3: import 스모크 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -c "from jhm.core import demucs_vocal, separate_vocals; print('ok', demucs_vocal.__name__)"`
Expected: `ok demucs_vocal`

- [ ] **Step 4: 커밋**

```bash
git add voicebox-clone-workspace/jhm/core.py
git commit -m "refactor(jhm): demucs_vocal() 분리 — native SR 반환으로 32k export 재사용"
```

> 주: demucs는 실오디오 I/O라 단위테스트 대상이 아니다. Task 8(probe 실측)에서 실제 동작을 검증한다.

---

## Task 4: 세그먼트 32k 추출/저장 + CLI `extract`

**Files:**
- Modify: `jhm/gptsovits.py` (세그먼트 export 함수 추가)
- Test: `tests/test_gptsovits.py` (슬라이스/저장 단위테스트 추가)

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_gptsovits.py` 끝에 추가:
```python
import numpy as np
from jhm.gptsovits import slice_segments, save_segments


def test_slice_segments_cuts_by_remapped_indices():
    vocal32 = np.arange(96000, dtype=np.float32)  # 3s @ 32k
    segs16 = [{"start": 16000, "end": 32000, "sim": 0.9, "dur": 1.0}]  # 1.0~2.0s
    clips = slice_segments(vocal32, segs16, sr16=16000, sr32=32000)
    assert len(clips) == 1
    clip, meta = clips[0]
    assert clip[0] == 32000 and clip[-1] == 63999
    assert meta["start_s"] == 1.0 and meta["end_s"] == 2.0


def test_save_segments_writes_wavs_and_manifest(tmp_path):
    vocal32 = np.sin(np.linspace(0, 50, 96000)).astype(np.float32)
    segs16 = [
        {"start": 0, "end": 16000, "sim": 0.91, "dur": 1.0},
        {"start": 32000, "end": 48000, "sim": 0.85, "dur": 1.0},
    ]
    entries = save_segments(vocal32, segs16, source="ep01.m4a", out_dir=tmp_path,
                            start_index=1, sr16=16000, sr32=32000)
    assert [e["id"] for e in entries] == ["jhm_0001", "jhm_0002"]
    for e in entries:
        assert (tmp_path / f"{e['id']}.wav").exists()
        assert e["source"] == "ep01.m4a"
    assert entries[0]["sim"] == 0.91
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py -v`
Expected: FAIL — `ImportError: cannot import name 'slice_segments'`

- [ ] **Step 3: 구현 추가**

`jhm/gptsovits.py`에 추가:
```python
def slice_segments(vocal32: np.ndarray, segs16: list[dict],
                   sr16: int = 16000, sr32: int = 32000) -> list[tuple[np.ndarray, dict]]:
    """16k 인덱스 세그먼트를 32k 보컬에서 슬라이스. (clip32k, meta) 리스트."""
    segs32 = remap_segments(segs16, sr_from=sr16, sr_to=sr32)
    out = []
    n = len(vocal32)
    for s16, s32 in zip(segs16, segs32):
        a, b = max(0, s32["start"]), min(n, s32["end"])
        if b <= a:
            continue
        meta = {
            "start_s": round(s16["start"] / sr16, 3),
            "end_s": round(s16["end"] / sr16, 3),
            "dur": round((b - a) / sr32, 3),
            "sim": float(s16.get("sim", 0.0)),
        }
        out.append((vocal32[a:b].astype(np.float32), meta))
    return out


def save_segments(vocal32: np.ndarray, segs16: list[dict], source: str, out_dir: Path,
                  start_index: int = 1, sr16: int = 16000, sr32: int = 32000) -> list[dict]:
    """채택 세그먼트를 32k wav로 저장 + manifest 엔트리 리스트 반환."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    idx = start_index
    for clip, meta in slice_segments(vocal32, segs16, sr16=sr16, sr32=sr32):
        sid = seg_id(idx)
        peak = float(np.max(np.abs(clip))) or 1.0
        y = (clip / peak * 0.97).astype(np.float32)
        sf.write(str(out_dir / f"{sid}.wav"), y, sr32)
        entries.append({"id": sid, "source": source, **meta})
        idx += 1
    return entries
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: CLI `extract` 추가**

`build_gptsovits_dataset.py` (없으면 생성). 우선 공용 헤더 + `extract` 핸들러:
```python
"""전현무 GPT-SoVITS 데이터셋 빌드 CLI.

서브커맨드: probe / rebuild-anchor / extract / montage / finalize
사용: ../.venv/bin/python build_gptsovits_dataset.py <cmd> [opts]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jhm.core import SR, cosine, demucs_vocal, embed_segments, log, vad_segments
from jhm.device import pick_device
from jhm import gptsovits as G

WORK = Path(__file__).resolve().parent
SRC_DIR = WORK / "전현무음성"
ANCHOR = WORK / "jhm_work" / "anchor" / "anchor.npy"
GS = WORK / "jhm_work" / "gptsovits"
STAGING = GS / "staging"
SEG_DIR = STAGING / "segments"
MANIFEST = STAGING / "segments.json"
EXPORT_SR = 32000


def _sources(limit: int | None) -> list[Path]:
    files = sorted(SRC_DIR.glob("*.m4a"))
    return files[:limit] if limit else files


def _anchor() -> np.ndarray:
    if not ANCHOR.exists():
        raise RuntimeError("anchor.npy 없음 — 먼저 `rebuild-anchor` 실행")
    return np.load(ANCHOR)


def cmd_extract(args: argparse.Namespace) -> int:
    device = pick_device()
    anchor = _anchor()
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if MANIFEST.exists():
        entries = G.read_manifest(MANIFEST)
    done = {e["source"] for e in entries}
    next_idx = len(entries) + 1
    import librosa

    for src in _sources(args.limit):
        if src.name in done:
            log(f"[skip] {src.name} (이미 추출됨)")
            continue
        try:
            vocals, vsr = demucs_vocal(src, device=device)
            voc16 = librosa.resample(vocals, orig_sr=vsr, target_sr=SR).astype(np.float32)
            voc32 = librosa.resample(vocals, orig_sr=vsr, target_sr=EXPORT_SR).astype(np.float32)
            segs = vad_segments(voc16)
            embs, kept = embed_segments(voc16, segs)
            if len(kept) == 0:
                log(f"[extract] {src.name}: 발화 0")
                continue
            sims = cosine(embs, anchor)
            chosen = [{**kept[i], "sim": float(sims[i])} for i in range(len(kept)) if sims[i] >= args.threshold]
            new = G.save_segments(voc32, chosen, source=src.name, out_dir=SEG_DIR,
                                  start_index=next_idx, sr16=SR, sr32=EXPORT_SR)
            next_idx += len(new)
            entries.extend(new)
            G.write_manifest(MANIFEST, entries)
            log(f"[extract] {src.name}: {len(chosen)}/{len(kept)} 채택 (T={args.threshold})")
        except Exception as exc:  # noqa: BLE001 — 한 파일 실패가 전체를 막지 않도록
            log(f"[extract] {src.name} 실패: {type(exc).__name__}: {exc}")
    total = sum(e["dur"] for e in entries)
    log(f"[extract] 총 {len(entries)}개 세그먼트 / {total/60:.1f}분 → {MANIFEST}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--threshold", type=float, required=True)
    pe.add_argument("--limit", type=int, default=None)
    pe.set_defaults(func=cmd_extract)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: CLI 파싱 스모크 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py extract --help`
Expected: usage 출력에 `--threshold`, `--limit` 표시

- [ ] **Step 7: 커밋**

```bash
git add voicebox-clone-workspace/jhm/gptsovits.py voicebox-clone-workspace/tests/test_gptsovits.py voicebox-clone-workspace/build_gptsovits_dataset.py
git commit -m "feat(jhm): 32k 세그먼트 export + extract CLI (anchor 임계 채택)"
```

---

## Task 5: montage 미리듣기 + reject + CLI `montage`/`rebuild-anchor`

**Files:**
- Modify: `jhm/gptsovits.py` (montage/timeline 빌드)
- Modify: `build_gptsovits_dataset.py` (`montage`, `rebuild-anchor` 핸들러)
- Test: `tests/test_gptsovits.py` (timeline 포맷 테스트)

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_gptsovits.py` 끝에 추가:
```python
from jhm.gptsovits import timeline_lines


def test_timeline_lines_format():
    placed = [
        {"id": "jhm_0001", "offset_s": 0.0, "sim": 0.91, "source": "ep01.m4a"},
        {"id": "jhm_0002", "offset_s": 3.2, "sim": 0.85, "source": "ep02.m4a"},
    ]
    lines = timeline_lines(placed)
    assert lines[0] == "00:00.0  jhm_0001  sim=0.910  ep01.m4a"
    assert lines[1] == "00:03.2  jhm_0002  sim=0.850  ep02.m4a"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py::test_timeline_lines_format -v`
Expected: FAIL — `ImportError: cannot import name 'timeline_lines'`

- [ ] **Step 3: 구현 추가**

`jhm/gptsovits.py`에 추가:
```python
def _mmss(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


def timeline_lines(placed: list[dict]) -> list[str]:
    """montage 배치 정보 → 'mm:ss.s  seg_id  sim=x.xxx  source' 라인."""
    return [f"{_mmss(p['offset_s'])}  {p['id']}  sim={p['sim']:.3f}  {p['source']}" for p in placed]


def build_montage(entries: list[dict], seg_dir: Path, out_wav: Path,
                  sr: int = 32000, take_s: float = 3.0, gap_s: float = 0.3) -> list[dict]:
    """세그먼트 앞부분을 이어붙인 미리듣기 wav 생성 + 배치(timeline) 반환."""
    seg_dir, out_wav = Path(seg_dir), Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(sr * gap_s), np.float32)
    parts, placed, offset = [], [], 0.0
    for e in entries:
        wav, _ = sf.read(str(seg_dir / f"{e['id']}.wav"), dtype="float32")
        take = wav[: int(sr * take_s)]
        parts.append(take)
        parts.append(gap)
        placed.append({"id": e["id"], "offset_s": round(offset, 1), "sim": e["sim"], "source": e["source"]})
        offset += (len(take) + len(gap)) / sr
    y = (np.concatenate(parts) if parts else np.zeros(1, np.float32)).astype(np.float32)
    sf.write(str(out_wav), y, sr)
    return placed
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: CLI `montage` + `rebuild-anchor` 핸들러 추가**

`build_gptsovits_dataset.py`에 추가(그리고 `build_parser`에 등록):
```python
PREVIEW = STAGING / "preview"
REJECT = STAGING / "reject.txt"
CLEAN_REFS = ["montage_cluster_01.wav", "montage_cluster_05.wav",
              "montage_cluster_07.wav", "montage_cluster_16.wav"]


def cmd_rebuild_anchor(args: argparse.Namespace) -> int:
    from resemblyzer import VoiceEncoder, preprocess_wav

    ref_dir = WORK / "jhm_work" / "anchor"
    enc = VoiceEncoder(device="cpu", verbose=False)
    embs = []
    for name in CLEAN_REFS:
        p = ref_dir / name
        if not p.exists():
            raise RuntimeError(f"클린 ref 없음: {p}")
        embs.append(enc.embed_utterance(preprocess_wav(str(p))))
    anchor = np.mean(embs, axis=0)
    anchor = (anchor / (np.linalg.norm(anchor) + 1e-9)).astype(np.float32)
    np.save(ANCHOR, anchor)
    log(f"[anchor] 클린 {len(embs)}-ref 평균 → {ANCHOR} (norm={np.linalg.norm(anchor):.3f})")
    return 0


def cmd_montage(args: argparse.Namespace) -> int:
    entries = G.read_manifest(MANIFEST)
    entries_sorted = sorted(entries, key=lambda e: e["sim"], reverse=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    placed = G.build_montage(entries_sorted, SEG_DIR, PREVIEW / "montage_all.wav", sr=EXPORT_SR)
    (PREVIEW / "timeline.txt").write_text("\n".join(G.timeline_lines(placed)) + "\n", encoding="utf-8")
    # 소스별 montage
    by_src: dict[str, list[dict]] = {}
    for e in entries_sorted:
        by_src.setdefault(e["source"], []).append(e)
    for src, es in by_src.items():
        safe = "".join(c if c.isalnum() else "_" for c in src)[:60]
        G.build_montage(es, SEG_DIR, PREVIEW / f"montage_{safe}.wav", sr=EXPORT_SR)
    if not REJECT.exists():
        REJECT.write_text(
            "# 제외할 세그먼트 id를 한 줄에 하나씩 적으세요 (예: jhm_0003).\n"
            "# '#' 뒤는 주석. preview/timeline.txt로 id↔시각 확인.\n",
            encoding="utf-8",
        )
    log(f"[montage] {len(placed)}개 → {PREVIEW}/montage_all.wav, timeline.txt, reject.txt")
    return 0
```
`build_parser`에 등록(기존 함수 내부에 추가):
```python
    sub.add_parser("rebuild-anchor").set_defaults(func=cmd_rebuild_anchor)
    sub.add_parser("montage").set_defaults(func=cmd_montage)
```

- [ ] **Step 6: CLI 스모크 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py --help`
Expected: subcommand 목록에 `rebuild-anchor`, `montage`, `extract` 표시

- [ ] **Step 7: 커밋**

```bash
git add voicebox-clone-workspace/jhm/gptsovits.py voicebox-clone-workspace/tests/test_gptsovits.py voicebox-clone-workspace/build_gptsovits_dataset.py
git commit -m "feat(jhm): montage 미리듣기/timeline/reject + rebuild-anchor·montage CLI"
```

---

## Task 6: faster-whisper 전사 + CLI `finalize` (데이터셋 패키징)

**Files:**
- Modify: `jhm/gptsovits.py` (전사 + 패키징 헬퍼)
- Modify: `build_gptsovits_dataset.py` (`finalize` 핸들러)
- Test: `tests/test_gptsovits.py` (requirements/README/relocate 생성 텍스트 테스트)

- [ ] **Step 1: faster-whisper 설치**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pip install faster-whisper`
Expected: `Successfully installed faster-whisper-...`
검증: `../.venv/bin/python -c "import faster_whisper; print(faster_whisper.__version__)"`

- [ ] **Step 2: 실패하는 테스트 추가**

`tests/test_gptsovits.py` 끝에 추가:
```python
from jhm.gptsovits import relocate_script_text


def test_relocate_script_text_contains_relocate_lines_call():
    txt = relocate_script_text()
    assert "relocate_lines" in txt
    assert "argparse" in txt
    assert "jhm.list" in txt
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py::test_relocate_script_text_contains_relocate_lines_call -v`
Expected: FAIL — `ImportError: cannot import name 'relocate_script_text'`

- [ ] **Step 4: 구현 추가**

`jhm/gptsovits.py`에 추가:
```python
def transcribe_segments(seg_paths: list[Path], device: str = "cpu",
                        model_name: str = "large-v3") -> dict[str, str]:
    """faster-whisper(ko)로 각 wav 전사 → {파일명stem: text}."""
    from faster_whisper import WhisperModel

    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute)
    out: dict[str, str] = {}
    for p in seg_paths:
        segs, _ = model.transcribe(str(p), language="ko", beam_size=5)
        out[Path(p).stem] = "".join(s.text for s in segs).strip()
    return out


def relocate_script_text() -> str:
    """dataset/relocate_list.py 내용. Windows에서 .list 경로를 절대경로로 변환."""
    return '''"""jhm.list의 상대경로를 이 폴더 기준 절대경로로 변환.

사용(Windows): python relocate_list.py --base "D:/gpt-sovits/jhm_dataset"
"""
import argparse
from pathlib import Path


def relocate_lines(lines, base, sep="/"):
    out = []
    base = base.rstrip("/\\\\")
    for ln in lines:
        parts = ln.split("|", 1)
        out.append(f"{base}{sep}{parts[0]}|{parts[1]}" if len(parts) == 2 else ln)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dataset 폴더의 절대경로")
    ap.add_argument("--list", default="jhm.list")
    ap.add_argument("--out", default="jhm.abs.list")
    a = ap.parse_args()
    lines = Path(a.list).read_text(encoding="utf-8").splitlines()
    Path(a.out).write_text("\\n".join(relocate_lines(lines, a.base)) + "\\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(lines)} lines)")
'''


def requirements_text() -> str:
    return "\n".join([
        "torch", "torchaudio", "demucs", "librosa", "soundfile",
        "numpy", "resemblyzer", "faster-whisper", "scikit-learn",
    ]) + "\n"


def readme_text(n_segments: int, total_min: float, threshold: float) -> str:
    return f"""# 전현무 GPT-SoVITS 데이터셋

- 세그먼트: {n_segments}개 / 약 {total_min:.1f}분 (32kHz mono)
- 추출 임계(anchor sim): {threshold}
- 라벨 파일: `jhm.list` (형식: `상대경로|jhm|ko|전사`)

## Windows GPT-SoVITS 사용
1. 이 폴더를 GPT-SoVITS 작업 위치로 복사.
2. 절대경로 변환:
   `python relocate_list.py --base "이_폴더_절대경로"`  → `jhm.abs.list` 생성
3. GPT-SoVITS WebUI에서 라벨 파일=`jhm.abs.list`, 오디오 폴더=`wavs/` 지정 후 학습.

## 재추출(선택)
`jhm/`, `build_gptsovits_dataset.py`, `jhm_work/anchor/anchor.npy`를 함께 복사하고
`pip install -r requirements.txt` 후:
`python build_gptsovits_dataset.py rebuild-anchor && python build_gptsovits_dataset.py extract --threshold {threshold}`
"""
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_gptsovits.py -v`
Expected: PASS (10 passed)

- [ ] **Step 6: CLI `finalize` 핸들러 추가**

`build_gptsovits_dataset.py`에 추가(+ `build_parser` 등록):
```python
import shutil

DATASET = GS / "dataset"
WAVS = DATASET / "wavs"


def cmd_finalize(args: argparse.Namespace) -> int:
    entries = G.read_manifest(MANIFEST)
    rejected = G.parse_reject(REJECT.read_text(encoding="utf-8")) if REJECT.exists() else set()
    kept = [e for e in entries if e["id"] not in rejected]
    if not kept:
        log("[finalize] 남은 세그먼트 0 — 중단")
        return 1
    WAVS.mkdir(parents=True, exist_ok=True)
    for e in kept:
        shutil.copyfile(SEG_DIR / f"{e['id']}.wav", WAVS / f"{e['id']}.wav")
    device = pick_device()
    texts = G.transcribe_segments([WAVS / f"{e['id']}.wav" for e in kept],
                                  device="cuda" if device == "cuda" else "cpu")
    list_entries = [{"relpath": f"wavs/{e['id']}.wav", "text": texts.get(e["id"], "")} for e in kept]
    lines = G.build_list_lines(list_entries, speaker="jhm", lang="ko")
    (DATASET / "jhm.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DATASET / "relocate_list.py").write_text(G.relocate_script_text(), encoding="utf-8")
    (DATASET / "requirements.txt").write_text(G.requirements_text(), encoding="utf-8")
    total_min = sum(e["dur"] for e in kept) / 60
    (DATASET / "README.md").write_text(G.readme_text(len(lines), total_min, args.threshold), encoding="utf-8")
    log(f"[finalize] {len(lines)}개 라벨 / {total_min:.1f}분 → {DATASET}/jhm.list "
        f"(제외 {len(rejected)}, 빈전사 {len(kept)-len(lines)})")
    return 0
```
`build_parser`에 등록:
```python
    pf = sub.add_parser("finalize")
    pf.add_argument("--threshold", type=float, default=0.0, help="README 기록용")
    pf.set_defaults(func=cmd_finalize)
```

- [ ] **Step 7: CLI 스모크 + 커밋**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py finalize --help`
Expected: usage 출력
```bash
git add voicebox-clone-workspace/jhm/gptsovits.py voicebox-clone-workspace/tests/test_gptsovits.py voicebox-clone-workspace/build_gptsovits_dataset.py
git commit -m "feat(jhm): faster-whisper 전사 + finalize (.list/relocate/README/requirements)"
```

---

## Task 7: 전체 단위테스트 회귀 확인

- [ ] **Step 1: 신규 테스트 전체 통과 확인**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python -m pytest tests/test_device.py tests/test_gptsovits.py -v`
Expected: 모두 PASS (device 5 + gptsovits 10 = 15 passed)

> 주: 깨진 레거시 테스트(`test_anchor.py` 등)는 범위 밖 — 실행 대상에서 제외.

---

## Task 8: 실측 probe — 순도 임계 T 캘리브레이션 (empirical-first)

**Files:**
- Modify: `build_gptsovits_dataset.py` (`probe` 핸들러)

- [ ] **Step 1: `probe` 핸들러 추가 + 등록**

`build_gptsovits_dataset.py`에 추가:
```python
def cmd_probe(args: argparse.Namespace) -> int:
    import librosa

    device = pick_device()
    anchor = _anchor()
    thresholds = args.thresholds or [0.70, 0.75, 0.80, 0.85]
    grand: dict[float, float] = {t: 0.0 for t in thresholds}
    for src in _sources(args.n):
        try:
            vocals, vsr = demucs_vocal(src, device=device)
            voc16 = librosa.resample(vocals, orig_sr=vsr, target_sr=SR).astype(np.float32)
            segs = vad_segments(voc16)
            embs, kept = embed_segments(voc16, segs)
            if len(kept) == 0:
                log(f"[probe] {src.name}: 발화 0"); continue
            sims = cosine(embs, anchor)
            log(f"[probe] {src.name}: 발화 {len(kept)}개 "
                f"sim mean={sims.mean():.3f} p75={np.percentile(sims,75):.3f} max={sims.max():.3f}")
            for t in thresholds:
                grand[t] += sum(kept[i]["dur"] for i in range(len(kept)) if sims[i] >= t)
        except Exception as exc:  # noqa: BLE001
            log(f"[probe] {src.name} 실패: {type(exc).__name__}: {exc}")
    log("=== 임계별 누적 채택 분량 (probe " + f"{args.n}개 파일) ===")
    for t in thresholds:
        log(f"  T={t:.2f} → {grand[t]/60:.1f}분")
    return 0
```
`build_parser` 등록:
```python
    pp = sub.add_parser("probe")
    pp.add_argument("--n", type=int, default=6)
    pp.add_argument("--thresholds", type=float, nargs="*", default=None)
    pp.set_defaults(func=cmd_probe)
```

- [ ] **Step 2: anchor 재생성(클린 4-ref)**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py rebuild-anchor`
Expected: `[anchor] 클린 4-ref 평균 → .../anchor.npy (norm=1.000)`

- [ ] **Step 3: probe 실행 (6개 파일)**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py probe --n 6 2>&1 | tail -20`
Expected: 파일별 sim 분포 + 임계별 누적 분량(분) 리포트

- [ ] **Step 4: 커밋 + 사용자 체크포인트**

```bash
git add voicebox-clone-workspace/build_gptsovits_dataset.py
git commit -m "feat(jhm): probe 임계 캘리브레이션 핸들러"
```
**사용자에게 보고:** 임계별 수율(분)을 제시하고, "순도 최우선" 기준으로 권장 T(예: p75 이상 또는 0.80~0.85)를 제안 → 사용자 확정 받기.

---

## Task 9: 본 추출 + montage 귀확인 (사용자 승인 게이트)

- [ ] **Step 1: 전체 추출 (확정 T 적용)**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py extract --threshold <확정T> 2>&1 | tail -30`
Expected: 파일별 채택 로그 + `총 N개 세그먼트 / M분`

- [ ] **Step 2: montage 생성**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py montage`
Expected: `staging/preview/montage_all.wav`, `timeline.txt`, `reject.txt` 생성

- [ ] **Step 3: 사용자 귀확인 체크포인트**

**사용자에게:** `staging/preview/montage_all.wav`(sim 내림차순)와 소스별 montage를 들어보고, 전현무 아닌 세그먼트 id를 `staging/reject.txt`에 기입하도록 안내. timeline.txt로 id↔시각 매핑 제공. 승인(또는 reject 작성 완료) 받기 전까지 finalize 진행 금지.

---

## Task 10: finalize → 데이터셋 패키징 + 검증

- [ ] **Step 1: finalize 실행**

Run: `cd voicebox-clone-workspace && ../.venv/bin/python build_gptsovits_dataset.py finalize --threshold <확정T> 2>&1 | tail -20`
Expected: `[finalize] N개 라벨 / M분 → .../dataset/jhm.list`

- [ ] **Step 2: 산출물 검증**

Run:
```bash
cd voicebox-clone-workspace
ls jhm_work/gptsovits/dataset/ jhm_work/gptsovits/dataset/wavs/ | head
head -3 jhm_work/gptsovits/dataset/jhm.list
../.venv/bin/python -c "
import soundfile as sf, glob
fs=sorted(glob.glob('jhm_work/gptsovits/dataset/wavs/*.wav'))
d,sr=sf.read(fs[0]); print('첫 wav', sr, 'Hz', round(len(d)/sr,2),'s', '개수', len(fs))
for ln in open('jhm_work/gptsovits/dataset/jhm.list').read().splitlines()[:3]: print(ln)
"
```
Expected: wav가 32000Hz, `.list` 각 줄이 `wavs/jhm_XXXX.wav|jhm|ko|<한국어 전사>` 형식, 빈 전사 없음

- [ ] **Step 3: 최종 사용자 보고**

데이터셋 위치·세그먼트 수·총 분량·Windows 사용법(README/relocate_list.py) 안내. `dataset/` 폴더를 Windows로 옮겨 GPT-SoVITS 학습 진행 가능함을 보고.

> 주: `jhm_work/` 이하 데이터·산출물은 `.gitignore` 대상(코드만 커밋). dataset 자체는 git에 올리지 않는다.

---

## Self-Review 결과 (계획 작성자 점검)

- **Spec 커버리지:** §3 구조→Task1-6, §4 단계(probe/rebuild-anchor/extract/montage/finalize)→Task4-10, §5 결정(anchor 클린4·16k분석/32k export·faster-whisper ko·.list 상대+relocate·반자동·device자동)→각 Task 반영, §7 테스트→Task1·2·4·5·6·7. 누락 없음.
- **플레이스홀더:** 임계 `<확정T>`는 Task8 실측 후 사용자 확정값(의도된 런타임 값). 코드 스텝엔 placeholder 없음.
- **타입/시그니처 일관성:** `remap_segments`/`slice_segments`/`save_segments`/`build_montage`/`timeline_lines`/`build_list_lines`/`relocate_lines`/`transcribe_segments` 시그니처가 CLI 호출부와 일치. manifest 엔트리 키(id/source/start_s/end_s/dur/sim)가 save_segments↔montage↔finalize에서 일관.
