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


def cluster_labels(embs: np.ndarray, distance_threshold: float = 0.5) -> np.ndarray:
    """ECAPA 임베딩 → agglomerative(cosine, average) 클러스터 레이블."""
    if len(embs) == 0:
        return np.zeros(0, dtype=int)
    if len(embs) == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(embs)


def dominant_cluster_indices(labels: np.ndarray, kept: list[dict]) -> list[int]:
    """총 발화시간(dur 합)이 가장 큰 클러스터의 세그먼트 인덱스 리스트(오름차순)."""
    if len(labels) == 0:
        return []
    durs: dict[int, float] = {}
    for i, lab in enumerate(labels):
        durs[int(lab)] = durs.get(int(lab), 0.0) + kept[i]["dur"]
    best = max(durs, key=durs.get)
    return [i for i in range(len(labels)) if int(labels[i]) == best]
