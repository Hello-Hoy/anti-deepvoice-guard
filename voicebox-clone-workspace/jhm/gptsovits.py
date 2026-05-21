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
