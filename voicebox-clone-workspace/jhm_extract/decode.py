from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

SR = 16000


def cache_key(src: Path) -> str:
    st = src.stat()
    raw = f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


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
