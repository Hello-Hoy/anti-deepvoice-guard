from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

SR = 16000


def cache_key(src: Path) -> str:
    st = src.stat()
    raw = f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
