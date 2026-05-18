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
