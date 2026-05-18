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
