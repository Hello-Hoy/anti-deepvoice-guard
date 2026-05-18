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
