from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import sys
import wave

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.engines import load_engine


def _write_test_wav(path: Path, sample_rate: int = 24000, duration_sec: float = 0.1) -> None:
    frame_count = max(1, int(sample_rate * duration_sec))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack("<" + ("h" * frame_count), *([0] * frame_count)))


def _write_manifest(path: Path, audio_dir: Path) -> Path:
    samples = [
        {
            "id": "clip_0001",
            "wav_path_24k": str((audio_dir / "clip_0001.wav").resolve()),
            "wav_path_16k": str((audio_dir / "clip_0001.wav").resolve()),
            "text": "안녕하세요 GPT 소빗츠 테스트입니다",
            "duration_sec": 0.1,
            "verified": True,
        },
        {
            "id": "clip_0002",
            "wav_path_24k": str((audio_dir / "clip_0002.wav").resolve()),
            "wav_path_16k": str((audio_dir / "clip_0002.wav").resolve()),
            "text": "this is the eval sample",
            "duration_sec": 0.1,
            "verified": True,
        },
    ]
    manifest = {
        "speaker": "tester",
        "samples": samples,
        "split": {"train": ["clip_0001"], "eval": ["clip_0002"]},
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_health_check_returns_install_hint_when_repo_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing-gpt-sovits"
    missing_python = tmp_path / "missing-venv" / "bin" / "python"
    monkeypatch.setenv("GPT_SOVITS_DIR", str(missing_dir))
    monkeypatch.setenv("GPT_SOVITS_VENV", str(missing_python))

    engine = load_engine("gpt_sovits")
    ok, message = engine.health_check()

    assert ok is False
    assert "git clone" in message
    assert "GPT-SoVITS" in message
    assert ".venv-gpt-sovits" in message


def test_load_engine_gpt_sovits_imports_expected_engine() -> None:
    engine = load_engine("gpt_sovits")

    assert engine.name == "gpt_sovits"
    assert engine.train_sample_rate == 32000
    assert engine.phase == "phase_b"
    assert "ko" in engine.supports_language


def test_prepare_data_creates_expected_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"
    _write_test_wav(audio_dir / "clip_0001.wav")
    _write_test_wav(audio_dir / "clip_0002.wav")
    manifest_path = _write_manifest(tmp_path / "manifest.json", audio_dir)

    engine = load_engine("gpt_sovits")
    module = sys.modules[engine.__class__.__module__]

    def _fake_convert_audio(src: Path, dst: Path, sr: int, channels: int = 1) -> bool:
        assert sr == 32000
        assert channels == 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True

    monkeypatch.setattr(module, "convert_audio", _fake_convert_audio)

    out_dir = tmp_path / "prepared"
    result = engine.prepare_data(manifest_path, out_dir)

    assert result == out_dir
    assert (out_dir / "raw" / "tester" / "clip_0001.wav").exists()
    assert (out_dir / "raw" / "tester" / "clip_0002.wav").exists()
    assert (out_dir / "lists" / "train.list").read_text(encoding="utf-8").strip() == (
        "clip_0001.wav|tester|ko|안녕하세요 GPT 소빗츠 테스트입니다"
    )
    assert (out_dir / "lists" / "val.list").read_text(encoding="utf-8").strip() == (
        "clip_0002.wav|tester|ko|this is the eval sample"
    )
