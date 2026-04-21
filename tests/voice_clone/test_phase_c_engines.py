from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.engines import load_engine


@pytest.mark.parametrize(
    ("engine_name", "expected_rate", "expected_phase", "expected_hw"),
    [
        ("rvc_postvc", 40000, "phase_c", "cuda"),
        ("qwen3_tts", 24000, "phase_c", "cuda"),
        ("indextts2", 24000, "phase_c", "cuda"),
        ("styletts2", 24000, "phase_c", "cuda"),
    ],
)
def test_phase_c_engines_expose_expected_metadata(
    engine_name: str,
    expected_rate: int,
    expected_phase: str,
    expected_hw: str,
) -> None:
    engine = load_engine(engine_name)

    assert engine.name == engine_name
    assert engine.train_sample_rate == expected_rate
    assert engine.phase == expected_phase
    assert engine.hardware_pref == expected_hw
    assert "ko" in engine.supports_language


def test_rvc_postvc_health_check_reports_install_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_repo = tmp_path / "missing-rvc"
    missing_python = tmp_path / "missing-rvc-venv" / "bin" / "python"
    monkeypatch.setenv("RVC_DIR", str(missing_repo))
    monkeypatch.setenv("RVC_VENV", str(missing_python))
    monkeypatch.setenv("RVC_UPSTREAM_ENGINE", "edge_tts")

    engine = load_engine("rvc_postvc")
    ok, message = engine.health_check()

    assert ok is False
    assert "RVC_DIR" in message
    assert "RVC_VENV" in message
    assert "edge-tts" in message


def test_qwen3_tts_health_check_reports_install_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_repo = tmp_path / "missing-qwen3"
    missing_python = tmp_path / "missing-qwen3-venv" / "bin" / "python"
    missing_model = tmp_path / "missing-qwen3-model"
    monkeypatch.setenv("QWEN3_TTS_DIR", str(missing_repo))
    monkeypatch.setenv("QWEN3_TTS_VENV", str(missing_python))
    monkeypatch.setenv("QWEN3_TTS_MODEL", str(missing_model))

    engine = load_engine("qwen3_tts")
    ok, message = engine.health_check()

    assert ok is False
    assert "QWEN3_TTS_DIR" in message
    assert "QWEN3_TTS_VENV" in message
    assert "QWEN3_TTS_MODEL" in message


def test_indextts2_health_check_reports_install_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_repo = tmp_path / "missing-indextts2"
    missing_python = tmp_path / "missing-indextts2-venv" / "bin" / "python"
    missing_model = tmp_path / "missing-indextts2-model"
    monkeypatch.setenv("INDEXTTS2_DIR", str(missing_repo))
    monkeypatch.setenv("INDEXTTS2_VENV", str(missing_python))
    monkeypatch.setenv("INDEXTTS2_MODEL", str(missing_model))

    engine = load_engine("indextts2")
    ok, message = engine.health_check()

    assert ok is False
    assert "INDEXTTS2_DIR" in message
    assert "INDEXTTS2_VENV" in message
    assert "INDEXTTS2_MODEL" in message


def test_styletts2_health_check_reports_install_hints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_repo = tmp_path / "missing-styletts2"
    missing_python = tmp_path / "missing-styletts2-venv" / "bin" / "python"
    missing_config = tmp_path / "missing-styletts2-config.yml"
    monkeypatch.setenv("STYLETTS2_DIR", str(missing_repo))
    monkeypatch.setenv("STYLETTS2_VENV", str(missing_python))
    monkeypatch.setenv("STYLETTS2_CONFIG", str(missing_config))
    monkeypatch.delenv("STYLETTS2_PIP_MODE", raising=False)

    engine = load_engine("styletts2")
    ok, message = engine.health_check()

    assert ok is False
    assert "STYLETTS2_DIR" in message
    assert "STYLETTS2_VENV" in message
    assert "STYLETTS2_CONFIG" in message
    assert "한국어" in message
