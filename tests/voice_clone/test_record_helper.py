from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "model-training" / "voice-clone" / "scripts" / "record_helper.py"


def load_record_helper_module():
    spec = spec_from_file_location("record_helper", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_script_lines_ignores_blank_lines() -> None:
    helper = load_record_helper_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "script.txt"
        script_path.write_text("첫 문장입니다.\n\n  \n둘째 문장입니다.\n", encoding="utf-8")

        lines = helper.load_script_lines(script_path)

    assert lines == ["첫 문장입니다.", "둘째 문장입니다."]


def test_compute_peak_dbfs_matches_full_scale_reference() -> None:
    helper = load_record_helper_module()

    audio = np.array([0.0, 0.5, -1.0, 0.25], dtype=np.float32)

    peak_dbfs = helper.compute_peak_dbfs(audio)

    assert peak_dbfs == 0.0


def test_estimate_snr_db_detects_clean_signal() -> None:
    helper = load_record_helper_module()

    sample_rate = 48_000
    noise = np.random.default_rng(7).normal(0.0, 0.001, sample_rate).astype(np.float32)
    tone = (0.12 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_rate) / sample_rate)).astype(
        np.float32
    )
    audio = np.concatenate([noise, tone])

    snr_db = helper.estimate_snr_db(audio, sample_rate)

    assert snr_db > 20.0


def test_assess_recording_and_resume_start_index_cover_edge_conditions() -> None:
    helper = load_record_helper_module()

    sample_rate = 48_000
    noisy_audio = np.random.default_rng(11).normal(0.0, 0.04, sample_rate // 2).astype(np.float32)
    assessment = helper.assess_recording(noisy_audio, sample_rate)

    assert "duration_out_of_range" in assessment.warning_codes
    assert "noise_detected" in assessment.warning_codes

    log_entries = [
        {"index": 1, "rec_status": "recorded"},
        {"index": 2, "rec_status": "skipped"},
        {"index": 4, "rec_status": "recorded"},
    ]

    assert helper.find_resume_start_index(log_entries, total_lines=6) == 2
