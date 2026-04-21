from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.detector_stress import (
    build_paired_index,
    slice_analysis,
    threshold_sweep,
)


def test_build_paired_index_matches_real_and_fake_by_stem() -> None:
    pairs = build_paired_index(
        real_wavs=[
            Path("/tmp/clip_0001.wav"),
            Path("/tmp/clip_0002.wav"),
        ],
        fake_wavs=[
            Path("/tmp/clip_0001_xtts_seed01.wav"),
            Path("/tmp/clip_0002_cosyvoice2_seed05.wav"),
        ],
    )

    assert set(pairs) == {"clip_0001", "clip_0002"}
    assert pairs["clip_0001"]["real"].name == "clip_0001.wav"
    assert len(pairs["clip_0002"]["fake"]) == 1


def test_threshold_sweep_reports_perfect_separation_metrics() -> None:
    metrics = threshold_sweep(real_scores=[0.02, 0.05, 0.1], fake_scores=[0.85, 0.9, 0.97])

    assert metrics["eer"] == 0.0
    assert metrics["fpr_at_tpr_95"] == 0.0
    assert metrics["fpr_at_tpr_99"] == 0.0
    assert 0.1 < metrics["recommended_threshold"] < 0.85


def test_slice_analysis_returns_fp_and_fn_breakdown() -> None:
    result = slice_analysis(
        scores={
            "clean": {
                "real": [
                    {"id": "real_ok", "score": 0.2},
                    {"id": "real_fp", "score": 0.8},
                ],
                "fake": [
                    {"id": "fake_ok", "score": 0.9},
                    {"id": "fake_fn", "score": 0.3},
                ],
            }
        },
        threshold=0.7,
    )

    clean = result["conditions"]["clean"]
    assert clean["counts"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert clean["false_positives"] == ["real_fp"]
    assert clean["false_negatives"] == ["fake_fn"]
