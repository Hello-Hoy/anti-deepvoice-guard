from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.eval_quality import compute_jamo_cer, summarize_metric_verdicts


def test_compute_jamo_cer_returns_zero_for_identical_text() -> None:
    result = compute_jamo_cer("안녕하세요", "안녕하세요")

    assert result["cer"] == 0.0
    assert result["reference_jamo_length"] > 0


def test_compute_jamo_cer_detects_partial_substitution() -> None:
    result = compute_jamo_cer("안녕하세요", "안녕하새요")

    assert 0.0 < result["cer"] < 1.0


def test_summarize_metric_verdicts_uses_project_thresholds() -> None:
    summary = summarize_metric_verdicts(
        {
            "wer": {"wer": 0.08},
            "utmosv2": 3.9,
            "dnsmos": {"ovrl": 3.2},
            "spk_sim": 0.8,
            "aasist": 0.71,
        }
    )

    assert summary["overall_pass"] is True
    assert summary["checks"]["wer"]["passed"] is True
    assert summary["checks"]["utmosv2"]["threshold"] == "> 3.5"
    assert summary["checks"]["aasist"]["direction"] == "informational"
