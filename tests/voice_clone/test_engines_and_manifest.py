from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.engines import list_engines, load_engine
from model_training.voice_clone.engines._manifest import (
    get_sample_by_id,
    load_manifest,
    split_samples,
    validate_manifest,
)


VOICE_CLONE_DIR = REPO_ROOT / "model-training" / "voice-clone"


def test_list_engines_returns_supported_engine_names() -> None:
    engines = list_engines()

    assert "xtts" in engines
    assert "cosyvoice2" in engines


@pytest.mark.parametrize("engine_name", ["xtts", "cosyvoice2"])
def test_load_engine_known_names_returns_expected_engine(engine_name: str) -> None:
    engine = load_engine(engine_name)

    assert engine.name == engine_name
    assert engine.train_sample_rate == 24000
    assert "ko" in engine.supports_language


def test_load_engine_unknown_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown voice clone engine"):
        load_engine("missing-engine")


def test_manifest_example_is_valid() -> None:
    example_path = VOICE_CLONE_DIR / "manifest_example.json"
    manifest = load_manifest(example_path)

    assert validate_manifest(manifest) == []
    assert get_sample_by_id(manifest, "clip_0001") is not None
    assert [sample["id"] for sample in split_samples(manifest, "train")] == ["clip_0001"]
    assert [sample["id"] for sample in split_samples(manifest, "eval")] == ["clip_0002"]


def test_validate_manifest_reports_missing_required_fields() -> None:
    invalid_manifest = {
        "speaker": "tester",
        "samples": [
            {
                "id": "bad-id!",
                "wav_path_24k": "wavs_24k/bad.wav",
                "wav_path_16k": "wavs_16k/bad.wav",
                "text": "",
                "duration_sec": 0,
                "verified": "yes",
            }
        ],
        "split": {"train": ["missing"], "eval": []},
    }

    errors = validate_manifest(invalid_manifest)

    assert errors
    joined = json.dumps(errors, ensure_ascii=False)
    assert "bad-id!" in joined
    assert "duration_sec" in joined
    assert "verified" in joined
