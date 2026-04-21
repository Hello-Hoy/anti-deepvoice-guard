from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.engines.xtts import (
    _find_xtts_pretrained_root,
    _resolve_xtts_pretrained_bundle,
)


def _touch_bundle(root: Path, *, missing: set[str] | None = None) -> None:
    missing = missing or set()
    root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "model.pth",
        "config.json",
        "vocab.json",
        "speakers_xtts.pth",
        "dvae.pth",
        "mel_stats.pth",
    ):
        if filename in missing:
            continue
        (root / filename).write_text(filename, encoding="utf-8")


def test_find_xtts_pretrained_root_prefers_first_matching_directory(tmp_path: Path) -> None:
    app_support = tmp_path / "app-support" / "tts_models--multilingual--multi-dataset--xtts_v2"
    local_share = tmp_path / "local-share" / "tts_models--multilingual--multi-dataset--xtts_v2"
    custom_root = tmp_path / "custom-root" / "tts_models--multilingual--multi-dataset--xtts_v2"
    _touch_bundle(app_support)
    _touch_bundle(local_share)
    _touch_bundle(custom_root)

    resolved = _find_xtts_pretrained_root([app_support.parent, local_share.parent, custom_root.parent])

    assert resolved == app_support


def test_resolve_xtts_pretrained_bundle_reports_missing_file_with_download_hint(tmp_path: Path) -> None:
    broken_root = tmp_path / "app-support" / "tts_models--multilingual--multi-dataset--xtts_v2"
    _touch_bundle(broken_root, missing={"dvae.pth"})

    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_xtts_pretrained_bundle([broken_root.parent])

    message = str(exc_info.value)
    assert "dvae.pth" in message
    assert "huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth" in message
