from __future__ import annotations

from pathlib import Path


_SOURCE_DIR = Path(__file__).resolve().parents[2] / "model-training" / "voice-clone"

if str(_SOURCE_DIR) not in __path__:
    __path__.append(str(_SOURCE_DIR))
