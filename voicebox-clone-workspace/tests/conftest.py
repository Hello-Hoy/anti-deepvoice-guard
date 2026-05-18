import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

SR = 16000


def _mod_noise(dur_s: float, gap_every: float = 1.5, gap_len: float = 0.25,
               amp: float = 0.2, seed: int = 0) -> np.ndarray:
    """진폭변조 대역잡음 + 주기적 완전무음 gap = '깨끗한 발화' 합성."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    base = rng.standard_normal(n).astype(np.float32)
    k = np.ones(8, np.float32) / 8.0
    base = np.convolve(base, k, mode="same").astype(np.float32)
    env = np.ones(n, np.float32)
    t = 0.0
    while t < dur_s:
        s = int(t * SR)
        e = int((t + gap_len) * SR)
        env[s:e] = 0.0
        t += gap_every
    return (base * env * amp).astype(np.float32)


@pytest.fixture
def clean_speech() -> np.ndarray:
    return _mod_noise(12.0)


@pytest.fixture
def speech_with_music_bed(clean_speech) -> np.ndarray:
    # 톤 진폭 0.007 (RMS≈5e-3): noise_floor(1e-3)의 gap_ratio_max(3)배는 넘되
    # voiced_mult(8)배는 안 넘어야 gap 프레임이 pause로 분류돼 gap-energy 분기가
    # 발동한다. 더 크면 gap이 voiced로 분류돼 no_pauses(다른 판별기)로 잡힘.
    n = clean_speech.shape[0]
    t = np.arange(n) / SR
    tone = (0.007 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    return (clean_speech + tone).astype(np.float32)


@pytest.fixture
def clipped(clean_speech) -> np.ndarray:
    return np.clip(clean_speech * 10.0, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def mostly_silence() -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.standard_normal(int(12.0 * SR)).astype(np.float32) * 1e-4)


@pytest.fixture
def continuous_tone() -> np.ndarray:
    t = np.arange(int(12.0 * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
