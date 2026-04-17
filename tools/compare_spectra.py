#!/usr/bin/env python3
"""known-good real_01.wav (학습 분포) vs user recording 스펙트럼 특성 비교."""
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
KNOWN_GOOD = ROOT / "test-samples/real/real_01.wav"
USER_DEMO = ROOT / "android-app/app/src/main/assets/demo/demo_01.wav"


def load(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def summary(name: str, samples: np.ndarray, sr: int) -> None:
    spec = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sr)

    # 주파수 대역별 평균 에너지 (dB)
    bands = [(0, 500), (500, 1500), (1500, 3000), (3000, 5000), (5000, 7000), (7000, 8000)]
    print(f"\n=== {name} ({sr}Hz, {len(samples)/sr:.2f}s) ===")
    print(f"RMS: {np.sqrt(np.mean(samples**2)):.4f}, Peak: {np.max(np.abs(samples)):.4f}")
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        avg_energy = np.mean(spec[mask])
        db = 20 * np.log10(avg_energy + 1e-10)
        bar = "#" * max(0, int(db / 2))
        print(f"  {lo:5d}-{hi:5d}Hz: {db:7.2f} dB  {bar}")


def main() -> None:
    for name, path in [("KNOWN-GOOD real_01", KNOWN_GOOD), ("USER demo_01", USER_DEMO)]:
        s, sr = load(path)
        summary(name, s, sr)


if __name__ == "__main__":
    main()
