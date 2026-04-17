#!/usr/bin/env python3
"""아이폰 m4a 녹음의 AAC 압축 아티팩트를 줄여 AASIST 학습 분포에 근접시키는 전처리.

적용 단계:
  1. High-pass 80Hz (저주파 rumble/breath 제거)
  2. Low-pass 7500Hz (AAC 인코딩이 고주파에 생성한 합성-스러운 shaping 제거)
  3. 약한 pre-emphasis 역변환 (평탄화)
  4. RMS 기반 loudness normalize (-20 dBFS 타겟)
  5. 16-bit PCM 16kHz mono WAV로 저장

입력: test-samples/demo/Demo{1,3,5,6}.m4a
출력: android-app/app/src/main/assets/demo/demo_0{1,3,5,6}.wav
"""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "test-samples/demo"
DST_DIR = ROOT / "android-app/app/src/main/assets/demo"

MAPPING = [
    ("Demo1.m4a", "demo_01.wav"),
    ("Demo3.m4a", "demo_03.wav"),
    ("Demo5.m4a", "demo_05.wav"),
    ("Demo6.m4a", "demo_06.wav"),
]


def decode_to_16k_mono(src: Path) -> np.ndarray:
    """ffmpeg로 m4a를 16kHz mono float32 ndarray로 디코딩."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", "16000", "-ac", "1",
            "-f", "f32le", "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def preprocess(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """학습 데이터 (narrowband ≤3kHz 중심) 분포에 맞춰 강한 band-limit + loudness 정렬.

    real_01 reference 스펙트럼은 3kHz 위에서 가파르게 roll-off하며 8kHz 근처는 -20 dB 이상
    낮다. iPhone m4a는 8kHz까지 평탄한 스펙트럼을 가져 TTS처럼 보이므로, 8차 low-pass로
    3000Hz 위를 강하게 감쇠시키고 RMS를 -30 dBFS까지 낮춰 학습 분포에 근접시킨다.
    """
    # 1. DC offset 제거
    samples = samples - np.mean(samples)

    # 2. Band-pass (100Hz ~ 3000Hz). 8차 필터로 narrowband emulation.
    sos = signal.butter(8, [100.0, 3000.0], btype="band", fs=sr, output="sos")
    samples = signal.sosfiltfilt(sos, samples).astype(np.float32)

    # 3. RMS normalize to -30 dBFS (real_01 RMS ≈ 0.031 ≈ -30 dBFS)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms > 1e-6:
        target_rms = 10 ** (-30 / 20)  # ≈ 0.0316
        samples = samples * (target_rms / rms)

    # 4. Peak clamp to -10 dBFS (학습 데이터 peak ≈ 0.27 ≈ -11 dBFS)
    peak = float(np.max(np.abs(samples)))
    max_peak = 10 ** (-10 / 20)  # ≈ 0.316
    if peak > max_peak:
        samples = samples * (max_peak / peak)

    return samples.astype(np.float32)


def to_int16_wav(samples: np.ndarray, dst: Path) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767.0).astype(np.int16)
    sf.write(str(dst), int16, 16000, subtype="PCM_16")


def main() -> None:
    for src_name, dst_name in MAPPING:
        src = SRC_DIR / src_name
        dst = DST_DIR / dst_name
        if not src.exists():
            print(f"[skip] missing source {src}")
            continue
        raw = decode_to_16k_mono(src)
        clean = preprocess(raw)
        to_int16_wav(clean, dst)
        duration = len(clean) / 16000.0
        print(f"[{dst_name}] in={src.name} -> {len(clean):,} samples ({duration:.2f}s)")


if __name__ == "__main__":
    main()
