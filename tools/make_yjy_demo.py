#!/usr/bin/env python3
"""양정윤 보이스클론(yjy_money_ver.wav)을 데모 asset(demo_11.wav)으로 변환.

GPT-SoVITS 합성 클론은 demo_08/09/10과 동일하게 **narrowband 전처리 없이 raw 16k**로
저장한다(band-limit는 real 음성 분포용이라 클론에 적용하면 탐지 분포가 어긋남). 음량은
demo_08/09/10과 동일하게 peak 0.97로 정규화 — 클론(fake) WAV 정규화는 탐지를 흔들지
않으며(오히려 발화 구간 fakeScore 0.92~0.98로 안정적 DANGER), '정규화 금지'는 real 데모
전용 규칙이다. 재생 음량은 앱의 LoudnessEnhancer가 적응형으로 추가 보정한다.

입력: GPT-SoVITS/outputs/jeongyoon/yjy_money_ver.wav (32kHz mono float)
출력: android-app/app/src/main/assets/demo/demo_11.wav (16kHz mono 16-bit PCM, peak 0.97)
"""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "GPT-SoVITS/outputs/jeongyoon/yjy_money_ver.wav"
DST = ROOT / "android-app/app/src/main/assets/demo/demo_11.wav"
TARGET_SR = 16000
TARGET_PEAK = 0.97


def main() -> None:
    data, sr = sf.read(str(SRC))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        data = signal.resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)

    # DC offset 제거 (narrowband 필터는 적용하지 않음 — raw 16k 원칙)
    data = data - np.mean(data)

    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = (data * (TARGET_PEAK / peak)).astype(np.float32)

    int16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    DST.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(DST), int16, TARGET_SR, subtype="PCM_16")

    dur = len(data) / TARGET_SR
    rms = float(np.sqrt(np.mean(data**2)))
    dbfs = 20 * np.log10(rms + 1e-9)
    print(f"[demo_11.wav] {SRC.name} {sr}Hz -> {TARGET_SR}Hz | "
          f"{len(data):,} samples ({dur:.2f}s) peak={np.max(np.abs(data)):.3f} "
          f"rms={dbfs:.1f}dBFS")


if __name__ == "__main__":
    main()
