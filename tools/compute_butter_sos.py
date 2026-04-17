#!/usr/bin/env python3
"""Butterworth 8차 band-pass (100-3000Hz @ 16kHz) SOS 계수 계산 및 Kotlin 상수 출력.

출력은 `android-app/app/src/main/java/com/deepvoiceguard/app/audio/NarrowbandPreprocessor.kt`의
companion object에 하드코딩할 DoubleArray 형태.

scipy.signal.butter(output='sos')는 (n_sections, 6) 배열을 반환하며, 각 row는
[b0, b1, b2, a0, a1, a2] (a0은 1.0으로 정규화).
"""
import sys

import numpy as np
from scipy import signal

SAMPLE_RATE = 16_000
LOW_HZ = 100.0
HIGH_HZ = 3_000.0
ORDER = 8  # 4 biquad sections


def main() -> None:
    sos = signal.butter(ORDER, [LOW_HZ, HIGH_HZ], btype="band", fs=SAMPLE_RATE, output="sos")
    n_sections = sos.shape[0]
    print(f"// scipy.signal.butter(N={ORDER}, Wn=[{LOW_HZ}, {HIGH_HZ}]Hz, btype='band', fs={SAMPLE_RATE}Hz, output='sos')")
    print(f"// {n_sections} second-order sections, coefficients [b0,b1,b2,a0,a1,a2] per section.")
    print("// a0은 항상 1.0으로 정규화됨.")
    print()
    print("private val SOS_COEFFS: Array<DoubleArray> = arrayOf(")
    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        print(
            f"    doubleArrayOf({b0: .17e}, {b1: .17e}, {b2: .17e}, "
            f"{a0: .17e}, {a1: .17e}, {a2: .17e}),  // section {i}"
        )
    print(")")

    # Sanity: 1kHz sine passband, 5kHz sine stopband — amplitude 체크
    print()
    print("// Sanity check (Python-side reference, should match Kotlin implementation):")
    for freq in [100.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 7000.0]:
        t = np.arange(32_000) / SAMPLE_RATE
        sine = np.sin(2 * np.pi * freq * t).astype(np.float64)
        filtered = signal.sosfiltfilt(sos, sine)
        # 필터 transient 제외한 중간 구간 RMS
        mid = filtered[8_000:24_000]
        rms_in = np.sqrt(np.mean(sine[8_000:24_000] ** 2))
        rms_out = np.sqrt(np.mean(mid ** 2))
        gain_db = 20 * np.log10(max(rms_out / rms_in, 1e-10))
        print(f"//   {freq:6.0f} Hz sine -> {gain_db:+7.2f} dB")


if __name__ == "__main__":
    main()
