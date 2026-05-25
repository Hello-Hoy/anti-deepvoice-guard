#!/usr/bin/env python3
"""GPT-SoVITS 합성 wav를 안드로이드 데모 자산(16kHz mono PCM16)으로 변환.

중요: GPT-SoVITS 데모는 **band-limit/narrowband 에뮬레이션 금지**.
  preprocess_demo_wav.py(iPhone m4a용 100~3000Hz band-pass)를 적용하면 AASIST 점수가
  깨진다. GPT-SoVITS 음성은 raw full-band 16k 그대로 둬야 ~0.70(WARNING)으로 탐지된다.
  (참고: 이후 tools/boost_demo_volume.py로 peak 0.97 부스트 → 점수 안정.)

사용:
    .venv/bin/python tools/make_gptsovits_demo_wav.py <src.wav> <android-app/.../demo/demo_NN.wav>
"""
import sys

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    # librosa.load: 고품질 리샘플(soxr) + mono. band-limit 없음(raw full-band).
    audio, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    sf.write(dst, audio.astype(np.float32), TARGET_SR, subtype="PCM_16")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    print(f"saved: {dst}  ({len(audio)/TARGET_SR:.2f}s @ {TARGET_SR}Hz mono PCM16, peak={peak:.3f})")


if __name__ == "__main__":
    main()
