#!/usr/bin/env python3
"""데모 wav를 피크 0.97로 라우드니스 정규화(재생 음량↑). 클리핑 없음.

진폭 변화가 AASIST 딥보이스 점수에 주는 영향을 확인하기 위해 적용 전 before/after
sliding-window fakeScore를 출력한다.
  dry-run(점수만):  python tools/boost_demo_volume.py
  실제 적용:        python tools/boost_demo_volume.py --apply
"""
import sys
import glob
import os
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "android-app/app/src/main/assets/demo"
TARGET_PEAK = 0.97
# 부스트 대상: GPT-SoVITS 데모만(점수 안정). 나머지는 진폭↑ 시 탐지가 깨져 제외.
TARGETS = ["demo_08.wav", "demo_09.wav"]


def load(f):
    a, sr = sf.read(f, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    return a, sr


def boost(a):
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    if peak < 1e-6:
        return a
    return (a * (TARGET_PEAK / peak)).astype(np.float32)


def sliding_fake(a, sess, inp):
    t = 64600
    if len(a) <= t:
        wins = [np.pad(a, (0, t - len(a)))]
    else:
        h = t // 2
        wins, s = [], 0
        while s + t <= len(a):
            wins.append(a[s:s + t]); s += h
        tail = np.zeros(t, np.float32); tail[:len(a) - s] = a[s:]; wins.append(tail)
    sc = []
    for w in wins:
        o = sess.run(None, {inp: w.reshape(1, -1).astype(np.float32)})[0]
        e = np.exp(o - o.max()); sc.append(float((e / e.sum()).ravel()[0]))
    return float(np.mean(sc))


def main():
    apply = "--apply" in sys.argv
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ROOT / "android-app/app/src/main/assets/aasist.onnx"),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    print(f"{'APPLY' if apply else 'DRY-RUN'}  target_peak={TARGET_PEAK}  targets={TARGETS}")
    for f in [str(DEMO / t) for t in TARGETS]:
        a, sr = load(f)
        b = boost(a)
        fa, fb = sliding_fake(a, sess, inp), sliding_fake(b, sess, inp)
        gain = (TARGET_PEAK / (np.max(np.abs(a)) + 1e-9))
        print(f"  {os.path.basename(f):16s} gain×{gain:4.1f}  fake {fa:.3f}→{fb:.3f}")
        if apply:
            sf.write(f, np.clip(b, -1, 1).astype(np.float32), sr, subtype="PCM_16")
    print("적용 완료" if apply else "(미적용 — --apply로 저장)")


if __name__ == "__main__":
    main()
