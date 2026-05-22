#!/usr/bin/env python3
"""GPT-SoVITS 전현무 클론 데모(32k) → raw 16k mono → assets/demo/demo_08,09.wav.

narrowband 전처리(100~3000Hz)는 AASIST가 보는 가짜 단서를 깎아 fakeScore를 떨어뜨리므로
적용하지 않는다(raw 16k 유지). 앱은 raw 오디오를 전체 클립 슬라이딩윈도우로 집계해 점수내므로
검증도 동일 방식(64600 윈도우, 50% overlap, 평균)으로 한다.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from preprocess_demo_wav import decode_to_16k_mono, to_int16_wav  # noqa: E402

SRC = ROOT / "GPT-SoVITS/outputs/jhm/demo"
DST = ROOT / "android-app/app/src/main/assets/demo"
MAPPING = [("01_call.wav", "demo_08.wav"), ("02_menu.wav", "demo_09.wav")]


def sliding_fake_score(audio: np.ndarray, sess, inp: str) -> float:
    target = 64600
    if len(audio) <= target:
        wins = [np.pad(audio, (0, target - len(audio)))]
    else:
        hop = target // 2
        wins = []
        s = 0
        while s + target <= len(audio):
            wins.append(audio[s:s + target]); s += hop
        tail = np.zeros(target, np.float32); tail[: len(audio) - s] = audio[s:]; wins.append(tail)
    scores = []
    for w in wins:
        out = sess.run(None, {inp: w.reshape(1, -1).astype(np.float32)})[0]
        e = np.exp(out - out.max()); scores.append(float((e / e.sum()).ravel()[0]))
    return sum(scores) / len(scores)


def main() -> None:
    import onnxruntime as ort
    DST.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(ROOT / "android-app/app/src/main/assets/aasist.onnx"),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    for src_name, dst_name in MAPPING:
        raw = decode_to_16k_mono(SRC / src_name)  # 16k mono float32, band-pass 없음
        to_int16_wav(raw, DST / dst_name)
        score = sliding_fake_score(raw, sess, inp)
        flag = "OK(fake)" if score >= 0.7 else "LOW"
        print(f"[{dst_name}] {len(raw) / 16000:.1f}s  sliding_fakeScore={score:.3f}  {flag}")


if __name__ == "__main__":
    main()
