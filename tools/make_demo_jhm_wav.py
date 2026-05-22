#!/usr/bin/env python3
"""GPT-SoVITS 전현무 클론 데모(32k) → 16k 협대역 전처리 → assets/demo/demo_08,09.wav.

기존 tools/preprocess_demo_wav.py의 전처리(100~3000Hz band-pass + RMS -30dBFS)를 재사용해
전화통화 대역을 에뮬레이션한다. 변환 후 aasist.onnx로 fakeScore를 출력해 데모 적합성을 검증한다.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from preprocess_demo_wav import decode_to_16k_mono, preprocess, to_int16_wav  # noqa: E402

SRC = ROOT / "GPT-SoVITS/outputs/jhm/demo"
DST = ROOT / "android-app/app/src/main/assets/demo"
MAPPING = [("01_call.wav", "demo_08.wav"), ("02_menu.wav", "demo_09.wav")]


def aasist_fake_score(wav_path: Path) -> float:
    import onnxruntime as ort
    import soundfile as sf
    model = ROOT / "android-app/app/src/main/assets/aasist.onnx"
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    audio, _ = sf.read(str(wav_path), dtype="float32")
    target = 64600
    if len(audio) >= target:
        audio = audio[:target]
    else:
        audio = np.pad(audio, (0, target - len(audio)))
    inp = sess.get_inputs()[0]
    x = audio.reshape(1, -1).astype(np.float32)
    out = sess.run(None, {inp.name: x})[0]
    e = np.exp(out - out.max())
    prob = (e / e.sum()).ravel()
    return float(prob[0])  # index 0 = spoof/fake


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in MAPPING:
        src = SRC / src_name
        dst = DST / dst_name
        raw = decode_to_16k_mono(src)
        clean = preprocess(raw)
        to_int16_wav(clean, dst)
        score = aasist_fake_score(dst)
        flag = "OK(fake)" if score >= 0.7 else "LOW — 데모 부적합 가능"
        print(f"[{dst_name}] {len(clean)/16000:.1f}s  fakeScore={score:.3f}  {flag}")


if __name__ == "__main__":
    main()
