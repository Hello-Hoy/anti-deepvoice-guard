#!/usr/bin/env python3
"""Demo 6개 샘플을 AASIST로 스코어링하여 fake/real 확률을 출력한다.

앱 재빌드 없이 전처리가 효과 있는지 검증.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model-training/weights/finetuned/aasist.onnx"
DEMO_DIR = ROOT / "android-app/app/src/main/assets/demo"
NB_SAMP = 64600


def load_wav(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path))
    if sr != 16000:
        raise RuntimeError(f"{path}: expected 16kHz, got {sr}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def pad_or_crop(samples: np.ndarray, target: int = NB_SAMP) -> np.ndarray:
    n = len(samples)
    if n == target:
        return samples
    if n > target:
        return samples[:target]
    out = np.zeros(target, dtype=np.float32)
    written = 0
    while written < target:
        chunk = min(n, target - written)
        out[written:written + chunk] = samples[:chunk]
        written += chunk
    return out


def score_segments(session: ort.InferenceSession, samples: np.ndarray):
    """50% overlap sliding window으로 여러 세그먼트 inference."""
    if len(samples) <= NB_SAMP:
        return [infer_single(session, pad_or_crop(samples))]
    hop = NB_SAMP // 2
    results = []
    start = 0
    while start + NB_SAMP <= len(samples):
        results.append(infer_single(session, samples[start:start + NB_SAMP]))
        start += hop
    if start < len(samples):
        tail = np.zeros(NB_SAMP, dtype=np.float32)
        tail[:len(samples) - start] = samples[start:]
        results.append(infer_single(session, tail))
    return results


def infer_single(session: ort.InferenceSession, segment: np.ndarray):
    inputs = {"audio": segment.reshape(1, -1).astype(np.float32)}
    out = session.run(None, inputs)
    probs = out[0][0]  # [fake, real]
    return {"fake": float(probs[0]), "real": float(probs[1])}


def main() -> None:
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])

    scenarios = [
        (1, "일상 통화", "SAFE"),
        (2, "TTS 일상", "DANGER"),
        (3, "실제 사람 피싱", "WARNING"),
        (4, "TTS 피싱", "CRITICAL"),
        (5, "은행 정상", "SAFE"),
        (6, "보험 사기", "WARNING"),
    ]
    print(f"{'#':<3} {'시나리오':<16} {'예상':<10} {'평균 fake':>10} {'최대 fake':>10}  판정")
    print("-" * 70)
    for idx, title, expected in scenarios:
        wav = DEMO_DIR / f"demo_0{idx}.wav"
        samples = load_wav(wav)
        results = score_segments(session, samples)
        fakes = [r["fake"] for r in results]
        avg_fake = sum(fakes) / len(fakes)
        max_fake = max(fakes)
        verdict = "FAKE" if avg_fake >= 0.5 else "REAL"
        print(
            f"{idx:<3} {title:<14} {expected:<10} {avg_fake:>10.2%} {max_fake:>10.2%}  {verdict}"
        )


if __name__ == "__main__":
    main()
