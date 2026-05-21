"""전현무 음성 추출 코어 파이프라인.

흐름: m4a → (ffmpeg 디코드) → Demucs htdemucs 보컬분리(BGM 제거)
     → Silero VAD 발화 분할 → resemblyzer 발화단위 임베딩.

anchor(전현무 단독 임베딩 평균) 대비 코사인 유사도로 전현무 발화만 채택.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .silero_vad import VadOptions, get_speech_timestamps

SR = 16000
WORK = Path(__file__).resolve().parent.parent / "jhm_work"
WORK.mkdir(exist_ok=True)
_CACHE = WORK / "cache"
_CACHE.mkdir(exist_ok=True)


def _key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def decode(path: Path, sr: int, mono: bool) -> np.ndarray:
    """ffmpeg로 임의 컨테이너 → float32 PCM. mono=False면 stereo(2,N)."""
    ch = 1 if mono else 2
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    if mono:
        return a.copy()
    return a.reshape(-1, 2).T.copy()


_DEMUCS = None


def _demucs_model():
    global _DEMUCS
    if _DEMUCS is None:
        from demucs.pretrained import get_model

        _DEMUCS = get_model("htdemucs")
        _DEMUCS.train(False)  # 추론 모드 (=.eval())
    return _DEMUCS


DEMUCS_SR = 44100  # htdemucs native sample rate


def demucs_vocal(src: Path, device: str = "mps") -> tuple[np.ndarray, int]:
    """htdemucs 보컬 분리 → (mono float32 보컬, native SR=44100). 캐시 안 함.

    호출측에서 1회 호출 후 원하는 SR로 리샘플(16k 분석 / 32k export)에 재사용.
    """
    import torch
    from demucs.apply import apply_model

    model = _demucs_model()
    stereo = decode(src, model.samplerate, mono=False)  # (2, N) float32
    mix = torch.from_numpy(stereo)
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = (mix - mean) / std
    log(f"  [demucs] {src.name} (device={device}) …")
    with torch.no_grad():
        out = apply_model(model, mix[None], device=device, split=True, overlap=0.25, progress=False)[0]
    out = out * std + mean
    voc_idx = model.sources.index("vocals")
    vocals = out[voc_idx].mean(0).cpu().numpy().astype(np.float32)  # mono @ 44100
    return vocals, int(model.samplerate)


def separate_vocals(src: Path, device: str = "mps") -> np.ndarray:
    """Demucs htdemucs 보컬 분리 → 16k mono 보컬(BGM 제거). 캐시됨."""
    cached = _CACHE / f"vocals_{_key(src)}.wav"
    if cached.exists():
        wav, _ = sf.read(str(cached), dtype="float32")
        return wav

    import librosa

    vocals, sr = demucs_vocal(src, device=device)
    y = librosa.resample(vocals, orig_sr=sr, target_sr=SR).astype(np.float32)
    sf.write(str(cached), y, SR)
    return y


def vad_segments(wav16k: np.ndarray, opts: VadOptions | None = None) -> list[dict]:
    """발화 구간 리스트 [{start,end} 샘플(16k)]."""
    if opts is None:
        opts = VadOptions(
            threshold=0.5,
            min_speech_duration_ms=700,
            max_speech_duration_s=15.0,
            min_silence_duration_ms=300,
            speech_pad_ms=120,
        )
    return get_speech_timestamps(wav16k, opts, sampling_rate=SR)


_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        from resemblyzer import VoiceEncoder

        _ENCODER = VoiceEncoder(device="cpu", verbose=False)
    return _ENCODER


def embed_segments(
    wav16k: np.ndarray, segs: list[dict], min_dur_s: float = 0.8
) -> tuple[np.ndarray, list[dict]]:
    """각 발화 세그먼트 → L2 정규화 d-vector(256). 짧은 건 제외.

    반환: (embeddings[N,256], kept_segments[N])
    """
    from resemblyzer import preprocess_wav

    enc = _encoder()
    embs, kept = [], []
    for s in segs:
        dur = (s["end"] - s["start"]) / SR
        if dur < min_dur_s:
            continue
        clip = wav16k[s["start"]: s["end"]]
        try:
            proc = preprocess_wav(clip, source_sr=SR)
            if len(proc) < SR * 0.4:
                continue
            e = enc.embed_utterance(proc)
        except Exception as ex:  # noqa: BLE001 — 한 세그먼트 실패가 전체를 막지 않도록만
            log(f"    embed skip ({dur:.1f}s): {type(ex).__name__}")
            continue
        e = e / (np.linalg.norm(e) + 1e-9)
        embs.append(e)
        kept.append({**s, "dur": dur})
    if not embs:
        return np.zeros((0, 256), np.float32), []
    return np.stack(embs).astype(np.float32), kept


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return a @ b


def write_concat(
    wav16k: np.ndarray, segs: list[dict], out: Path, gap_s: float = 0.25
) -> float:
    """채택 세그먼트를 짧은 무음으로 이어붙여 wav 저장. 총 길이(초) 반환."""
    if not segs:
        sf.write(str(out), np.zeros(1, np.float32), SR)
        return 0.0
    gap = np.zeros(int(SR * gap_s), np.float32)
    parts = []
    for s in segs:
        parts.append(wav16k[s["start"]: s["end"]])
        parts.append(gap)
    y = np.concatenate(parts[:-1]).astype(np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak * 0.97).astype(np.float32)
    sf.write(str(out), y, SR)
    return len(y) / SR


def montage(wav16k: np.ndarray, segs: list[dict], out: Path, total_s: float = 20.0):
    """세그먼트들에서 앞부분을 모아 ~total_s 미리듣기 wav 생성."""
    budget = int(SR * total_s)
    parts, used = [], 0
    gap = np.zeros(int(SR * 0.2), np.float32)
    for s in segs:
        clip = wav16k[s["start"]: s["end"]]
        take = clip[: min(len(clip), int(SR * 3.0))]
        parts.append(take)
        parts.append(gap)
        used += len(take) + len(gap)
        if used >= budget:
            break
    y = (np.concatenate(parts) if parts else np.zeros(1, np.float32)).astype(np.float32)
    sf.write(str(out), y, SR)
