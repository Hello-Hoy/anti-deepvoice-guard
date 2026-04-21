#!/usr/bin/env python3
"""Preprocess voice-clone recordings into a split manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.engines._manifest import save_manifest, validate_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
_WHISPER_MODELS: dict[str, Any] = {}


def convert_audio(src: Path, dst: Path, sr: int = 24000, channels: int = 1) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ar",
            str(sr),
            "-ac",
            str(channels),
            "-f",
            "wav",
            str(dst),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    return result.returncode == 0


def true_peak_dbfs(wav: np.ndarray, sr: int) -> float:
    if wav.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(wav)))
    try:
        import librosa

        oversampled = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=sr * 4)
        peak = max(peak, float(np.max(np.abs(oversampled))) if oversampled.size else peak)
    except Exception:
        pass
    if peak <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(max(peak, 1e-12)))


def has_clipping(wav: np.ndarray, threshold: float = 0.999) -> bool:
    if wav.size == 0:
        return False
    clipped = np.abs(wav) >= threshold
    if int(np.sum(clipped)) >= 3:
        return True
    return bool(np.any(clipped[:-1] & clipped[1:])) if clipped.size > 1 else bool(clipped[0])


def compute_snr(wav: np.ndarray, sr: int) -> float:
    if wav.size == 0:
        return 0.0
    frame = max(int(sr * 0.05), 1)
    usable = (wav.size // frame) * frame
    if usable < frame * 4:
        return 0.0
    frames = wav[:usable].reshape(-1, frame)
    levels = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    levels = np.clip(levels, 1e-8, None)
    noise = float(np.mean(np.percentile(levels, [10, 20])))
    speech = float(np.mean(np.percentile(levels, [80, 90])))
    if speech <= noise:
        return 0.0
    return float(20.0 * np.log10(speech / max(noise, 1e-8)))


def vad_trim(wav: np.ndarray, sr: int, aggressiveness: int = 2) -> np.ndarray:
    try:
        import webrtcvad
    except ModuleNotFoundError:
        return wav
    signal, vad_sr = _resample_for_vad(wav, sr)
    frame_samples = int(vad_sr * 0.03)
    frames = _frame_signal(signal, frame_samples)
    if not frames:
        return wav
    vad = webrtcvad.Vad(aggressiveness)
    voiced = [idx for idx, frame in enumerate(frames) if vad.is_speech(_float_to_pcm16(frame), vad_sr)]
    if not voiced:
        return wav
    start = max((voiced[0] * frame_samples) - int(0.1 * vad_sr), 0)
    end = min(((voiced[-1] + 1) * frame_samples) + int(0.1 * vad_sr), len(signal))
    scale = sr / vad_sr
    trimmed = wav[int(start * scale): int(end * scale)]
    return trimmed if trimmed.size else wav


def segment_audio(wav: np.ndarray, sr: int, min_sec: float = 3.0, max_sec: float = 15.0) -> list[tuple[int, int]]:
    try:
        import librosa
        intervals = librosa.effects.split(wav, top_db=40)
    except Exception:
        intervals = np.asarray([[0, len(wav)]])
    segments: list[tuple[int, int]] = []
    min_len = int(min_sec * sr)
    max_len = int(max_sec * sr)
    for start, end in intervals:
        cursor = int(start)
        end = int(end)
        while end - cursor > max_len:
            segments.append((cursor, cursor + max_len))
            cursor += max_len
        if end - cursor >= min_len:
            segments.append((cursor, end))
    return segments


def transcribe_whisper(wav_path: Path, model_name: str = "large-v3", language: str = "ko") -> str:
    try:
        import whisper
    except ModuleNotFoundError:
        warnings.warn("openai-whisper is not installed; skipping transcription.", RuntimeWarning)
        return ""
    model = _WHISPER_MODELS.get(model_name)
    if model is None:
        model = whisper.load_model(model_name)
        _WHISPER_MODELS[model_name] = model
    result = model.transcribe(str(wav_path), language=language, task="transcribe", fp16=False)
    return " ".join(str(result.get("text", "")).split())


def build_manifest(
    segment_items: list[dict[str, Any]],
    script_path: Path,
    speaker: str,
    split: tuple[int, int, int],
    split_seed: int,
    rejections: list[dict[str, Any]],
    target_sr_train: int,
    target_sr_eval: int,
) -> dict[str, Any]:
    script_lines = _load_script_lines(script_path)
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(segment_items):
        text = script_lines[index] if index < len(script_lines) else str(item.get("whisper_text") or "")
        samples.append(
            {
                "id": item["id"],
                "wav_path_24k": item["wav_path_24k"],
                "wav_path_16k": item["wav_path_16k"],
                "text": text,
                "text_normalized": _normalize_text(text),
                "duration_sec": float(item["duration_sec"]),
                "snr_db": float(item["snr_db"]),
                "verified": True,
                "metadata": {
                    "true_peak_dbfs": item["true_peak_dbfs"],
                    "clipping": item["clipping"],
                    "split_seed": split_seed,
                },
            }
        )
        if item.get("whisper_text"):
            samples[-1]["whisper_text"] = item["whisper_text"]

    ids = [sample["id"] for sample in samples]
    shuffled = ids[:]
    random.Random(split_seed).shuffle(shuffled)
    train_pct, val_pct, test_pct = split
    total = len(shuffled)
    train_n = min(max(round(total * train_pct / 100), 1), max(total - 2, 1)) if total >= 3 else max(total - 1, 0)
    val_n = max(round(total * val_pct / 100), 1) if total - train_n >= 2 else max(total - train_n, 0)
    train_ids = sorted(shuffled[:train_n])
    val_ids = sorted(shuffled[train_n: train_n + val_n])
    test_ids = sorted(shuffled[train_n + val_n:])
    for sample in samples:
        if sample["id"] in train_ids:
            sample["split"] = "train"
        elif sample["id"] in val_ids:
            sample["split"] = "eval"
        else:
            sample["split"] = "test"

    manifest = {
        "speaker": speaker,
        "version": "1.0",
        "sample_rate_train": target_sr_train,
        "sample_rate_eval": target_sr_eval,
        "samples": samples,
        "split": {"train": train_ids, "eval": val_ids, "test": test_ids},
        "stats": {
            "total_segments": len(samples),
            "total_duration_sec": round(sum(float(s["duration_sec"]) for s in samples), 3),
            "avg_snr_db": round(sum(float(s["snr_db"]) for s in samples) / max(len(samples), 1), 3),
            "split_ratio": f"{train_pct}/{val_pct}/{test_pct}",
            "split_seed": split_seed,
            "rejections": rejections,
        },
    }
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    return manifest


def preprocess(
    raw_dir: Path,
    out_dir: Path,
    speaker: str,
    script_path: Path,
    whisper_model: str = "large-v3",
    target_sr_train: int = 24000,
    target_sr_eval: int = 16000,
    skip_transcription: bool = False,
    min_snr_db: float = 35.0,
    max_true_peak_dbfs: float = -1.0,
    split: tuple[int, int, int] = (75, 10, 15),
    split_seed: int = 42,
) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw recording directory not found: {raw_dir}")
    audio_files = sorted(path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO)
    if not audio_files:
        raise ValueError(f"No supported audio files found in {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = out_dir / "_workspace"
    raw_train_dir = workspace / "raw_train"
    wavs_train_dir = out_dir / "wavs_24k"
    wavs_eval_dir = out_dir / "wavs_16k"
    segment_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    clip_counter = 1

    for file_index, src_path in enumerate(audio_files, start=1):
        converted = raw_train_dir / f"{_safe_stem(src_path, file_index)}.wav"
        if not convert_audio(src_path, converted, sr=target_sr_train, channels=1):
            rejections.append({"source_file": src_path.name, "reason": "ffmpeg_convert_failed"})
            continue
        wav, sr = _read_audio(converted)
        trimmed = vad_trim(wav, sr)
        snr_db = compute_snr(trimmed, sr)
        clipping = has_clipping(trimmed)
        peak_db = true_peak_dbfs(trimmed, sr)
        if snr_db < min_snr_db or clipping or peak_db > max_true_peak_dbfs:
            rejections.append(
                {
                    "source_file": src_path.name,
                    "reason": "quality_filter",
                    "snr_db": round(snr_db, 3),
                    "clipping": clipping,
                    "true_peak_dbfs": round(peak_db, 3),
                }
            )
            continue
        for start, end in segment_audio(trimmed, sr):
            segment = trimmed[start:end]
            if segment.size == 0:
                continue
            clip_id = f"clip_{clip_counter:04d}"
            clip_counter += 1
            out_train = wavs_train_dir / f"{clip_id}.wav"
            out_eval = wavs_eval_dir / f"{clip_id}.wav"
            _write_audio(out_train, segment, sr)
            if not convert_audio(out_train, out_eval, sr=target_sr_eval, channels=1):
                rejections.append({"source_file": src_path.name, "clip_id": clip_id, "reason": "eval_convert_failed"})
                continue
            whisper_text = "" if skip_transcription else transcribe_whisper(out_train, model_name=whisper_model, language="ko")
            segment_rows.append(
                {
                    "id": clip_id,
                    "source_file": src_path.name,
                    "wav_path_24k": f"wavs_24k/{out_train.name}",
                    "wav_path_16k": f"wavs_16k/{out_eval.name}",
                    "duration_sec": round(len(segment) / sr, 4),
                    "snr_db": round(snr_db, 4),
                    "true_peak_dbfs": round(peak_db, 4),
                    "clipping": clipping,
                    "whisper_text": whisper_text,
                }
            )

    if not segment_rows:
        raise ValueError("No usable segments were produced from the input recordings.")

    (out_dir / "segment_index.json").write_text(
        json.dumps(
            {
                "speaker": speaker,
                "sample_rate_train": target_sr_train,
                "sample_rate_eval": target_sr_eval,
                "segments": segment_rows,
                "rejections": rejections,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_manifest(segment_rows, script_path, speaker, split, split_seed, rejections, target_sr_train, target_sr_eval)
    manifest_path = out_dir / "manifest.json"
    save_manifest(manifest, manifest_path)
    return manifest_path


def _safe_stem(path: Path, index: int) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in path.stem)
    return cleaned or f"recording_{index:03d}"


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        return wav, int(sr)
    except Exception:
        import librosa

        wav, sr = librosa.load(path, sr=None, mono=True)
        return np.asarray(wav, dtype=np.float32), int(sr)


def _write_audio(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)


def _resample_for_vad(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    target_sr = 16000
    mono = np.asarray(wav, dtype=np.float32)
    if sr == target_sr:
        return mono, sr
    import librosa

    return np.asarray(librosa.resample(mono, orig_sr=sr, target_sr=target_sr), dtype=np.float32), target_sr


def _frame_signal(signal: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    usable = len(signal) - (len(signal) % frame_samples)
    if usable <= 0:
        return []
    return [signal[index:index + frame_samples] for index in range(0, usable, frame_samples)]


def _float_to_pcm16(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _load_script_lines(script_path: Path) -> list[str]:
    return [line.strip() for line in script_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def _normalize_text(text: str) -> str:
    return _load_normalize()(text)


def _load_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_normalize() -> Callable[[str], str]:
    module = _load_module_from_path(SCRIPT_DIR / "scripts" / "normalize_korean.py", "voice_clone_normalize_korean")
    normalize = getattr(module, "normalize", None)
    if not callable(normalize):
        raise AttributeError("normalize_korean.py does not expose normalize(text)")
    return normalize


def _parse_split(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split("/")]
    if len(parts) != 3 or sum(parts) != 100:
        raise argparse.ArgumentTypeError("--split must be like 75/10/15 and sum to 100")
    return parts[0], parts[1], parts[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="voice-clone recording preprocessing")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speaker", type=str, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--whisper-model", type=str, default="large-v3")
    parser.add_argument("--target-sr-train", type=int, default=24000)
    parser.add_argument("--target-sr-eval", type=int, default=16000)
    parser.add_argument("--skip-transcription", action="store_true")
    parser.add_argument("--min-snr-db", "--min-snr", dest="min_snr_db", type=float, default=35.0)
    parser.add_argument("--max-true-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--split", type=_parse_split, default=(75, 10, 15))
    parser.add_argument("--split-seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    manifest_path = preprocess(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        speaker=args.speaker,
        script_path=args.script,
        whisper_model=args.whisper_model,
        target_sr_train=args.target_sr_train,
        target_sr_eval=args.target_sr_eval,
        skip_transcription=args.skip_transcription,
        min_snr_db=args.min_snr_db,
        max_true_peak_dbfs=args.max_true_peak_dbfs,
        split=args.split,
        split_seed=args.split_seed,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
