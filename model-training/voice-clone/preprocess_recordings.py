#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable, Iterable
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.engines._manifest import save_manifest, validate_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
_WHISPER_MODELS: dict[str, Any] = {}


def convert_audio(src: Path, dst: Path, sr: int = 24000, channels: int = 1) -> bool:
    """ffmpeg로 WAV를 변환한다."""
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
        timeout=120,
        check=False,
    )
    return result.returncode == 0


def compute_snr(wav: np.ndarray, sr: int) -> float:
    """VAD voiced/unvoiced RMS 비율로 근사 SNR(dB)을 계산한다."""
    vad_signal, vad_sr = _resample_for_vad(wav, sr)
    frame_samples = int(vad_sr * 0.03)
    frames = _frame_signal(vad_signal, frame_samples)
    if not frames:
        return 0.0

    vad = _lazy_import_webrtcvad().Vad(2)
    voiced_rms: list[float] = []
    noise_rms: list[float] = []

    for frame in frames:
        pcm = _float_to_pcm16(frame)
        is_voiced = vad.is_speech(pcm, vad_sr)
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        if is_voiced:
            voiced_rms.append(rms)
        else:
            noise_rms.append(rms)

    if not voiced_rms:
        return 0.0

    speech_floor = max(float(np.mean(voiced_rms)), 1e-8)
    noise_floor = max(float(np.mean(noise_rms)) if noise_rms else 1e-4, 1e-8)
    return float(20.0 * np.log10(speech_floor / noise_floor))


def vad_trim(
    wav: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    min_voiced_ms: int = 300,
) -> np.ndarray:
    """webrtcvad로 앞뒤 무음을 제거한다."""
    vad_signal, vad_sr = _resample_for_vad(wav, sr)
    frame_samples = int(vad_sr * 0.03)
    frames = _frame_signal(vad_signal, frame_samples)
    if not frames:
        return wav

    vad = _lazy_import_webrtcvad().Vad(aggressiveness)
    voiced_flags = [vad.is_speech(_float_to_pcm16(frame), vad_sr) for frame in frames]
    voiced_indices = [index for index, flag in enumerate(voiced_flags) if flag]
    if not voiced_indices:
        return wav

    voiced_duration_ms = len(voiced_indices) * 30
    if voiced_duration_ms < min_voiced_ms:
        return wav

    start_index = voiced_indices[0] * frame_samples
    end_index = min((voiced_indices[-1] + 1) * frame_samples, len(vad_signal))
    scale = sr / vad_sr
    start = max(int(start_index * scale), 0)
    end = min(int(end_index * scale), len(wav))
    trimmed = wav[start:end]
    return trimmed if trimmed.size else wav


def segment_audio(
    wav: np.ndarray,
    sr: int,
    min_sec: float = 3.0,
    max_sec: float = 14.0,
    split_db: float = -40,
) -> list[tuple[int, int]]:
    """silence 기반으로 세그먼트를 나누고 길이 제약을 적용한다."""
    librosa = _lazy_import_librosa()
    intervals = librosa.effects.split(wav, top_db=abs(int(split_db)))
    if len(intervals) == 0:
        if len(wav) / sr >= min_sec:
            return [(0, len(wav))]
        return []

    merged: list[tuple[int, int]] = []
    gap_limit = int(sr * 0.35)
    for start, end in intervals:
        if not merged:
            merged.append((int(start), int(end)))
            continue
        last_start, last_end = merged[-1]
        if start - last_end <= gap_limit and (end - last_start) / sr <= max_sec:
            merged[-1] = (last_start, int(end))
        else:
            merged.append((int(start), int(end)))

    normalized: list[tuple[int, int]] = []
    max_samples = int(max_sec * sr)
    min_samples = int(min_sec * sr)

    for start, end in merged:
        length = end - start
        if length < min_samples:
            continue
        if length <= max_samples:
            normalized.append((start, end))
            continue

        cursor = start
        while cursor < end:
            next_end = min(cursor + max_samples, end)
            if end - next_end < min_samples and next_end != end:
                next_end = end
            if next_end - cursor >= min_samples:
                normalized.append((cursor, next_end))
            cursor = next_end

    return normalized


def transcribe_whisper(
    wav_path: Path,
    model_name: str = "large-v3",
    language: str = "ko",
) -> str:
    """openai-whisper로 전사한다. 실패 시 빈 문자열을 반환한다."""
    try:
        import whisper
    except ModuleNotFoundError:
        warnings.warn("openai-whisper is not installed; skipping transcription.", RuntimeWarning)
        return ""

    model = _WHISPER_MODELS.get(model_name)
    if model is None:
        model = whisper.load_model(model_name)
        _WHISPER_MODELS[model_name] = model

    try:
        result = model.transcribe(str(wav_path), language=language, task="transcribe", fp16=False)
    except Exception as exc:  # pragma: no cover - depends on runtime model/device
        warnings.warn(f"Whisper transcription failed for {wav_path.name}: {exc}", RuntimeWarning)
        return ""

    text = str(result.get("text", "")).strip()
    return " ".join(text.split())


def align_with_script(
    segments_text: list[str],
    script_lines: list[str],
) -> list[tuple[int | None, float]]:
    """Whisper 전사 결과를 대본 라인에 순차적으로 정렬한다."""
    from difflib import SequenceMatcher

    normalized_script = [_normalize_for_match(line) for line in script_lines]
    search_start = 0
    alignments: list[tuple[int | None, float]] = []

    for segment_text in segments_text:
        normalized_segment = _normalize_for_match(segment_text)
        if not normalized_segment:
            alignments.append((None, 0.0))
            continue

        best_index: int | None = None
        best_score = 0.0
        for index in range(search_start, len(script_lines)):
            score = SequenceMatcher(None, normalized_segment, normalized_script[index]).ratio()
            if score > best_score:
                best_index = index
                best_score = score

        if best_index is not None and best_score >= 0.35:
            alignments.append((best_index, float(best_score)))
            search_start = best_index + 1
        else:
            alignments.append((None, float(best_score)))

    return alignments


def build_manifest(
    segments_dir: Path,
    script_path: Path,
    speaker: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """세그먼트 인덱스와 대본으로 공통 manifest를 구성한다."""
    segment_index_path = segments_dir / "segment_index.json"
    if not segment_index_path.exists():
        raise FileNotFoundError(f"Segment index not found: {segment_index_path}")

    segment_index = json.loads(segment_index_path.read_text(encoding="utf-8"))
    segment_items = segment_index.get("segments", [])
    if not segment_items:
        raise ValueError(f"No segments were recorded in {segment_index_path}")

    script_lines = _load_script_lines(script_path)
    whisper_texts = [str(item.get("whisper_text", "")).strip() for item in segment_items]
    alignments = align_with_script(whisper_texts, script_lines) if any(whisper_texts) else []

    used_script_indices: set[int] = set()
    next_script_index = 0
    samples: list[dict[str, Any]] = []

    for index, item in enumerate(segment_items):
        matched_index = None
        match_score = 0.0
        if alignments:
            matched_index, match_score = alignments[index]
            if matched_index is not None and matched_index in used_script_indices:
                matched_index = None

        if matched_index is None:
            while next_script_index in used_script_indices and next_script_index < len(script_lines):
                next_script_index += 1
            if next_script_index < len(script_lines):
                matched_index = next_script_index

        if matched_index is None or matched_index >= len(script_lines):
            raise ValueError(
                f"Could not map segment {item['id']} to a script line. "
                f"Check recording order or disable transcription and verify manually."
            )

        used_script_indices.add(matched_index)
        next_script_index = matched_index + 1
        text = script_lines[matched_index]
        whisper_text = str(item.get("whisper_text", "")).strip()
        verified = bool(match_score >= 0.6 or not whisper_text)
        sample = {
            "id": item["id"],
            "wav_path_24k": item["wav_path_24k"],
            "wav_path_16k": item["wav_path_16k"],
            "text": text,
            "text_normalized": _normalize_text(text),
            "duration_sec": float(item["duration_sec"]),
            "snr_db": float(item["snr_db"]),
            "verified": verified,
        }
        if whisper_text:
            sample["whisper_text"] = whisper_text
            sample["whisper_match"] = round(float(match_score), 4)
        samples.append(sample)

    ids = [sample["id"] for sample in samples]
    shuffled_ids = ids[:]
    random.Random(seed).shuffle(shuffled_ids)
    train_cutoff = max(1, int(len(shuffled_ids) * train_ratio))
    if len(shuffled_ids) > 1:
        train_cutoff = min(train_cutoff, len(shuffled_ids) - 1)

    coverage_result = _load_coverage_analyze()([sample["text"] for sample in samples])
    stats = {
        "total_segments": len(samples),
        "total_duration_sec": round(sum(sample["duration_sec"] for sample in samples), 3),
        "avg_snr_db": round(
            sum(sample.get("snr_db", 0.0) for sample in samples) / max(len(samples), 1),
            3,
        ),
        "verified_segments": sum(1 for sample in samples if sample["verified"]),
        "coverage_matrix": coverage_result,
    }

    manifest = {
        "speaker": speaker,
        "version": "1.0",
        "sample_rate_train": int(segment_index.get("sample_rate_train", 24000)),
        "sample_rate_eval": int(segment_index.get("sample_rate_eval", 16000)),
        "samples": samples,
        "split": {
            "train": sorted(shuffled_ids[:train_cutoff]),
            "eval": sorted(shuffled_ids[train_cutoff:]),
        },
        "stats": stats,
    }
    if not manifest["split"]["eval"]:
        manifest["split"]["eval"] = [manifest["split"]["train"].pop()]

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
) -> Path:
    """녹음 폴더를 전처리해 공통 manifest를 생성한다."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw recording directory not found: {raw_dir}")
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")

    audio_files = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    )
    if not audio_files:
        raise ValueError(f"No supported audio files found in {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = out_dir / "_workspace"
    raw_24k_dir = workspace / "raw_24k"
    raw_16k_dir = workspace / "raw_16k"
    wavs_24k_dir = out_dir / "wavs_24k"
    wavs_16k_dir = out_dir / "wavs_16k"
    segment_index_path = out_dir / "segment_index.json"

    segment_rows: list[dict[str, Any]] = []
    clip_counter = 1

    for file_index, src_path in enumerate(audio_files, start=1):
        base_name = _safe_stem(src_path, file_index)
        converted_24k = raw_24k_dir / f"{base_name}.wav"
        converted_16k = raw_16k_dir / f"{base_name}.wav"

        if not convert_audio(src_path, converted_24k, sr=target_sr_train, channels=1):
            raise RuntimeError(f"ffmpeg failed to convert {src_path} to {target_sr_train}Hz WAV")
        if not convert_audio(src_path, converted_16k, sr=target_sr_eval, channels=1):
            raise RuntimeError(f"ffmpeg failed to convert {src_path} to {target_sr_eval}Hz WAV")

        wav_24k, sr_24k = _read_audio(converted_24k)
        trimmed = vad_trim(wav_24k, sr_24k)
        snr_db = compute_snr(trimmed, sr_24k)
        if snr_db < 20.0:
            warnings.warn(
                f"Low SNR detected for {src_path.name}: {snr_db:.2f} dB",
                RuntimeWarning,
            )

        segments = segment_audio(trimmed, sr_24k)
        if not segments and len(trimmed) / sr_24k >= 3.0:
            segments = [(0, len(trimmed))]

        for start, end in segments:
            segment_wav = trimmed[start:end]
            if segment_wav.size == 0:
                continue
            clip_id = f"clip_{clip_counter:04d}"
            clip_counter += 1
            out_24k = wavs_24k_dir / f"{clip_id}.wav"
            out_16k = wavs_16k_dir / f"{clip_id}.wav"

            _write_audio(out_24k, segment_wav, sr_24k)
            if not convert_audio(out_24k, out_16k, sr=target_sr_eval, channels=1):
                raise RuntimeError(f"ffmpeg failed to convert segment {clip_id} to {target_sr_eval}Hz")

            whisper_text = ""
            if not skip_transcription:
                whisper_text = transcribe_whisper(out_24k, model_name=whisper_model, language="ko")

            segment_rows.append(
                {
                    "id": clip_id,
                    "source_file": src_path.name,
                    "wav_path_24k": f"wavs_24k/{out_24k.name}",
                    "wav_path_16k": f"wavs_16k/{out_16k.name}",
                    "duration_sec": round(len(segment_wav) / sr_24k, 4),
                    "snr_db": round(float(snr_db), 4),
                    "whisper_text": whisper_text,
                }
            )

    if not segment_rows:
        raise ValueError("No usable segments were produced from the input recordings.")

    segment_index = {
        "speaker": speaker,
        "sample_rate_train": target_sr_train,
        "sample_rate_eval": target_sr_eval,
        "segments": segment_rows,
    }
    segment_index_path.write_text(
        json.dumps(segment_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = build_manifest(out_dir, script_path, speaker=speaker)
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
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        return np.asarray(wav, dtype=np.float32), int(sr)
    except Exception:
        librosa = _lazy_import_librosa()
        wav, sr = librosa.load(path, sr=None, mono=True)
        return np.asarray(wav, dtype=np.float32), int(sr)


def _write_audio(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)


def _resample_for_vad(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    target_sr = 16000
    mono = np.asarray(wav, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1)
    if sr == target_sr:
        return mono, sr

    librosa = _lazy_import_librosa()
    resampled = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return np.asarray(resampled, dtype=np.float32), target_sr


def _frame_signal(signal: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    usable = len(signal) - (len(signal) % frame_samples)
    if usable <= 0:
        return []
    trimmed = signal[:usable]
    return [trimmed[index:index + frame_samples] for index in range(0, usable, frame_samples)]


def _float_to_pcm16(frame: np.ndarray) -> bytes:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    return "".join(_normalize_text(text).split())


def _normalize_text(text: str) -> str:
    return _load_normalize()(text)


def _load_script_lines(script_path: Path) -> list[str]:
    lines = script_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _load_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_normalize() -> Callable[[str], str]:
    module = _load_module_from_path(
        SCRIPT_DIR / "scripts" / "normalize_korean.py",
        "voice_clone_normalize_korean",
    )
    normalize = getattr(module, "normalize", None)
    if not callable(normalize):
        raise AttributeError("normalize_korean.py does not expose normalize(text: str) -> str")
    return normalize


def _load_coverage_analyze() -> Callable[[list[str]], dict[str, Any]]:
    module = _load_module_from_path(
        SCRIPT_DIR / "scripts" / "korean_coverage_matrix.py",
        "voice_clone_korean_coverage_matrix",
    )
    analyze = getattr(module, "analyze", None)
    if not callable(analyze):
        raise AttributeError("korean_coverage_matrix.py does not expose analyze(texts: list[str])")
    return analyze


def _lazy_import_librosa() -> Any:
    import librosa

    return librosa


def _lazy_import_webrtcvad() -> Any:
    import webrtcvad

    return webrtcvad


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="voice-clone 녹음 전처리 파이프라인")
    parser.add_argument("--raw-dir", type=Path, required=True, help="원본 녹음 폴더")
    parser.add_argument("--out-dir", type=Path, required=True, help="전처리 출력 폴더")
    parser.add_argument("--speaker", type=str, required=True, help="화자 이름")
    parser.add_argument("--script", type=Path, required=True, help="녹음 대본 파일")
    parser.add_argument("--whisper-model", type=str, default="large-v3")
    parser.add_argument("--target-sr-train", type=int, default=24000)
    parser.add_argument("--target-sr-eval", type=int, default=16000)
    parser.add_argument("--skip-transcription", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest_path = preprocess(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        speaker=args.speaker,
        script_path=args.script,
        whisper_model=args.whisper_model,
        target_sr_train=args.target_sr_train,
        target_sr_eval=args.target_sr_eval,
        skip_transcription=args.skip_transcription,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
