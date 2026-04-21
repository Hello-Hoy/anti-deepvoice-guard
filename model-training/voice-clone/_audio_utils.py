from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable
import wave

import numpy as np


SAMPLE_RATE_16K = 16000
AASIST_NUM_SAMPLES = 64600


def ensure_mono(wav: np.ndarray) -> np.ndarray:
    mono = np.asarray(wav, dtype=np.float32)
    if mono.ndim == 1:
        return mono
    return np.mean(mono, axis=1, dtype=np.float32)


def load_wav_generic(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        wav, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return ensure_mono(np.asarray(wav, dtype=np.float32)), int(sample_rate)
    except Exception:
        try:
            import librosa
        except ModuleNotFoundError as exc:
            return _load_audio_via_ffmpeg(path)

        wav, sample_rate = librosa.load(path, sr=None, mono=True)
        return ensure_mono(np.asarray(wav, dtype=np.float32)), int(sample_rate)


def resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    mono = ensure_mono(wav)
    if orig_sr == target_sr:
        return mono

    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "리샘플링에 `librosa`가 필요합니다. `pip install librosa` 후 다시 시도하세요."
        ) from exc

    resampled = librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr)
    return np.asarray(resampled, dtype=np.float32)


def load_wav_16k(path: Path) -> np.ndarray:
    wav, sample_rate = load_wav_generic(path)
    return resample_audio(wav, orig_sr=sample_rate, target_sr=SAMPLE_RATE_16K)


def write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "WAV 저장에 `soundfile`이 필요합니다. `pip install soundfile` 후 다시 시도하세요."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, ensure_mono(wav), sample_rate)


def pad_or_trim_audio(wav: np.ndarray, max_len: int = AASIST_NUM_SAMPLES) -> np.ndarray:
    audio = ensure_mono(wav)
    if audio.size == 0:
        raise ValueError("오디오 길이가 0입니다.")
    if audio.shape[0] >= max_len:
        return np.asarray(audio[:max_len], dtype=np.float32)

    repeats = (max_len // audio.shape[0]) + 1
    return np.tile(audio, repeats)[:max_len].astype(np.float32)


def load_aasist_input(path: Path, max_len: int = AASIST_NUM_SAMPLES) -> np.ndarray:
    return pad_or_trim_audio(load_wav_16k(path), max_len=max_len)


def list_audio_files(directory: Path, patterns: Iterable[str] | None = None) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"디렉토리가 존재하지 않습니다: {directory}")

    globs = tuple(patterns or ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg"))
    files = {
        path
        for pattern in globs
        for path in directory.rglob(pattern)
        if path.is_file()
    }
    return sorted(files)


def require_ffmpeg() -> str:
    binary = os.environ.get("FFMPEG_BIN", "ffmpeg")
    resolved = shutil.which(binary) if os.path.sep not in binary else binary
    if not resolved or not Path(resolved).exists():
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다. 시스템에 ffmpeg를 설치하거나 `FFMPEG_BIN` 환경변수를 설정하세요."
        )
    return resolved


def run_ffmpeg(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    binary = require_ffmpeg()
    result = subprocess.run(
        [binary, "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(stderr 없음)"
        raise RuntimeError(f"ffmpeg 실행 실패: {' '.join(args)}\n{stderr}")
    return result


def convert_audio(src: Path, dst: Path, sr: int = 24000, channels: int = 1) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(
            [
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
            timeout=300,
        )
    except Exception:
        return False
    return True


def _load_audio_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        run_ffmpeg(["-i", str(path), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(temp_path)])
        with wave.open(str(temp_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
        if sample_width != 2:
            raise RuntimeError(f"지원하지 않는 PCM sample width: {sample_width}")
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, int(sample_rate)
    except Exception as exc:
        raise RuntimeError(
            "오디오 로딩에 실패했습니다. `pip install soundfile librosa` 또는 ffmpeg 설치 상태를 확인하세요."
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


def rms(signal: np.ndarray) -> float:
    audio = ensure_mono(signal)
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    mono = ensure_mono(audio)
    if mono.shape[0] == target_len:
        return mono.astype(np.float32)
    if mono.shape[0] > target_len:
        start = max((mono.shape[0] - target_len) // 2, 0)
        return mono[start:start + target_len].astype(np.float32)
    repeats = (target_len // mono.shape[0]) + 1
    return np.tile(mono, repeats)[:target_len].astype(np.float32)
