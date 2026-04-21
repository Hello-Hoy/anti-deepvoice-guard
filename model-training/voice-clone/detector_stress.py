#!/usr/bin/env python3
"""Generate detector stress conditions and score them with AASIST."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone._audio_utils import SAMPLE_RATE_16K, list_audio_files, load_wav_16k, load_wav_generic, match_length, rms, run_ffmpeg, write_wav
from model_training.voice_clone.eval_quality import DEFAULT_AASIST_ONNX, compute_aasist_score


DEFAULT_CONDITIONS = [
    "clean",
    "g711",
    "amr_nb",
    "opus",
    "reverb_0.3",
    "reverb_0.6",
    "noise_20db",
    "noise_10db",
    "noise_0db",
    "mp3_128",
    "mp3_64",
    "aac_96",
    "bandpass_telephony",
    "agc",
    "compressor",
    "soft_clipping",
    "packet_loss_1",
    "packet_loss_3",
    "packet_loss_5",
    "resample_chain_16_8_16",
]


def _safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "sample"


def apply_codec(wav_path: Path, condition: str, out_path: Path) -> Path:
    if condition == "g711":
        temp = out_path.with_suffix(".mulaw.wav")
        run_ffmpeg(["-i", str(wav_path), "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(temp)])
        run_ffmpeg(["-i", str(temp), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
        temp.unlink(missing_ok=True)
    elif condition == "amr_nb":
        temp = out_path.with_suffix(".amr")
        run_ffmpeg(["-i", str(wav_path), "-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", str(temp)])
        run_ffmpeg(["-i", str(temp), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
        temp.unlink(missing_ok=True)
    elif condition == "opus":
        temp = out_path.with_suffix(".opus")
        run_ffmpeg(["-i", str(wav_path), "-ar", "16000", "-ac", "1", "-c:a", "libopus", "-b:a", "24k", str(temp)])
        run_ffmpeg(["-i", str(temp), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
        temp.unlink(missing_ok=True)
    elif condition in {"mp3_128", "mp3_64"}:
        bitrate = "128k" if condition == "mp3_128" else "64k"
        temp = out_path.with_suffix(".mp3")
        run_ffmpeg(["-i", str(wav_path), "-ar", "16000", "-ac", "1", "-codec:a", "libmp3lame", "-b:a", bitrate, str(temp)])
        run_ffmpeg(["-i", str(temp), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
        temp.unlink(missing_ok=True)
    elif condition == "aac_96":
        temp = out_path.with_suffix(".m4a")
        run_ffmpeg(["-i", str(wav_path), "-ar", "16000", "-ac", "1", "-c:a", "aac", "-b:a", "96k", str(temp)])
        run_ffmpeg(["-i", str(temp), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
        temp.unlink(missing_ok=True)
    else:
        raise ValueError(condition)
    return out_path


def apply_filter(wav_path: Path, condition: str, out_path: Path) -> Path:
    filters = {
        "bandpass_telephony": "highpass=f=300,lowpass=f=3400",
        "agc": "dynaudnorm=f=150:g=15",
        "compressor": "acompressor=threshold=-18dB:ratio=4:attack=5:release=80",
        "resample_chain_16_8_16": "aresample=8000,aresample=16000",
    }
    run_ffmpeg(["-i", str(wav_path), "-filter:a", filters[condition], "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
    return out_path


def apply_reverb(wav_path: Path, rt60: float, out_path: Path) -> Path:
    audio = load_wav_16k(wav_path)
    try:
        from pedalboard import Pedalboard, Reverb

        processed = Pedalboard([Reverb(room_size=min(max(rt60 / 1.2, 0.1), 1.0), wet_level=0.35)])(audio, sample_rate=SAMPLE_RATE_16K)
    except ModuleNotFoundError:
        impulse_len = int(SAMPLE_RATE_16K * rt60)
        impulse = np.exp(-np.linspace(0, 5, max(impulse_len, 128))).astype(np.float32)
        processed = np.convolve(audio, impulse, mode="full")[: audio.size]
        processed = 0.7 * audio + 0.3 * processed
    write_wav(out_path, np.asarray(processed, dtype=np.float32) / max(float(np.max(np.abs(processed))), 1.0), SAMPLE_RATE_16K)
    return out_path


def apply_noise(wav_path: Path, snr_db: float, noise_dir: Path | None, out_path: Path) -> Path:
    clean = load_wav_16k(wav_path)
    if noise_dir and noise_dir.exists():
        candidates = list_audio_files(noise_dir)
        noise, _ = load_wav_generic(random.choice(candidates)) if candidates else (np.random.normal(0, 1, clean.size), SAMPLE_RATE_16K)
        noise = match_length(noise, clean.size)
    else:
        noise = np.random.default_rng(abs(hash((str(wav_path), snr_db))) % (2**32)).normal(0, 1, clean.size).astype(np.float32)
    scaled = noise * ((max(rms(clean), 1e-6) / (10 ** (snr_db / 20))) / max(rms(noise), 1e-6))
    mixed = clean + scaled
    write_wav(out_path, mixed / max(float(np.max(np.abs(mixed))), 1.0), SAMPLE_RATE_16K)
    return out_path


def apply_soft_clipping(wav_path: Path, out_path: Path) -> Path:
    audio = load_wav_16k(wav_path)
    processed = np.tanh(audio * 2.5) / np.tanh(2.5)
    write_wav(out_path, processed.astype(np.float32), SAMPLE_RATE_16K)
    return out_path


def apply_packet_loss(wav_path: Path, percent: float, out_path: Path) -> Path:
    audio = load_wav_16k(wav_path)
    frame = int(SAMPLE_RATE_16K * 0.02)
    rng = np.random.default_rng(abs(hash((str(wav_path), percent))) % (2**32))
    processed = audio.copy()
    for start in range(0, processed.size, frame):
        if rng.random() < percent / 100.0:
            processed[start:start + frame] = 0.0
    write_wav(out_path, processed, SAMPLE_RATE_16K)
    return out_path


def _apply_condition(wav_path: Path, condition: str, out_path: Path, noise_dir: Path | None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if condition == "clean":
        write_wav(out_path, load_wav_16k(wav_path), SAMPLE_RATE_16K)
    elif condition in {"g711", "amr_nb", "opus", "mp3_128", "mp3_64", "aac_96"}:
        apply_codec(wav_path, condition, out_path)
    elif condition.startswith("reverb_"):
        apply_reverb(wav_path, float(condition.split("_", 1)[1]), out_path)
    elif condition.startswith("noise_") and condition.endswith("db"):
        apply_noise(wav_path, float(condition[len("noise_"):-2]), noise_dir, out_path)
    elif condition in {"bandpass_telephony", "agc", "compressor", "resample_chain_16_8_16"}:
        apply_filter(wav_path, condition, out_path)
    elif condition == "soft_clipping":
        apply_soft_clipping(wav_path, out_path)
    elif condition.startswith("packet_loss_"):
        apply_packet_loss(wav_path, float(condition.rsplit("_", 1)[1]), out_path)
    else:
        raise ValueError(f"Unsupported stress condition: {condition}")
    return out_path


def generate_stress_set(clean_wavs: list[Path], out_dir: Path, conditions: list[str], noise_dir: Path | None) -> dict[str, list[Path]]:
    generated: dict[str, list[Path]] = {}
    for condition in conditions:
        outputs: list[Path] = []
        for index, wav in enumerate(clean_wavs):
            out = out_dir / condition / f"{index:04d}_{_safe_name(wav)}.wav"
            outputs.append(_apply_condition(wav, condition, out, noise_dir))
        generated[condition] = outputs
    return generated


def _stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "count": len(scores),
        "mean": round(float(statistics.fmean(scores)), 6),
        "median": round(float(statistics.median(scores)), 6),
        "std": round(float(statistics.stdev(scores)) if len(scores) > 1 else 0.0, 6),
    }


def stress_test(
    clone_dir: Path,
    real_dir: Path,
    aasist_onnx: Path,
    out_dir: Path,
    conditions: list[str],
    noise_dir: Path | None = None,
    seeds: int = 5,
) -> Path:
    real_wavs = list_audio_files(real_dir, patterns=("*.wav",))
    fake_wavs = list_audio_files(clone_dir, patterns=("*.wav",))[: max(seeds, 1)]
    if not real_wavs or not fake_wavs:
        raise RuntimeError("Both --real-dir and --clone-dir must contain WAV files.")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_real = generate_stress_set(real_wavs, out_dir / "generated" / "real", conditions, noise_dir)
    generated_fake = generate_stress_set(fake_wavs, out_dir / "generated" / "fake", conditions, noise_dir)
    payload: dict[str, Any] = {"conditions": conditions, "scores": {}, "summary": {}}
    for condition in conditions:
        real_scores = [compute_aasist_score(path, aasist_onnx) for path in generated_real[condition]]
        fake_scores = [compute_aasist_score(path, aasist_onnx) for path in generated_fake[condition]]
        payload["scores"][condition] = {
            "real": [{"path": str(path), "score": score} for path, score in zip(generated_real[condition], real_scores, strict=True)],
            "fake": [{"path": str(path), "score": score} for path, score in zip(generated_fake[condition], fake_scores, strict=True)],
        }
        payload["summary"][condition] = {"real": _stats(real_scores), "fake": _stats(fake_scores)}
    (out_dir / "raw.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = out_dir / "stress_report.md"
    rows = ["| condition | real_mean | fake_mean | fake_n |", "| --- | --- | --- | --- |"]
    for condition, summary in payload["summary"].items():
        rows.append(f"| {condition} | {summary['real']['mean']:.3f} | {summary['fake']['mean']:.3f} | {summary['fake']['count']} |")
    report.write_text("# Detector Stress Report\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return report


def _parse_conditions(value: str) -> list[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unsupported conditions: {', '.join(unknown)}")
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AASIST detector robustness stress test")
    parser.add_argument("--clone-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--aasist-onnx", type=Path, default=DEFAULT_AASIST_ONNX)
    parser.add_argument("--out", "--report", dest="out", type=Path, required=True)
    parser.add_argument("--conditions", type=_parse_conditions, default=DEFAULT_CONDITIONS)
    parser.add_argument("--noise-dir", type=Path)
    parser.add_argument("--seeds", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = stress_test(args.clone_dir, args.real_dir, args.aasist_onnx, args.out, args.conditions, args.noise_dir, args.seeds)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
