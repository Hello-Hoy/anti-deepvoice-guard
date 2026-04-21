#!/usr/bin/env python3
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

from model_training.voice_clone._audio_utils import (
    SAMPLE_RATE_16K,
    list_audio_files,
    load_wav_16k,
    load_wav_generic,
    match_length,
    rms,
    run_ffmpeg,
    write_wav,
)
from model_training.voice_clone.eval_quality import DEFAULT_AASIST_ONNX, compute_aasist_score


DEFAULT_CONDITIONS = [
    "clean",
    "g711",
    "amr_nb",
    "reverb_0.3",
    "reverb_0.6",
    "noise_20db",
    "noise_10db",
]


def _safe_name(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return cleaned or "sample"


def _seed_from_path(path: Path, salt: str) -> int:
    return abs(hash((str(path.resolve()), salt))) % (2**32)


def apply_codec(wav_path: Path, codec: str, out_path: Path) -> Path:
    codec_key = codec.lower()
    temp_encoded = out_path.with_suffix(f".{codec_key}.wav")

    if codec_key in {"g711", "mulaw", "pcmu"}:
        run_ffmpeg(
            [
                "-i", str(wav_path),
                "-ar", "8000",
                "-ac", "1",
                "-c:a", "pcm_mulaw",
                str(temp_encoded),
            ]
        )
        run_ffmpeg(["-i", str(temp_encoded), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
    elif codec_key in {"amr", "amr_nb"}:
        temp_encoded = out_path.with_suffix(".amr")
        run_ffmpeg(
            [
                "-i", str(wav_path),
                "-ar", "8000",
                "-ac", "1",
                "-c:a", "libopencore_amrnb",
                "-b:a", "12.2k",
                str(temp_encoded),
            ]
        )
        run_ffmpeg(["-i", str(temp_encoded), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
    elif codec_key == "opus":
        temp_encoded = out_path.with_suffix(".opus")
        run_ffmpeg(
            [
                "-i", str(wav_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "libopus",
                "-b:a", "24k",
                str(temp_encoded),
            ]
        )
        run_ffmpeg(["-i", str(temp_encoded), "-ar", str(SAMPLE_RATE_16K), "-ac", "1", str(out_path)])
    else:
        raise ValueError(f"지원하지 않는 codec: {codec}")

    temp_encoded.unlink(missing_ok=True)
    return out_path


def _fft_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n = signal.size + kernel.size - 1
    size = 1 << (n - 1).bit_length()
    signal_fft = np.fft.rfft(signal, size)
    kernel_fft = np.fft.rfft(kernel, size)
    return np.fft.irfft(signal_fft * kernel_fft, size)[:n].astype(np.float32)


def _fallback_reverb(audio: np.ndarray, sample_rate: int, rt60: float) -> np.ndarray:
    length = max(int(sample_rate * min(max(rt60 * 1.5, 0.2), 1.6)), 512)
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    for delay_ms, scale in ((25, 0.35), (70, 0.2), (120, 0.12), (180, 0.08)):
        index = int(sample_rate * delay_ms / 1000.0)
        if index < length:
            impulse[index] += scale
    decay = np.exp(-6.907755 * np.arange(length, dtype=np.float32) / max(sample_rate * rt60, 1.0))
    impulse *= decay
    impulse /= max(np.max(np.abs(impulse)), 1e-6)
    wet = _fft_convolve(audio, impulse)[:audio.shape[0]]
    mixed = 0.65 * audio + 0.35 * wet
    peak = max(np.max(np.abs(mixed)), 1.0)
    return (mixed / peak).astype(np.float32)


def apply_reverb(wav_path: Path, rt60: float, out_path: Path) -> Path:
    audio = load_wav_16k(wav_path)
    sample_rate = SAMPLE_RATE_16K

    try:
        from pedalboard import Pedalboard, Reverb

        board = Pedalboard([Reverb(room_size=min(max(rt60 / 1.2, 0.1), 1.0), damping=0.3, wet_level=0.35)])
        processed = board(audio, sample_rate=sample_rate)
        processed = np.asarray(processed, dtype=np.float32)
        if processed.ndim > 1:
            processed = np.mean(processed, axis=0, dtype=np.float32)
    except ModuleNotFoundError:
        processed = _fallback_reverb(np.asarray(audio, dtype=np.float32), sample_rate, rt60)

    write_wav(out_path, processed, sample_rate)
    return out_path


def _choose_noise(noise_dir: Path | None, target_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if noise_dir and noise_dir.exists():
        candidates = list_audio_files(noise_dir)
        if candidates:
            chosen = candidates[seed % len(candidates)]
            noise, _ = load_wav_generic(chosen)
            return match_length(noise, target_len)
    return rng.normal(0.0, 1.0, size=target_len).astype(np.float32)


def apply_noise(wav_path: Path, snr_db: float, noise_dir: Path | None, out_path: Path) -> Path:
    clean = load_wav_16k(wav_path)
    seed = _seed_from_path(wav_path, f"noise-{snr_db}")
    noise = _choose_noise(noise_dir, clean.shape[0], seed)

    clean_rms = max(rms(clean), 1e-6)
    noise_rms = max(rms(noise), 1e-6)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = clean + scaled_noise
    peak = max(np.max(np.abs(mixed)), 1.0)
    write_wav(out_path, mixed / peak, SAMPLE_RATE_16K)
    return out_path


def apply_gain_tempo(wav_path: Path, gain_db: float, tempo_ratio: float, out_path: Path) -> Path:
    filters = [f"volume={gain_db}dB", f"atempo={tempo_ratio}"]
    try:
        run_ffmpeg(
            [
                "-i", str(wav_path),
                "-filter:a", ",".join(filters),
                "-ar", str(SAMPLE_RATE_16K),
                "-ac", "1",
                str(out_path),
            ]
        )
        return out_path
    except RuntimeError:
        audio = load_wav_16k(wav_path)
        sample_rate = SAMPLE_RATE_16K
        gain = 10 ** (gain_db / 20.0)
        audio = audio * gain
        if not math.isclose(tempo_ratio, 1.0):
            try:
                import librosa
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "gain/tempo fallback에는 `librosa`가 필요합니다. `pip install librosa` 후 다시 시도하세요."
                ) from exc
            audio = librosa.effects.time_stretch(np.asarray(audio, dtype=np.float32), rate=tempo_ratio)
        peak = max(np.max(np.abs(audio)), 1.0)
        write_wav(out_path, audio / peak, sample_rate)
        return out_path


def _apply_condition(wav_path: Path, condition: str, out_path: Path, noise_dir: Path | None) -> Path:
    if condition == "clean":
        audio = load_wav_16k(wav_path)
        write_wav(out_path, audio, SAMPLE_RATE_16K)
        return out_path
    if condition == "g711":
        return apply_codec(wav_path, "g711", out_path)
    if condition == "amr_nb":
        return apply_codec(wav_path, "amr_nb", out_path)
    if condition == "opus":
        return apply_codec(wav_path, "opus", out_path)
    if condition.startswith("reverb_"):
        rt60 = float(condition.split("_", 1)[1])
        return apply_reverb(wav_path, rt60, out_path)
    if condition.startswith("noise_") and condition.endswith("db"):
        snr = float(condition[len("noise_"):-2])
        return apply_noise(wav_path, snr, noise_dir, out_path)
    if condition.startswith("gain_"):
        value = condition[len("gain_"):].replace("p", "").replace("m", "-").replace("db", "")
        return apply_gain_tempo(wav_path, float(value), 1.0, out_path)
    if condition.startswith("tempo_"):
        value = condition[len("tempo_"):].replace("pct", "")
        if value.endswith("%"):
            ratio = 1.0 + (float(value[:-1]) / 100.0)
        else:
            ratio = float(value)
        return apply_gain_tempo(wav_path, 0.0, ratio, out_path)
    raise ValueError(f"지원하지 않는 stress condition: {condition}")


def generate_stress_set(
    clean_wavs: list[Path],
    out_dir: Path,
    conditions: list[str] = DEFAULT_CONDITIONS,
    noise_dir: Path | None = None,
) -> dict[str, list[Path]]:
    generated: dict[str, list[Path]] = {}
    for condition in conditions:
        condition_outputs: list[Path] = []
        condition_dir = out_dir / condition
        for index, wav_path in enumerate(clean_wavs):
            output_name = f"{index:04d}_{_safe_name(wav_path)}.wav"
            out_path = condition_dir / output_name
            condition_outputs.append(_apply_condition(wav_path, condition, out_path, noise_dir))
        generated[condition] = condition_outputs
    return generated


def score_aasist_batch(wavs: list[Path], onnx_path: Path) -> list[float]:
    return [compute_aasist_score(wav, onnx_path) for wav in wavs]


def build_paired_index(real_wavs: list[Path], fake_wavs: list[Path]) -> dict[str, dict[str, Any]]:
    real_map = {_safe_name(path): path for path in real_wavs}
    sorted_real_ids = sorted(real_map, key=len, reverse=True)
    paired: dict[str, dict[str, Any]] = {}

    for sample_id, real_path in real_map.items():
        paired[sample_id] = {"real": real_path, "fake": []}

    for fake_path in fake_wavs:
        stem = _safe_name(fake_path)
        matched = next(
            (sample_id for sample_id in sorted_real_ids if stem == sample_id or stem.startswith(f"{sample_id}_")),
            None,
        )
        if matched is None:
            matched = stem.split("_", 1)[0]
            paired.setdefault(matched, {"real": None, "fake": []})
        paired[matched]["fake"].append(fake_path)

    return {
        sample_id: info
        for sample_id, info in paired.items()
        if info.get("real") is not None and info.get("fake")
    }


def _extract_seed_key(path: Path) -> str:
    match = re.search(r"(seed[_-]?\d+|temp[_-]?\d+(?:\.\d+)?)", path.stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else path.stem.lower()


def _limit_fake_variants(fake_paths: list[Path], seeds: int) -> list[Path]:
    selected: list[Path] = []
    used_keys: set[str] = set()
    for path in sorted(fake_paths):
        key = _extract_seed_key(path)
        if key in used_keys:
            continue
        selected.append(path)
        used_keys.add(key)
        if len(selected) >= seeds:
            break
    if len(selected) < min(seeds, len(fake_paths)):
        for path in sorted(fake_paths):
            if path not in selected:
                selected.append(path)
            if len(selected) >= min(seeds, len(fake_paths)):
                break
    return selected


def threshold_sweep(real_scores: list[float], fake_scores: list[float]) -> dict[str, Any]:
    if not real_scores or not fake_scores:
        raise ValueError("threshold_sweep에는 real/fake score가 모두 필요합니다.")

    all_scores = np.asarray([*real_scores, *fake_scores], dtype=np.float64)
    thresholds = sorted({-1e-6, 1.0 + 1e-6, *all_scores.tolist()}, reverse=True)

    curve: list[dict[str, float]] = []
    best_eer_gap = float("inf")
    best_threshold = thresholds[0]
    best_eer = 1.0
    fpr_at_95 = 1.0
    fpr_at_99 = 1.0

    real = np.asarray(real_scores, dtype=np.float64)
    fake = np.asarray(fake_scores, dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        fp = float(np.mean(real >= threshold))
        tp = float(np.mean(fake >= threshold))
        fnr = 1.0 - tp
        gap = abs(fp - fnr)
        curve.append({"threshold": threshold, "fpr": fp, "tpr": tp, "fnr": fnr})
        if gap < best_eer_gap:
            best_eer_gap = gap
            next_threshold = thresholds[index + 1] if index + 1 < len(thresholds) else threshold
            best_threshold = (threshold + next_threshold) / 2.0
            best_eer = (fp + fnr) / 2.0
        if tp >= 0.95:
            fpr_at_95 = min(fpr_at_95, fp)
        if tp >= 0.99:
            fpr_at_99 = min(fpr_at_99, fp)

    return {
        "eer": round(float(best_eer), 6),
        "recommended_threshold": round(float(best_threshold), 6),
        "fpr_at_tpr_95": round(float(fpr_at_95), 6),
        "fpr_at_tpr_99": round(float(fpr_at_99), 6),
        "roc": curve,
    }


def _entries_with_scores(entries: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            normalized.append(
                {
                    "id": str(entry.get("id", f"sample_{index:04d}")),
                    "score": float(entry["score"]),
                    "path": entry.get("path"),
                }
            )
        else:
            normalized.append({"id": f"sample_{index:04d}", "score": float(entry), "path": None})
    return normalized


def _summary_stats(entries: list[dict[str, Any]]) -> dict[str, float]:
    scores = [float(entry["score"]) for entry in entries]
    if not scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": round(float(statistics.fmean(scores)), 6),
        "median": round(float(statistics.median(scores)), 6),
        "std": round(float(statistics.stdev(scores)) if len(scores) > 1 else 0.0, 6),
        "count": len(scores),
    }


def slice_analysis(
    scores: dict[str, dict[str, list[Any]]],
    threshold: float,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition, grouped in scores.items():
        real_entries = _entries_with_scores(grouped.get("real", []))
        fake_entries = _entries_with_scores(grouped.get("fake", []))

        tp = [entry for entry in fake_entries if entry["score"] >= threshold]
        fn = [entry for entry in fake_entries if entry["score"] < threshold]
        fp = [entry for entry in real_entries if entry["score"] >= threshold]
        tn = [entry for entry in real_entries if entry["score"] < threshold]

        conditions[condition] = {
            "counts": {"tp": len(tp), "fp": len(fp), "tn": len(tn), "fn": len(fn)},
            "fake_stats": _summary_stats(fake_entries),
            "real_stats": _summary_stats(real_entries),
            "fake_detect_rate": round(len(tp) / len(fake_entries), 6) if fake_entries else 0.0,
            "real_false_positive_rate": round(len(fp) / len(real_entries), 6) if real_entries else 0.0,
            "false_positives": [entry["id"] for entry in fp],
            "false_negatives": [entry["id"] for entry in fn],
        }
    return {"threshold": threshold, "conditions": conditions}


def _maybe_plot_roc(curve: list[dict[str, float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("ROC 플롯에는 `matplotlib`가 필요합니다. `pip install matplotlib` 후 다시 시도하세요.") from exc

    xs = [point["fpr"] for point in curve]
    ys = [point["tpr"] for point in curve]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(xs, ys, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(out_path.stem)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_report(
    out_dir: Path,
    raw: dict[str, Any],
    threshold_metrics: dict[str, Any],
    slice_summary: dict[str, Any],
) -> Path:
    summary_rows: list[list[str]] = []
    breakdown_rows: list[list[str]] = []
    fp_fn_lines: list[str] = []

    for condition, details in slice_summary["conditions"].items():
        summary_rows.append(
            [
                condition,
                f"{details['fake_detect_rate']:.3f}",
                f"{details['real_false_positive_rate']:.3f}",
                str(details["counts"]["tp"]),
                str(details["counts"]["fp"]),
                str(details["counts"]["fn"]),
            ]
        )
        breakdown_rows.append(
            [
                condition,
                f"{details['fake_stats']['mean']:.3f}",
                f"{details['fake_stats']['median']:.3f}",
                f"{details['fake_stats']['std']:.3f}",
                f"{details['real_stats']['mean']:.3f}",
                f"{details['real_stats']['median']:.3f}",
                f"{details['real_stats']['std']:.3f}",
            ]
        )
        if details["false_positives"]:
            fp_fn_lines.append(f"- `{condition}` FP: {', '.join(details['false_positives'])}")
        if details["false_negatives"]:
            fp_fn_lines.append(f"- `{condition}` FN: {', '.join(details['false_negatives'])}")

    report = "\n\n".join(
        [
            "# Detector Stress Report",
            "## 요약",
            _markdown_table(
                ["condition", "clone fake 판정률", "real 오탐률", "TP", "FP", "FN"],
                summary_rows,
            ),
            "## Threshold Sweep (clean)",
            _markdown_table(
                ["metric", "value"],
                [
                    ["EER", f"{threshold_metrics['eer']:.6f}"],
                    ["FPR@0.95", f"{threshold_metrics['fpr_at_tpr_95']:.6f}"],
                    ["FPR@0.99", f"{threshold_metrics['fpr_at_tpr_99']:.6f}"],
                    ["recommended_threshold", f"{threshold_metrics['recommended_threshold']:.6f}"],
                ],
            ),
            "## 조건별 Breakdown",
            _markdown_table(
                ["condition", "fake_mean", "fake_median", "fake_std", "real_mean", "real_median", "real_std"],
                breakdown_rows,
            ),
            "## FP/FN Slice",
            "\n".join(fp_fn_lines) if fp_fn_lines else "- 없음",
        ]
    )

    report_path = out_dir / f"stress_{_timestamp()}.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    return report_path


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stress_test(
    clone_dir: Path,
    real_dir: Path,
    aasist_onnx: Path,
    out_dir: Path,
    conditions: list[str] = DEFAULT_CONDITIONS,
    noise_dir: Path | None = None,
    seeds: int = 5,
    paired: bool = True,
    plot: bool = False,
) -> Path:
    clone_dir = clone_dir.expanduser().resolve()
    real_dir = real_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    real_wavs = list_audio_files(real_dir, patterns=("*.wav",))
    fake_wavs = list_audio_files(clone_dir, patterns=("*.wav",))
    if not real_wavs:
        raise RuntimeError(f"real_dir에 WAV가 없습니다: {real_dir}")
    if not fake_wavs:
        raise RuntimeError(f"clone_dir에 WAV가 없습니다: {clone_dir}")

    if paired:
        pair_index = build_paired_index(real_wavs, fake_wavs)
        if not pair_index:
            raise RuntimeError("paired mode인데 real/fake를 같은 텍스트 id로 매칭하지 못했습니다.")
        selected_real: list[Path] = []
        selected_fake: list[Path] = []
        pair_metadata: dict[str, Any] = {}
        for sample_id, info in sorted(pair_index.items()):
            limited_fake = _limit_fake_variants(info["fake"], seeds)
            pair_metadata[sample_id] = {
                "real": str(info["real"]),
                "fake_variants": [str(path) for path in limited_fake],
            }
            selected_real.append(info["real"])
            selected_fake.extend(limited_fake)
    else:
        selected_real = real_wavs
        selected_fake = fake_wavs[: max(seeds, 1)]
        pair_metadata = {}

    generated_real = generate_stress_set(selected_real, out_dir / "generated" / "real", conditions, noise_dir=noise_dir)
    generated_fake = generate_stress_set(selected_fake, out_dir / "generated" / "fake", conditions, noise_dir=noise_dir)

    raw_scores: dict[str, dict[str, list[dict[str, Any]]]] = {}
    roc_metrics: dict[str, Any] = {}
    for condition in conditions:
        real_paths = generated_real[condition]
        fake_paths = generated_fake[condition]
        real_scores = score_aasist_batch(real_paths, aasist_onnx)
        fake_scores = score_aasist_batch(fake_paths, aasist_onnx)

        raw_scores[condition] = {
            "real": [
                {"id": path.stem, "path": str(path), "score": score}
                for path, score in zip(real_paths, real_scores, strict=True)
            ],
            "fake": [
                {"id": path.stem, "path": str(path), "score": score}
                for path, score in zip(fake_paths, fake_scores, strict=True)
            ],
        }
        roc_metrics[condition] = threshold_sweep(real_scores, fake_scores)
        if plot:
            _maybe_plot_roc(roc_metrics[condition]["roc"], out_dir / f"roc_{condition}.png")

    clean_condition = "clean" if "clean" in roc_metrics else conditions[0]
    clean_threshold = float(roc_metrics[clean_condition]["recommended_threshold"])
    slices = slice_analysis(raw_scores, threshold=clean_threshold)
    raw_payload = {
        "clone_dir": str(clone_dir),
        "real_dir": str(real_dir),
        "aasist_onnx": str(aasist_onnx),
        "conditions": conditions,
        "paired": paired,
        "seeds_requested": seeds,
        "pair_metadata": pair_metadata,
        "scores": raw_scores,
        "roc_metrics": roc_metrics,
        "slice_analysis": slices,
    }
    raw_json = out_dir / "raw.json"
    raw_json.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return _write_report(out_dir, raw_payload, roc_metrics[clean_condition], slices)


def _parse_conditions(value: str) -> list[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    if not conditions:
        raise argparse.ArgumentTypeError("최소 1개 condition이 필요합니다.")
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AASIST detector robustness stress test")
    parser.add_argument("--clone-dir", type=Path, required=True, help="합성 WAV 디렉토리")
    parser.add_argument("--real-dir", type=Path, required=True, help="원본 WAV 디렉토리")
    parser.add_argument("--aasist-onnx", type=Path, default=DEFAULT_AASIST_ONNX, help="AASIST ONNX 경로")
    parser.add_argument("--out", type=Path, required=True, help="리포트 출력 디렉토리")
    parser.add_argument("--conditions", type=_parse_conditions, default=DEFAULT_CONDITIONS, help="쉼표 구분 stress 조건")
    parser.add_argument("--noise-dir", type=Path, help="배경 소음 샘플 디렉토리")
    parser.add_argument("--seeds", type=int, default=5, help="샘플별 최대 fake variant 수")
    parser.add_argument("--no-paired", action="store_true", help="paired 매칭 없이 전체 set 사용")
    parser.add_argument("--plot", action="store_true", help="ROC PNG 저장")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = stress_test(
        clone_dir=args.clone_dir,
        real_dir=args.real_dir,
        aasist_onnx=args.aasist_onnx,
        out_dir=args.out,
        conditions=args.conditions,
        noise_dir=args.noise_dir,
        seeds=args.seeds,
        paired=not args.no_paired,
        plot=args.plot,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
