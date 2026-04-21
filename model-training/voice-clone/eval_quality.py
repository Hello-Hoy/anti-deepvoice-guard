#!/usr/bin/env python3
"""Objective quality evaluation for voice-clone outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import warnings
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone._audio_utils import AASIST_NUM_SAMPLES, SAMPLE_RATE_16K, list_audio_files, load_aasist_input, load_wav_16k, match_length
from model_training.voice_clone.engines._manifest import load_manifest, split_samples


DEFAULT_AASIST_ONNX = REPO_ROOT / "model-training" / "weights" / "finetuned" / "aasist_embedded.onnx"
DEFAULT_DNSMOS_MODEL = Path(os.environ.get("DNSMOS_MODEL", "~/tools/dnsmos/sig_bak_ovr.onnx")).expanduser()
DEFAULT_UTMOS_CMD = os.environ.get("UTMOS_PREDICT_CMD", "")
DEFAULT_METRICS = ["wer", "ecapa", "dnsmos", "utmos", "aasist"]
_WHISPER_MODELS: dict[str, Any] = {}
_ORT_SESSIONS: dict[Path, Any] = {}
_ECAPA: Any | None = None


@dataclass(frozen=True)
class MetricThreshold:
    label: str
    comparator: str
    threshold: float | None
    display: str

    def passed(self, value: float | None) -> bool | None:
        if value is None or self.threshold is None:
            return None
        if self.comparator == "lt":
            return value < self.threshold
        if self.comparator == "gt":
            return value > self.threshold
        return None


METRIC_THRESHOLDS = {
    "wer": MetricThreshold("WER", "lt", 0.10, "< 0.10"),
    "ecapa": MetricThreshold("ECAPA cosine", "gt", 0.75, "> 0.75"),
    "dnsmos": MetricThreshold("DNSMOS P.835 OVRL", "gt", 3.5, "> 3.5"),
    "utmos": MetricThreshold("UTMOS", "gt", 3.8, "> 3.8"),
    "aasist": MetricThreshold("AASIST fake score", "info", None, "informational"),
}


def normalized_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s가-힣]", " ", text or "", flags=re.UNICODE)
    return " ".join(cleaned.lower().split())


def _load_whisper_model(model_name: str) -> Any:
    model = _WHISPER_MODELS.get(model_name)
    if model is not None:
        return model
    try:
        import whisper
    except ModuleNotFoundError as exc:
        raise RuntimeError("openai-whisper is required for WER.") from exc
    model = whisper.load_model(model_name)
    _WHISPER_MODELS[model_name] = model
    return model


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        cur = [i]
        for j, other in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (token != other)))
        prev = cur
    return prev[-1]


def compute_wer(wav: Path, expected_text: str, whisper_model: str = "large-v3", language: str = "ko") -> dict[str, Any]:
    model = _load_whisper_model(whisper_model)
    result = model.transcribe(str(wav), language=language, task="transcribe", fp16=False)
    hypothesis = normalized_text(str(result.get("text", "")))
    reference = normalized_text(expected_text)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    wer = _levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)
    return {"wer": round(float(wer), 6), "reference": reference, "hypothesis": hypothesis}


def compute_ecapa_cosine(wav: Path, ref_wav: Path) -> float | None:
    global _ECAPA
    try:
        import torch
        import torch.nn.functional as F
        from speechbrain.inference.speaker import EncoderClassifier
    except ModuleNotFoundError:
        warnings.warn("speechbrain/torch unavailable; ECAPA skipped.", RuntimeWarning)
        return None
    if _ECAPA is None:
        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(REPO_ROOT / ".cache" / "speechbrain" / "ecapa"),
            run_opts={"device": "cpu"},
        )
    emb = _ECAPA.encode_batch(torch.from_numpy(load_wav_16k(wav)).unsqueeze(0))
    ref_emb = _ECAPA.encode_batch(torch.from_numpy(load_wav_16k(ref_wav)).unsqueeze(0))
    return round(float(F.cosine_similarity(emb, ref_emb, dim=-1).mean().item()), 6)


def _require_ort() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is required for ONNX metrics.") from exc
    return ort


def _ort_session(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    if resolved not in _ORT_SESSIONS:
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        _ORT_SESSIONS[resolved] = _require_ort().InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
    return _ORT_SESSIONS[resolved]


def _prepare_onnx_input(audio: np.ndarray, session: Any) -> dict[str, np.ndarray]:
    input_meta = session.get_inputs()[0]
    dims = len(input_meta.shape)
    tensor = audio.astype(np.float32)
    if dims == 2:
        tensor = tensor[np.newaxis, :]
    elif dims == 3:
        tensor = tensor[np.newaxis, :, np.newaxis]
    return {input_meta.name: tensor}


def compute_dnsmos(wav: Path, onnx_path: Path | None = None) -> dict[str, Any] | None:
    model_path = (onnx_path or DEFAULT_DNSMOS_MODEL).expanduser()
    if not model_path.exists():
        warnings.warn(f"DNSMOS model missing: {model_path}", RuntimeWarning)
        return None
    session = _ort_session(model_path)
    audio = load_wav_16k(wav)
    window = int(9.01 * SAMPLE_RATE_16K)
    if audio.size < window:
        audio = match_length(audio, window)
    outputs = session.run(None, _prepare_onnx_input(audio[:window], session))
    merged = np.concatenate([np.asarray(out, dtype=np.float32).reshape(-1) for out in outputs])
    if merged.size < 3:
        raise RuntimeError("DNSMOS output did not include SIG/BAK/OVRL values.")
    return {"sig": round(float(merged[0]), 6), "bak": round(float(merged[1]), 6), "ovrl": round(float(merged[2]), 6)}


def compute_utmos(wav: Path, command_template: str = DEFAULT_UTMOS_CMD) -> float | None:
    if not command_template:
        warnings.warn("UTMOS_PREDICT_CMD is not set; UTMOS skipped.", RuntimeWarning)
        return None
    import shlex

    command = [part.format(wav=str(wav)) for part in shlex.split(command_template)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise RuntimeError(f"UTMOS command failed: {' '.join(command)}\n{output}")
    match = re.search(r"([0-5](?:\.\d+)?)", output)
    return round(float(match.group(1)), 6) if match else None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_aasist_score(wav: Path, onnx_path: Path = DEFAULT_AASIST_ONNX) -> float:
    session = _ort_session(onnx_path)
    audio = load_aasist_input(wav, max_len=AASIST_NUM_SAMPLES)
    output = np.asarray(session.run(None, _prepare_onnx_input(audio, session))[0], dtype=np.float32).reshape(1, -1)
    if output.shape[1] < 2:
        raise RuntimeError(f"AASIST output shape is invalid: {output.shape}")
    probs = output if abs(float(np.sum(output[0])) - 1.0) < 1e-3 else _softmax(output)
    return round(float(probs[0][0]), 6)


def evaluate_one(
    wav: Path,
    expected_text: str,
    ref_wav: Path | None,
    metrics: list[str],
    aasist_onnx: Path,
    dnsmos_onnx: Path | None,
    whisper_model: str,
    language: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if "wer" in metrics:
        results["wer"] = compute_wer(wav, expected_text, whisper_model=whisper_model, language=language)
    if "ecapa" in metrics:
        if ref_wav is None:
            raise RuntimeError("ECAPA requires ref_wav.")
        results["ecapa"] = compute_ecapa_cosine(wav, ref_wav)
    if "dnsmos" in metrics:
        results["dnsmos"] = compute_dnsmos(wav, dnsmos_onnx)
    if "utmos" in metrics:
        results["utmos"] = compute_utmos(wav)
    if "aasist" in metrics:
        results["aasist"] = compute_aasist_score(wav, aasist_onnx)
    return {"wav": str(wav), "text": expected_text, "ref_wav": str(ref_wav) if ref_wav else None, "metrics": results, "verdict": verdict(results)}


def _comparable(name: str, value: Any) -> float | None:
    if name == "wer" and isinstance(value, dict):
        return float(value["wer"])
    if name == "dnsmos" and isinstance(value, dict):
        return float(value["ovrl"])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    required: list[bool] = []
    for name, threshold in METRIC_THRESHOLDS.items():
        value = _comparable(name, metrics.get(name))
        passed = threshold.passed(value)
        checks[name] = {"value": value, "threshold": threshold.display, "passed": passed}
        if passed is not None:
            required.append(passed)
    return {"overall_pass": bool(required) and all(required), "checks": checks}


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": round(float(statistics.fmean(values)), 6),
        "std": round(float(statistics.stdev(values)) if len(values) > 1 else 0.0, 6),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def evaluate_from_manifest(
    manifest_path: Path,
    clone_root: Path,
    engines: list[str],
    ref_modes: list[str],
    metrics: list[str],
    aasist_onnx: Path,
    dnsmos_onnx: Path | None,
    out_json: Path | None,
    report: Path | None,
    whisper_model: str = "large-v3",
    language: str = "ko",
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    samples = split_samples(manifest, "eval") + split_samples(manifest, "test")
    results: list[dict[str, Any]] = []
    for engine in engines:
        for ref_mode in ref_modes:
            clone_dir = clone_root / engine / ref_mode
            for sample in samples:
                wavs = sorted(clone_dir.rglob(f"{sample['id']}*.wav")) if clone_dir.exists() else []
                ref = Path(sample["wav_path_24k"])
                if not ref.is_absolute():
                    ref = manifest_path.parent / ref
                for wav in wavs:
                    payload = evaluate_one(wav, sample["text"], ref, metrics, aasist_onnx, dnsmos_onnx, whisper_model, language)
                    payload.update({"engine": engine, "ref_mode": ref_mode, "sample_id": sample["id"]})
                    results.append(payload)

    aggregate: dict[str, Any] = {}
    for engine in engines:
        aggregate[engine] = {}
        for ref_mode in ref_modes:
            subset = [item for item in results if item["engine"] == engine and item["ref_mode"] == ref_mode]
            metric_values = {name: [] for name in DEFAULT_METRICS}
            for item in subset:
                for name in metric_values:
                    value = _comparable(name, item["metrics"].get(name))
                    if value is not None:
                        metric_values[name].append(value)
            aggregate[engine][ref_mode] = {name: _stats(values) for name, values in metric_values.items()}

    payload = {"manifest": str(manifest_path), "clone_root": str(clone_root), "results": results, "aggregate": aggregate}
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report:
        write_markdown_report(report, payload)
    return payload


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Voice Clone Quality Report", "", "## Aggregate"]
    for engine, modes in payload["aggregate"].items():
        lines.append(f"### {engine}")
        lines.append("| ref_mode | WER | ECAPA | DNSMOS | UTMOS | AASIST |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for mode, metrics in modes.items():
            row = [mode]
            for name in ("wer", "ecapa", "dnsmos", "utmos", "aasist"):
                stats = metrics.get(name)
                row.append("" if not stats else f"{stats['mean']:.3f} (n={stats['count']})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="voice-clone objective quality evaluation")
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--ref-wav", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clone-root", "--clone-dir", dest="clone_root", type=Path)
    parser.add_argument("--engines", type=_parse_csv, default=["gpt_sovits", "openvoice2"])
    parser.add_argument("--ref-modes", type=_parse_csv, default=["rotate", "best"])
    parser.add_argument("--metrics", type=_parse_csv, default=DEFAULT_METRICS)
    parser.add_argument("--aasist-onnx", type=Path, default=DEFAULT_AASIST_ONNX)
    parser.add_argument("--dnsmos-onnx", type=Path, default=DEFAULT_DNSMOS_MODEL)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.manifest:
        if not args.clone_root:
            raise SystemExit("--manifest requires --clone-root/--clone-dir")
        payload = evaluate_from_manifest(
            args.manifest,
            args.clone_root,
            args.engines,
            args.ref_modes,
            args.metrics,
            args.aasist_onnx,
            args.dnsmos_onnx,
            args.out,
            args.report,
            args.whisper_model,
            args.language,
        )
        print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
        return 0
    if not args.wav or not args.text:
        raise SystemExit("Single-file mode requires --wav and --text.")
    payload = evaluate_one(args.wav, args.text, args.ref_wav, args.metrics, args.aasist_onnx, args.dnsmos_onnx, args.whisper_model, args.language)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
