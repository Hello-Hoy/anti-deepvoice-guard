#!/usr/bin/env python3
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

from model_training.voice_clone._audio_utils import (
    AASIST_NUM_SAMPLES,
    SAMPLE_RATE_16K,
    load_aasist_input,
    load_wav_16k,
    load_wav_generic,
    match_length,
)
from model_training.voice_clone.engines._manifest import load_manifest, split_samples
from model_training.voice_clone.scripts.normalize_korean import normalize as normalize_korean


DEFAULT_AASIST_ONNX = (
    REPO_ROOT / "model-training" / "weights" / "finetuned" / "aasist_embedded.onnx"
)
DEFAULT_DNSMOS_MODEL = Path(os.environ.get("DNSMOS_MODEL", "~/tools/dnsmos/sig_bak_ovr.onnx")).expanduser()
DEFAULT_UTMOSV2_DIR = Path(os.environ.get("UTMOSV2_DIR", "~/tools/UTMOSv2")).expanduser()
DEFAULT_METRICS = ["wer", "utmosv2", "dnsmos", "spk_sim", "aasist"]

_WHISPER_MODELS: dict[str, Any] = {}
_ORT_SESSIONS: dict[Path, Any] = {}
_SPEAKER_ENCODER: Any | None = None

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_CHOSUNG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
_JWUNGSUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
_JONGSUNG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


@dataclass(frozen=True)
class MetricThreshold:
    label: str
    direction: str
    comparator: str
    display: str
    threshold: float | None

    def evaluate(self, value: float | None) -> bool | None:
        if value is None or self.comparator == "informational" or self.threshold is None:
            return None
        if self.comparator == "lt":
            return value < self.threshold
        if self.comparator == "gt":
            return value > self.threshold
        raise ValueError(f"Unsupported comparator: {self.comparator}")


METRIC_THRESHOLDS: dict[str, MetricThreshold] = {
    "wer": MetricThreshold("WER", "lower", "lt", "< 0.10", 0.10),
    "utmosv2": MetricThreshold("UTMOSv2", "higher", "gt", "> 3.5", 3.5),
    "dnsmos": MetricThreshold("DNSMOS OVRL", "higher", "gt", "> 3.0", 3.0),
    "spk_sim": MetricThreshold("Speaker similarity", "higher", "gt", "> 0.75", 0.75),
    "aasist": MetricThreshold("AASIST fake score", "informational", "informational", "informational", None),
}


def dependency_message(package: str, command: str) -> str:
    return f"{package}가 필요합니다. 설치: `{command}`"


def normalized_text(text: str) -> str:
    cleaned = normalize_korean(text.strip()) if text else ""
    cleaned = re.sub(r"[^\w\s가-힣]", " ", cleaned, flags=re.UNICODE)
    return " ".join(cleaned.split())


def decompose_to_jamo(text: str) -> list[str]:
    output: list[str] = []
    for char in normalized_text(text).replace(" ", ""):
        code = ord(char)
        if _HANGUL_BASE <= code <= _HANGUL_END:
            syllable_index = code - _HANGUL_BASE
            chosung = syllable_index // 588
            jwungsung = (syllable_index % 588) // 28
            jongsung = syllable_index % 28
            output.extend([_CHOSUNG[chosung], _JWUNGSUNG[jwungsung]])
            tail = _JONGSUNG[jongsung]
            if tail:
                output.append(tail)
        else:
            output.append(char)
    return output


def levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, start=1):
        current = [row_index]
        for col_index, hyp_item in enumerate(hypothesis, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            substitute_cost = previous[col_index - 1] + (0 if ref_item == hyp_item else 1)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def compute_jamo_cer(reference: str, hypothesis: str) -> dict[str, float | int]:
    ref_jamo = decompose_to_jamo(reference)
    hyp_jamo = decompose_to_jamo(hypothesis)
    distance = levenshtein_distance(ref_jamo, hyp_jamo)
    denominator = max(len(ref_jamo), 1)
    cer = distance / denominator
    return {
        "cer": round(float(cer), 6),
        "distance": distance,
        "reference_jamo_length": len(ref_jamo),
        "hypothesis_jamo_length": len(hyp_jamo),
    }


def _load_whisper_model(model_name: str) -> Any:
    model = _WHISPER_MODELS.get(model_name)
    if model is not None:
        return model

    try:
        import whisper
    except ModuleNotFoundError as exc:
        raise RuntimeError(dependency_message("openai-whisper", "pip install openai-whisper")) from exc

    model = whisper.load_model(model_name)
    _WHISPER_MODELS[model_name] = model
    return model


def compute_wer(
    wav: Path,
    expected_text: str,
    whisper_model: str = "large-v3",
    language: str = "ko",
) -> dict[str, Any]:
    try:
        import jiwer
    except ModuleNotFoundError as exc:
        raise RuntimeError(dependency_message("jiwer", "pip install jiwer")) from exc

    model = _load_whisper_model(whisper_model)
    result = model.transcribe(str(wav), language=language, task="transcribe", fp16=False)
    hypothesis = normalized_text(str(result.get("text", "")))
    reference = normalized_text(expected_text)
    wer_value = float(jiwer.wer(reference, hypothesis))
    jamo = compute_jamo_cer(reference, hypothesis)
    return {
        "wer": round(wer_value, 6),
        "hypothesis": hypothesis,
        "reference": reference,
        "jamo_cer": jamo["cer"],
        "jamo_details": jamo,
    }


def _warn_metric_unavailable(metric: str, message: str) -> None:
    warnings.warn(f"[{metric}] {message}", RuntimeWarning, stacklevel=2)


def compute_utmosv2(wav: Path, utmosv2_dir: Path | None = None) -> float | None:
    repo_dir = (utmosv2_dir or DEFAULT_UTMOSV2_DIR).expanduser()
    if not repo_dir.exists():
        _warn_metric_unavailable(
            "utmosv2",
            f"UTMOSv2 repo를 찾지 못했습니다: {repo_dir}. "
            "환경변수 `UTMOSV2_DIR` 또는 `--utmosv2-dir`를 지정하세요.",
        )
        return None

    python_bin = repo_dir / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    env_cmd = os.environ.get("UTMOSV2_PREDICT_CMD")
    if env_cmd:
        import shlex

        command = [part.format(wav=str(wav), repo=str(repo_dir)) for part in shlex.split(env_cmd)]
    else:
        script_candidates = [
            repo_dir / "predict.py",
            repo_dir / "bin" / "predict.py",
            repo_dir / "inference.py",
            repo_dir / "demo.py",
        ]
        script = next((candidate for candidate in script_candidates if candidate.exists()), None)
        if script is None:
            _warn_metric_unavailable(
                "utmosv2",
                "UTMOSv2 실행 스크립트를 찾지 못했습니다. "
                "repo 루트에 `predict.py`가 있는지 확인하거나 `UTMOSV2_PREDICT_CMD`를 설정하세요.",
            )
            return None
        command = [str(python_bin), str(script), str(wav)]

    result = subprocess.run(
        command,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"UTMOSv2 실행 실패 (exit={result.returncode}).\n"
            f"command={' '.join(command)}\n{output or '(출력 없음)'}"
        )

    match = re.search(r"([0-5](?:\.\d+)?)", output)
    if not match:
        raise RuntimeError(
            "UTMOSv2 출력에서 MOS 점수를 파싱하지 못했습니다. "
            "필요하면 `UTMOSV2_PREDICT_CMD`로 사용자 정의 실행 명령을 지정하세요.\n"
            f"raw output:\n{output or '(출력 없음)'}"
        )
    return round(float(match.group(1)), 6)


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError(dependency_message("onnxruntime", "pip install onnxruntime")) from exc
    return ort


def _get_ort_session(model_path: Path) -> Any:
    ort = _require_onnxruntime()
    resolved = model_path.expanduser().resolve()
    session = _ORT_SESSIONS.get(resolved)
    if session is None:
        if not resolved.exists():
            raise FileNotFoundError(f"ONNX 모델을 찾지 못했습니다: {resolved}")
        session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
        _ORT_SESSIONS[resolved] = session
    return session


def _prepare_onnx_input(audio: np.ndarray, session: Any) -> dict[str, np.ndarray]:
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    dims = len(input_meta.shape)
    if dims == 1:
        tensor = audio.astype(np.float32)
    elif dims == 2:
        tensor = audio[np.newaxis, :].astype(np.float32)
    elif dims == 3:
        tensor = audio[np.newaxis, :, np.newaxis].astype(np.float32)
    else:
        raise RuntimeError(f"지원하지 않는 ONNX 입력 shape: {input_meta.shape}")
    return {input_name: tensor}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_dnsmos(wav: Path, onnx_path: Path | None = None) -> dict[str, Any] | None:
    model_path = (onnx_path or DEFAULT_DNSMOS_MODEL).expanduser()
    if not model_path.exists():
        _warn_metric_unavailable(
            "dnsmos",
            f"DNSMOS 모델을 찾지 못했습니다: {model_path}. "
            "환경변수 `DNSMOS_MODEL` 또는 `--dnsmos-onnx`를 지정하세요.",
        )
        return None

    session = _get_ort_session(model_path)
    wav_16k = load_wav_16k(wav)
    window = int(9.01 * SAMPLE_RATE_16K)
    hop = SAMPLE_RATE_16K
    if wav_16k.size < window:
        wav_16k = match_length(wav_16k, window)

    windows: list[np.ndarray] = []
    for start in range(0, max(wav_16k.size - window + 1, 1), hop):
        segment = wav_16k[start:start + window]
        if segment.size < window:
            segment = match_length(segment, window)
        windows.append(segment.astype(np.float32))
    if not windows:
        windows = [match_length(wav_16k, window)]

    scores: list[np.ndarray] = []
    for segment in windows:
        outputs = session.run(None, _prepare_onnx_input(segment, session))
        values = [np.asarray(output, dtype=np.float32).reshape(-1) for output in outputs]
        merged = np.concatenate(values)
        if merged.size < 3:
            raise RuntimeError(
                f"DNSMOS 출력 형식이 예상과 다릅니다. outputs={session.get_outputs()}"
            )
        scores.append(merged[:4])

    averaged = np.mean(np.vstack(scores), axis=0)
    response = {
        "sig": round(float(averaged[0]), 6),
        "bak": round(float(averaged[1]), 6),
        "ovrl": round(float(averaged[2]), 6),
        "model_path": str(model_path),
        "num_windows": len(windows),
    }
    if averaged.size >= 4:
        response["p808"] = round(float(averaged[3]), 6)
    return response


def compute_speaker_similarity(wav: Path, ref_wav: Path) -> float | None:
    global _SPEAKER_ENCODER

    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise RuntimeError(dependency_message("torch", "pip install torch")) from exc

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ModuleNotFoundError:
        _warn_metric_unavailable(
            "spk_sim",
            dependency_message("speechbrain", "pip install speechbrain"),
        )
        return None

    if _SPEAKER_ENCODER is None:
        _SPEAKER_ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(REPO_ROOT / ".cache" / "speechbrain" / "ecapa"),
            run_opts={"device": "cpu"},
        )

    wav_audio = torch.from_numpy(load_wav_16k(wav)).unsqueeze(0)
    ref_audio = torch.from_numpy(load_wav_16k(ref_wav)).unsqueeze(0)
    wav_embedding = _SPEAKER_ENCODER.encode_batch(wav_audio)
    ref_embedding = _SPEAKER_ENCODER.encode_batch(ref_audio)
    similarity = F.cosine_similarity(wav_embedding, ref_embedding, dim=-1).mean().item()
    return round(float(similarity), 6)


def compute_aasist_score(wav: Path, onnx_path: Path) -> float:
    audio = load_aasist_input(wav, max_len=AASIST_NUM_SAMPLES)
    session = _get_ort_session(onnx_path)
    outputs = session.run(None, _prepare_onnx_input(audio, session))
    if not outputs:
        raise RuntimeError("AASIST ONNX 추론 결과가 비어 있습니다.")

    output = np.asarray(outputs[0], dtype=np.float32)
    flattened = output.reshape(1, -1)
    if flattened.shape[1] < 2:
        raise RuntimeError(f"AASIST 출력 shape이 잘못되었습니다: {output.shape}")

    probs = flattened
    row_sum = float(np.sum(probs[0]))
    if not math.isfinite(row_sum) or abs(row_sum - 1.0) > 1e-3:
        probs = _softmax(flattened)
    return round(float(probs[0][0]), 6)


def summarize_metric_verdicts(metrics: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    thresholded_passes: list[bool] = []
    thresholded_available = True

    for name, threshold in METRIC_THRESHOLDS.items():
        raw_value = metrics.get(name)
        if name == "dnsmos" and isinstance(raw_value, dict):
            comparable = raw_value.get("ovrl")
        elif name == "wer" and isinstance(raw_value, dict):
            comparable = raw_value.get("wer")
        elif isinstance(raw_value, (int, float)):
            comparable = raw_value
        else:
            comparable = None

        passed = threshold.evaluate(float(comparable) if comparable is not None else None)
        checks[name] = {
            "label": threshold.label,
            "threshold": threshold.display,
            "direction": threshold.direction,
            "value": comparable,
            "passed": passed,
            "available": raw_value is not None,
        }

        if threshold.comparator != "informational":
            if comparable is None:
                thresholded_available = False
            else:
                thresholded_passes.append(bool(passed))

    overall_pass = bool(thresholded_passes) and thresholded_available and all(thresholded_passes)
    return {
        "overall_pass": overall_pass,
        "all_required_available": thresholded_available,
        "checks": checks,
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(
    wav: Path,
    expected_text: str,
    ref_wav: Path | None = None,
    aasist_onnx: Path | None = None,
    metrics: list[str] = DEFAULT_METRICS,
    out_json: Path | None = None,
    utmosv2_dir: Path | None = None,
    dnsmos_onnx: Path | None = None,
    whisper_model: str = "large-v3",
    language: str = "ko",
) -> dict[str, Any]:
    wav = wav.expanduser().resolve()
    selected = [metric.strip() for metric in metrics if metric.strip()]
    results: dict[str, Any] = {}

    if "wer" in selected:
        results["wer"] = compute_wer(wav, expected_text, whisper_model=whisper_model, language=language)
    if "utmosv2" in selected:
        results["utmosv2"] = compute_utmosv2(wav, utmosv2_dir=utmosv2_dir)
    if "dnsmos" in selected:
        results["dnsmos"] = compute_dnsmos(wav, onnx_path=dnsmos_onnx)
    if "spk_sim" in selected:
        if ref_wav is None:
            raise RuntimeError("`spk_sim` 계산에는 `--ref-wav`가 필요합니다.")
        results["spk_sim"] = compute_speaker_similarity(wav, ref_wav.expanduser().resolve())
    if "aasist" in selected:
        model_path = (aasist_onnx or DEFAULT_AASIST_ONNX).expanduser()
        results["aasist"] = compute_aasist_score(wav, model_path)

    payload = {
        "wav": str(wav),
        "text": expected_text,
        "ref_wav": str(ref_wav.expanduser().resolve()) if ref_wav else None,
        "metrics": results,
        "verdict": summarize_metric_verdicts(results),
    }
    if out_json is not None:
        _save_json(out_json, payload)
    return payload


def _resolve_manifest_audio(manifest_path: Path, sample: dict[str, Any], key: str) -> Path:
    candidate = Path(sample[key])
    if candidate.is_absolute():
        return candidate
    return manifest_path.parent / candidate


def _resolve_clone_matches(clone_dir: Path, sample_id: str) -> list[Path]:
    direct = sorted(clone_dir.rglob(f"{sample_id}.wav"))
    prefixed = sorted(clone_dir.rglob(f"{sample_id}_*.wav"))
    return list(dict.fromkeys([*direct, *prefixed]))


def _numeric_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": round(float(mean), 6),
        "std": round(float(std), 6),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def evaluate_from_manifest(
    manifest_path: Path,
    clone_dir: Path,
    ref_wav: Path | None = None,
    aasist_onnx: Path | None = None,
    metrics: list[str] = DEFAULT_METRICS,
    out_json: Path | None = None,
    utmosv2_dir: Path | None = None,
    dnsmos_onnx: Path | None = None,
    whisper_model: str = "large-v3",
    language: str = "ko",
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    clone_dir = clone_dir.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    eval_samples = split_samples(manifest, "eval")

    results: list[dict[str, Any]] = []
    missing_clones: list[str] = []
    for sample in eval_samples:
        clone_matches = _resolve_clone_matches(clone_dir, sample["id"])
        if not clone_matches:
            missing_clones.append(sample["id"])
            continue

        sample_ref = ref_wav or _resolve_manifest_audio(manifest_path, sample, "wav_path_24k")
        for clone_wav in clone_matches:
            payload = evaluate(
                wav=clone_wav,
                expected_text=sample["text"],
                ref_wav=sample_ref if "spk_sim" in metrics else None,
                aasist_onnx=aasist_onnx,
                metrics=metrics,
                utmosv2_dir=utmosv2_dir,
                dnsmos_onnx=dnsmos_onnx,
                whisper_model=whisper_model,
                language=language,
            )
            payload["sample_id"] = sample["id"]
            results.append(payload)

    aggregate_inputs: dict[str, list[float]] = {
        "wer": [],
        "utmosv2": [],
        "dnsmos": [],
        "spk_sim": [],
        "aasist": [],
    }
    pass_counter = 0
    for item in results:
        if item["verdict"]["overall_pass"]:
            pass_counter += 1
        metrics_payload = item["metrics"]
        if isinstance(metrics_payload.get("wer"), dict):
            aggregate_inputs["wer"].append(float(metrics_payload["wer"]["wer"]))
        if isinstance(metrics_payload.get("utmosv2"), (int, float)):
            aggregate_inputs["utmosv2"].append(float(metrics_payload["utmosv2"]))
        if isinstance(metrics_payload.get("dnsmos"), dict) and metrics_payload["dnsmos"].get("ovrl") is not None:
            aggregate_inputs["dnsmos"].append(float(metrics_payload["dnsmos"]["ovrl"]))
        if isinstance(metrics_payload.get("spk_sim"), (int, float)):
            aggregate_inputs["spk_sim"].append(float(metrics_payload["spk_sim"]))
        if isinstance(metrics_payload.get("aasist"), (int, float)):
            aggregate_inputs["aasist"].append(float(metrics_payload["aasist"]))

    aggregate = {
        "samples_evaluated": len(results),
        "missing_clones": missing_clones,
        "pass_rate": round(pass_counter / len(results), 6) if results else 0.0,
        "metrics": {name: _numeric_stats(values) for name, values in aggregate_inputs.items()},
    }
    payload = {
        "manifest": str(manifest_path),
        "clone_dir": str(clone_dir),
        "metrics_requested": metrics,
        "results": results,
        "aggregate": aggregate,
    }
    if out_json is not None:
        _save_json(out_json, payload)
    return payload


def _print_human_summary(payload: dict[str, Any]) -> None:
    if "aggregate" in payload:
        aggregate = payload["aggregate"]
        print(f"evaluated={aggregate['samples_evaluated']} pass_rate={aggregate['pass_rate']:.3f}")
        if aggregate["missing_clones"]:
            print(f"missing_clones={','.join(aggregate['missing_clones'])}")
        for metric, stats in aggregate["metrics"].items():
            if not stats:
                continue
            print(f"{metric}: mean={stats['mean']:.4f} std={stats['std']:.4f} n={stats['count']}")
        return

    verdict = payload["verdict"]
    print(f"wav={payload['wav']}")
    print(f"overall_pass={verdict['overall_pass']} required_available={verdict['all_required_available']}")
    for name, info in verdict["checks"].items():
        if info["value"] is None:
            continue
        print(f"{name}: value={info['value']} threshold={info['threshold']} passed={info['passed']}")


def _parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(metrics) - set(DEFAULT_METRICS))
    if unknown:
        raise argparse.ArgumentTypeError(f"지원하지 않는 metric: {', '.join(unknown)}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="voice-clone 객관 품질 평가")
    parser.add_argument("--wav", type=Path, help="평가할 합성 WAV")
    parser.add_argument("--text", help="ground-truth 텍스트")
    parser.add_argument("--ref-wav", type=Path, help="화자 유사도 계산용 기준 화자 WAV")
    parser.add_argument("--manifest", type=Path, help="manifest 기반 일괄 평가 입력")
    parser.add_argument("--clone-dir", type=Path, help="manifest 기반 합성 결과 디렉토리")
    parser.add_argument("--aasist-onnx", type=Path, default=DEFAULT_AASIST_ONNX, help="AASIST ONNX 경로")
    parser.add_argument("--dnsmos-onnx", type=Path, default=DEFAULT_DNSMOS_MODEL, help="DNSMOS ONNX 경로")
    parser.add_argument("--utmosv2-dir", type=Path, default=DEFAULT_UTMOSV2_DIR, help="UTMOSv2 repo 경로")
    parser.add_argument("--metrics", type=_parse_metrics, default=DEFAULT_METRICS, help="쉼표 구분 metric 목록")
    parser.add_argument("--whisper-model", default="large-v3", help="openai-whisper 모델명")
    parser.add_argument("--language", default="ko", help="Whisper 언어 코드")
    parser.add_argument("--out", type=Path, help="JSON 리포트 저장 경로")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.manifest:
        if not args.clone_dir:
            parser.error("--manifest 사용 시 --clone-dir 도 필요합니다.")
        payload = evaluate_from_manifest(
            manifest_path=args.manifest,
            clone_dir=args.clone_dir,
            ref_wav=args.ref_wav,
            aasist_onnx=args.aasist_onnx,
            metrics=args.metrics,
            out_json=args.out,
            utmosv2_dir=args.utmosv2_dir,
            dnsmos_onnx=args.dnsmos_onnx,
            whisper_model=args.whisper_model,
            language=args.language,
        )
        _print_human_summary(payload)
        return 0

    if not args.wav or not args.text:
        parser.error("단일 파일 평가는 --wav 와 --text 가 필요합니다.")

    payload = evaluate(
        wav=args.wav,
        expected_text=args.text,
        ref_wav=args.ref_wav,
        aasist_onnx=args.aasist_onnx,
        metrics=args.metrics,
        out_json=args.out,
        utmosv2_dir=args.utmosv2_dir,
        dnsmos_onnx=args.dnsmos_onnx,
        whisper_model=args.whisper_model,
        language=args.language,
    )
    _print_human_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
