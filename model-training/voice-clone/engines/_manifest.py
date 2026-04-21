from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import warnings


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "manifest_schema.json"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_manifest(data)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid manifest at {path}:\n{joined}")
    return data


def save_manifest(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def get_sample_by_id(manifest: dict[str, Any], sample_id: str) -> dict[str, Any] | None:
    for sample in manifest.get("samples", []):
        if sample.get("id") == sample_id:
            return sample
    return None


def split_samples(manifest: dict[str, Any], which: str = "train") -> list[dict[str, Any]]:
    sample_map = {sample.get("id"): sample for sample in manifest.get("samples", [])}
    return [
        sample_map[sample_id]
        for sample_id in manifest.get("split", {}).get(which, [])
        if sample_id in sample_map
    ]


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_schema_validate(data))
    errors.extend(_manual_validate(data))
    return errors


def _schema_validate(data: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        warnings.warn(
            "jsonschema is not installed; falling back to manual manifest validation.",
            RuntimeWarning,
        )
        return []

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return [
        _format_schema_error(error.path, error.message)
        for error in sorted(validator.iter_errors(data), key=lambda item: list(item.path))
    ]


def _manual_validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["Manifest must be a JSON object."]

    if not isinstance(data.get("speaker"), str) or not data.get("speaker", "").strip():
        errors.append("speaker must be a non-empty string.")

    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("samples must be a non-empty list.")
        samples = []

    sample_ids: list[str] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"samples[{index}] must be an object.")
            continue
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not _ID_PATTERN.fullmatch(sample_id):
            errors.append(
                f"samples[{index}].id must match ^[A-Za-z0-9_]+$, got {sample_id!r}."
            )
        else:
            sample_ids.append(sample_id)

        for key in ("wav_path_24k", "wav_path_16k", "text"):
            value = sample.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"samples[{index}].{key} must be a non-empty string.")

        duration = sample.get("duration_sec")
        if not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"samples[{index}].duration_sec must be > 0.")

        verified = sample.get("verified")
        if not isinstance(verified, bool):
            errors.append(f"samples[{index}].verified must be a boolean.")

        whisper_match = sample.get("whisper_match")
        if whisper_match is not None and (
            not isinstance(whisper_match, (int, float)) or not (0.0 <= whisper_match <= 1.0)
        ):
            errors.append(f"samples[{index}].whisper_match must be between 0.0 and 1.0.")

    duplicate_ids = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    for sample_id in duplicate_ids:
        errors.append(f"Duplicate sample id: {sample_id}")

    split = data.get("split")
    if not isinstance(split, dict):
        errors.append("split must be an object with train/eval lists.")
        return errors

    for key in ("train", "eval"):
        split_ids = split.get(key)
        if not isinstance(split_ids, list):
            errors.append(f"split.{key} must be a list.")
            continue
        for split_index, sample_id in enumerate(split_ids):
            if not isinstance(sample_id, str):
                errors.append(f"split.{key}[{split_index}] must be a string.")
                continue
            if sample_id not in sample_ids:
                errors.append(f"split.{key}[{split_index}] references unknown sample id '{sample_id}'.")

    return errors


def _format_schema_error(path: Any, message: str) -> str:
    location = "root"
    if path:
        location = ".".join(str(part) for part in path)
    return f"{location}: {message}"
