from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence
import warnings
import wave

import numpy as np

from . import register
from ._manifest import load_manifest, split_samples
from .base import VoiceCloneEngine
from model_training.voice_clone.scripts.normalize_korean import normalize


os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_DIRNAME = "tts_models--multilingual--multi-dataset--xtts_v2"
XTTS_DOWNLOAD_BASE = "https://huggingface.co/coqui/XTTS-v2/resolve/main"
XTTS_REQUIRED_FILES = {
    "model.pth": "xtts_checkpoint",
    "config.json": "config",
    "vocab.json": "tokenizer_file",
    "speakers_xtts.pth": "speaker_file",
    "dvae.pth": "dvae_checkpoint",
    "mel_stats.pth": "mel_norm_file",
}

_TORCH_LOAD_PATCHED = False


def _ensure_torch_load_patch():
    global _TORCH_LOAD_PATCHED
    import torch

    if not _TORCH_LOAD_PATCHED:
        original_load = torch.load
        torch.load = lambda *a, **k: original_load(*a, **{**k, "weights_only": False})
        _TORCH_LOAD_PATCHED = True
    return torch


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _resolve_audio_path(manifest_path: Path, sample: dict[str, Any], key: str = "wav_path_24k") -> Path:
    raw_value = sample.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Sample '{sample.get('id', '<unknown>')}' is missing '{key}'.")
    path = Path(raw_value)
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file referenced by manifest does not exist: {path}")
    return path


def _convert_audio_or_raise(src: Path, dst: Path, sample_rate: int, channels: int = 1) -> None:
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
            str(sample_rate),
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
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg 오디오 변환 실패: "
            f"{src} -> {dst} ({sample_rate}Hz)\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}"
        )


def _probe_xtts_runtime() -> tuple[bool, dict[str, Any]]:
    payload: dict[str, Any] = {"tos_agreed": os.getenv("COQUI_TOS_AGREED") == "1"}
    try:
        torch = _ensure_torch_load_patch()
        if torch.cuda.is_available():
            payload["device"] = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            payload["device"] = "mps"
        else:
            payload["device"] = "cpu"
    except Exception as exc:
        payload["device"] = "cpu"
        payload["torch_error"] = repr(exc)

    try:
        from TTS.api import TTS  # noqa: F401

        payload["tts_import"] = True
    except Exception as exc:
        payload["tts_import"] = False
        payload["tts_error"] = repr(exc)

    return bool(payload.get("tts_import")), payload


def _xtts_download_url(filename: str) -> str:
    return f"{XTTS_DOWNLOAD_BASE}/{filename}"


def _iter_xtts_search_roots(search_roots: Sequence[Path | str] | None = None) -> list[Path]:
    roots: list[Path] = []
    if search_roots is not None:
        roots.extend(Path(raw).expanduser() for raw in search_roots)
    else:
        roots.extend(
            [
                Path("~/Library/Application Support/tts").expanduser(),
                Path("~/.local/share/tts").expanduser(),
            ]
        )
        for raw in os.getenv("COQUI_MODELS", "").split(os.pathsep):
            if raw.strip():
                roots.append(Path(raw.strip()).expanduser())

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(root.resolve())
    return unique_roots


def _find_xtts_pretrained_root(search_roots: Sequence[Path | str] | None = None) -> Path:
    searched: list[Path] = []
    for root in _iter_xtts_search_roots(search_roots):
        candidates = [root] if root.name == XTTS_DIRNAME else [root / XTTS_DIRNAME]
        for candidate in candidates:
            searched.append(candidate)
            if candidate.is_dir():
                return candidate
    looked_in = "\n".join(f"  - {path}" for path in searched) or "  - (none)"
    hints = "\n".join(f"  - {name}: {_xtts_download_url(name)}" for name in XTTS_REQUIRED_FILES)
    raise FileNotFoundError(
        "XTTS pretrained bundle 디렉토리를 찾지 못했습니다.\n"
        f"찾아본 경로:\n{looked_in}\n"
        "필수 파일 다운로드 URL:\n"
        f"{hints}"
    )


def _resolve_xtts_pretrained_bundle(search_roots: Sequence[Path | str] | None = None) -> dict[str, Path]:
    root = _find_xtts_pretrained_root(search_roots)
    artifacts = {"root": root}
    missing: list[str] = []
    for filename, key in XTTS_REQUIRED_FILES.items():
        path = root / filename
        if path.exists():
            artifacts[key] = path
        else:
            missing.append(filename)

    if missing:
        hints = "\n".join(f"  - {name}: {_xtts_download_url(name)}" for name in missing)
        raise FileNotFoundError(
            "XTTS pretrained bundle이 불완전합니다.\n"
            f"디렉토리: {root}\n"
            f"누락 파일: {', '.join(missing)}\n"
            "다운로드 URL:\n"
            f"{hints}"
        )
    return artifacts


def _validate_xtts_training_data_dir(data_dir: Path) -> None:
    required_paths = [
        data_dir / "metadata_train.csv",
        data_dir / "metadata_eval.csv",
        data_dir / "wavs",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if not missing:
        return
    raise FileNotFoundError(
        "XTTS 학습 데이터 디렉토리가 불완전합니다.\n"
        f"data_dir: {data_dir}\n"
        "누락 경로:\n"
        + "\n".join(f"  - {path}" for path in missing)
    )


def _select_training_runtime(batch_size: int, mixed_precision: bool) -> dict[str, Any]:
    torch = _ensure_torch_load_patch()

    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "batch_size": batch_size,
            "mixed_precision": mixed_precision,
            "note": "CUDA 사용",
        }

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        warnings.warn(
            "XTTS fine-tuning on MPS is experimental; mixed_precision=False and batch_size=2 forced.",
            RuntimeWarning,
        )
        return {
            "device": "mps",
            "batch_size": 2,
            "mixed_precision": False,
            "note": "MPS 사용 (실험적, mixed_precision=False, batch_size=2 강제)",
        }

    warnings.warn(
        "XTTS fine-tuning on CPU is extremely slow; 반나절~1일 소요 예상.",
        RuntimeWarning,
    )
    return {
        "device": "cpu",
        "batch_size": batch_size,
        "mixed_precision": False,
        "note": "CPU 사용 (반나절~1일 소요 예상)",
    }


def _copy_xtts_sidecar_assets(run_dir: Path, pretrained: dict[str, Path]) -> None:
    for key in ("tokenizer_file", "speaker_file"):
        src = pretrained.get(key)
        if src is None:
            continue
        _copy_or_link(src, run_dir / src.name)


def _find_checkpoint_file(root: Path) -> Path:
    best_model = root / "best_model.pth"
    if best_model.exists():
        return best_model

    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
    )
    if not candidates:
        raise RuntimeError(f"XTTS 학습은 끝났지만 checkpoint를 찾지 못했습니다: {root}")
    return candidates[0]


def _resolve_xtts_artifacts(ckpt: Path) -> dict[str, Path]:
    ckpt_path = ckpt.expanduser().resolve()
    if ckpt_path.is_dir():
        model_path = _find_checkpoint_file(ckpt_path)
    else:
        model_path = ckpt_path

    search_dirs = [model_path.parent]
    if model_path.parent.parent != model_path.parent:
        search_dirs.append(model_path.parent.parent)

    def _first_existing(filename: str) -> Path | None:
        for directory in search_dirs:
            candidate = directory / filename
            if candidate.exists():
                return candidate
        return None

    config_path = _first_existing("config.json")
    vocab_path = _first_existing("vocab.json")
    speaker_path = _first_existing("speakers_xtts.pth")

    missing = [
        filename
        for filename, path in {
            "config.json": config_path,
            "vocab.json": vocab_path,
            "speakers_xtts.pth": speaker_path,
        }.items()
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            "XTTS 추론에 필요한 sidecar 파일을 찾지 못했습니다.\n"
            f"checkpoint: {model_path}\n"
            f"누락 파일: {', '.join(missing)}\n"
            "학습 산출물 디렉토리 또는 pretrained bundle의 파일이 checkpoint 옆에 있어야 합니다."
        )

    return {
        "model_path": model_path,
        "config_path": config_path,
        "vocab_path": vocab_path,
        "speaker_file_path": speaker_path,
    }


def _write_wave_pcm16(path: Path, samples: Any, sample_rate: int) -> None:
    array = np.asarray(samples, dtype=np.float32)
    clipped = np.clip(array, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


class _EpochLossMonitor:
    def __init__(self, patience: int) -> None:
        self.patience = max(0, patience)
        self.best_loss = float("inf")
        self.bad_epochs = 0

    @staticmethod
    def _get_avg_loss(trainer: Any, keep_avg_attr: str) -> float | None:
        keep_avg = getattr(trainer, keep_avg_attr, None)
        if keep_avg is None:
            return None
        if hasattr(trainer, "_pick_target_avg_loss"):
            value = trainer._pick_target_avg_loss(keep_avg)
            if isinstance(value, (int, float)):
                return float(value)
        avg_values = getattr(keep_avg, "avg_values", None) or {}
        for key in ("avg_loss", "loss"):
            raw = avg_values.get(key)
            if isinstance(raw, (int, float)):
                return float(raw)
        return None

    def on_epoch_end(self, trainer: Any) -> None:
        epoch = int(getattr(trainer, "epochs_done", 0))
        train_loss = self._get_avg_loss(trainer, "keep_avg_train")
        eval_loss = self._get_avg_loss(trainer, "keep_avg_eval")
        train_text = f"{train_loss:.6f}" if train_loss is not None else "n/a"
        eval_text = f"{eval_loss:.6f}" if eval_loss is not None else "n/a"
        print(f"[xtts] epoch={epoch} train_avg_loss={train_text} eval_avg_loss={eval_text}")

        target_loss = eval_loss if eval_loss is not None else train_loss
        if self.patience <= 0 or target_loss is None:
            return
        if target_loss < self.best_loss:
            self.best_loss = target_loss
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print(f"[xtts] early stopping triggered after {self.bad_epochs} stale epochs")
            setattr(trainer, "should_stop", True)


def build_xtts_training_session(
    data_dir: Path,
    ckpt_out: Path,
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 5e-6,
    early_stop_patience: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    ok, message = XttsEngine().health_check()
    if not ok:
        raise RuntimeError(f"XTTS 학습 준비 실패: {message}")

    _ensure_torch_load_patch()

    try:
        from TTS.tts.layers.xtts.trainer.gpt_trainer import (
            GPTArgs,
            GPTTrainer,
            GPTTrainerConfig,
            XttsAudioConfig,
        )
        from TTS.tts.configs.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        import trainer.analytics as trainer_analytics
        import trainer.trainer as trainer_module
        from trainer import Trainer, TrainerArgs
    except ImportError as exc:
        raise RuntimeError(
            "Coqui XTTS GPTTrainer import 실패. "
            "model-training/requirements-cloning.txt 의 pin 버전을 설치하세요. "
            f"원인: {exc}"
        ) from exc

    trainer_analytics.ping_training_run = lambda: None
    trainer_module.ping_training_run = lambda: None

    data_dir = data_dir.expanduser().resolve()
    ckpt_out = ckpt_out.expanduser().resolve()
    ckpt_out.mkdir(parents=True, exist_ok=True)
    _validate_xtts_training_data_dir(data_dir)

    pretrained = _resolve_xtts_pretrained_bundle(kwargs.get("pretrained_search_roots"))
    runtime = _select_training_runtime(
        batch_size=batch_size,
        mixed_precision=bool(kwargs.get("mixed_precision", True)),
    )
    print(f"[xtts] runtime={runtime['device']} ({runtime['note']})")

    run_name = str(kwargs.get("run_name") or f"xtts_{ckpt_out.name}")
    num_loader_workers = int(kwargs.get("num_loader_workers", 0 if runtime["device"] == "cpu" else 2))
    num_eval_loader_workers = int(kwargs.get("num_eval_loader_workers", max(0, min(1, num_loader_workers))))
    eval_split_max_size = kwargs.get("eval_split_max_size")

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        language="ko",
        path=str(data_dir),
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=str(pretrained["mel_norm_file"]),
        dvae_checkpoint=str(pretrained["dvae_checkpoint"]),
        xtts_checkpoint=str(pretrained["xtts_checkpoint"]),
        tokenizer_file=str(pretrained["tokenizer_file"]),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=str(ckpt_out),
        model_args=model_args,
        run_name=run_name,
        project_name="voice-clone",
        run_description="XTTS v2 fine-tune for voice cloning",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=runtime["batch_size"],
        batch_group_size=48,
        eval_batch_size=runtime["batch_size"],
        num_loader_workers=num_loader_workers,
        num_eval_loader_workers=num_eval_loader_workers,
        eval_split_max_size=eval_split_max_size,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=1000,
        save_n_checkpoints=5,
        save_checkpoints=True,
        print_eval=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[],
        mixed_precision=runtime["mixed_precision"],
        epochs=epochs,
        datasets=[dataset_config],
        run_eval=True,
        test_delay_epochs=0,
        target_loss="loss",  # Trainer는 avg_loss 키를 찾음. eval_loss 키는 avg_eval에 없음.
    )

    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=eval_split_max_size,
        eval_split_size=None,
    )
    eval_samples = eval_samples or []
    print(f"[xtts] loaded samples train={len(train_samples)} eval={len(eval_samples)}")

    model = GPTTrainer.init_from_config(config)
    callbacks = _EpochLossMonitor(early_stop_patience)
    trainer = Trainer(
        TrainerArgs(
            restore_path="",
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=int(kwargs.get("grad_accum_steps", 1)),
        ),
        config,
        output_path=str(ckpt_out),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        callbacks={"on_epoch_end": callbacks.on_epoch_end},
        parse_command_line_args=False,
    )
    _copy_xtts_sidecar_assets(Path(trainer.output_path), pretrained)

    return {
        "config": config,
        "trainer": trainer,
        "model": model,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "runtime": runtime,
        "pretrained": pretrained,
    }


@register("xtts")
class XttsEngine(VoiceCloneEngine):
    name = "xtts"
    train_sample_rate = 24000
    supports_language = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh-cn",
        "ja",
        "hu",
        "ko",
    ]
    license = "CPML (non-commercial)"
    phase = "phase0+phase_a_fallback"
    hardware_pref = "cuda"

    def health_check(self) -> tuple[bool, str]:
        ok, payload = _probe_xtts_runtime()
        if not ok:
            error = payload.get("tts_error", "unknown import error")
            return (
                False,
                "TTS 라이브러리 미설치: pip install -r model-training/requirements-cloning.txt "
                f"({error})",
            )

        device = str(payload.get("device", "cpu"))
        device_hint = {
            "cuda": "CUDA 사용 가능",
            "mps": "MPS 사용 가능 (학습은 실험적, CUDA 권장)",
            "cpu": "CPU만 사용 가능",
        }.get(device, f"device={device}")
        tos_agreed = bool(payload.get("tos_agreed"))
        if not tos_agreed:
            return False, f"COQUI_TOS_AGREED=1 환경변수가 필요합니다. {device_hint}"
        return True, f"XTTS runtime OK. {device_hint}"

    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        manifest = load_manifest(manifest_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = out_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        for split_name, metadata_name in (("train", "metadata_train.csv"), ("eval", "metadata_eval.csv")):
            rows: list[tuple[str, str, str]] = []
            for sample in split_samples(manifest, split_name):
                sample_id = str(sample["id"])
                src = _resolve_audio_path(manifest_path, sample, "wav_path_24k")
                dst = wavs_dir / f"{sample_id}.wav"
                _copy_or_link(src, dst)
                text = str(sample.get("text", "")).strip()
                normalized_text = str(sample.get("text_normalized") or normalize(text)).strip()
                rows.append((sample_id, text, normalized_text))

            metadata_path = out_dir / metadata_name
            with metadata_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="|", lineterminator="\n")
                writer.writerows(rows)

        return out_dir

    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 15,
        batch_size: int = 2,
        lr: float = 5e-6,
        early_stop_patience: int = 3,
        **kwargs: Any,
    ) -> Path:
        session = build_xtts_training_session(
            data_dir=data_dir,
            ckpt_out=ckpt_out,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            early_stop_patience=early_stop_patience,
            **kwargs,
        )
        trainer = session["trainer"]
        trainer.fit()

        run_dir = Path(trainer.output_path)
        best_model = _find_checkpoint_file(run_dir)
        if best_model.name != "best_model.pth":
            stable_best_model = run_dir / "best_model.pth"
            shutil.copy2(best_model, stable_best_model)
            best_model = stable_best_model
        print(f"[xtts] best_model={best_model}")
        return best_model

    def synthesize(
        self,
        text: str,
        ckpt: Path | None = None,
        ref_wav: Path | None = None,
        language: str = "ko",
        out_path: Path = Path("out.wav"),
        sample_rate: int | None = None,
    ) -> Path:
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"XTTS 추론 준비 실패: {message}")

        normalized_text = normalize(text)
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sample_rate = sample_rate or self.train_sample_rate

        if ckpt is None:
            if ref_wav is None:
                raise ValueError("XTTS zero-shot 모드는 ref_wav가 필요합니다.")
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise RuntimeError(
                    "TTS 라이브러리 미설치: pip install -r model-training/requirements-cloning.txt"
                ) from exc

            temp_out = out_path if target_sample_rate == self.train_sample_rate else out_path.with_suffix(".xtts_tmp.wav")
            tts = TTS(XTTS_MODEL_NAME)
            tts.tts_to_file(
                text=normalized_text,
                speaker_wav=str(ref_wav),
                language=language,
                file_path=str(temp_out),
            )
        else:
            try:
                _ensure_torch_load_patch()
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts
            except ImportError as exc:
                raise RuntimeError(
                    "XTTS fine-tuned checkpoint 추론에 필요한 Coqui TTS import 실패. "
                    "requirements-cloning.txt pin 버전을 확인하세요."
                ) from exc

            artifacts = _resolve_xtts_artifacts(ckpt)
            config = XttsConfig()
            if hasattr(config, "load_json"):
                config.load_json(str(artifacts["config_path"]))
            else:
                loaded = json.loads(artifacts["config_path"].read_text(encoding="utf-8"))
                for key, value in loaded.items():
                    setattr(config, key, value)

            model = Xtts.init_from_config(config)
            checkpoint_kwargs = {
                "config": config,
                "checkpoint_path": str(artifacts["model_path"]),
                "vocab_path": str(artifacts["vocab_path"]),
                "speaker_file_path": str(artifacts["speaker_file_path"]),
                "eval": True,
            }
            try:
                model.load_checkpoint(**checkpoint_kwargs)
            except TypeError:
                model.load_checkpoint(
                    config,
                    checkpoint_path=str(artifacts["model_path"]),
                    vocab_path=str(artifacts["vocab_path"]),
                    speaker_file_path=str(artifacts["speaker_file_path"]),
                    eval=True,
                )

            torch = _ensure_torch_load_patch()
            if torch.cuda.is_available():
                model = model.to("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                model = model.to("mps")

            synth_kwargs = {
                "text": normalized_text,
                "config": config,
                "speaker_wav": str(ref_wav) if ref_wav else None,
                "language": language,
            }
            try:
                result = model.synthesize(**synth_kwargs)
            except TypeError:
                result = model.synthesize(
                    normalized_text,
                    config,
                    speaker_wav=str(ref_wav) if ref_wav else None,
                    language=language,
                )

            wav_samples = result["wav"] if isinstance(result, dict) and "wav" in result else result
            temp_out = out_path if target_sample_rate == self.train_sample_rate else out_path.with_suffix(".xtts_tmp.wav")
            _write_wave_pcm16(temp_out, wav_samples, self.train_sample_rate)

        if target_sample_rate != self.train_sample_rate:
            _convert_audio_or_raise(temp_out, out_path, sample_rate=target_sample_rate)
            temp_out.unlink(missing_ok=True)

        return out_path
