from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from . import register
from ._manifest import load_manifest, split_samples
from .base import VoiceCloneEngine, run_in_venv
from model_training.voice_clone._audio_utils import convert_audio
from model_training.voice_clone.scripts.normalize_korean import normalize


_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


def _indextts2_paths() -> dict[str, Path]:
    repo_dir = _env_path("INDEXTTS2_DIR", Path("~/tools/index-tts"))
    return {
        "dir": repo_dir,
        "venv": _env_path("INDEXTTS2_VENV", repo_dir / ".venv-indextts2" / "bin" / "python"),
        "model_dir": _env_path("INDEXTTS2_MODEL", repo_dir / "checkpoints"),
        "infer_script": _env_path(
            "INDEXTTS2_INFER_SCRIPT",
            repo_dir / "indextts" / "infer.py",
        ),
        "finetune_script": _env_path("INDEXTTS2_FINETUNE_SCRIPT", repo_dir / "finetune.py"),
    }


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _resolve_audio_path(
    manifest_path: Path,
    sample: dict[str, Any],
    key: str = "wav_path_24k",
) -> Path:
    raw_value = sample.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Sample '{sample.get('id', '<unknown>')}' is missing '{key}'.")
    path = Path(raw_value)
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file referenced by manifest does not exist: {path}")
    return path


def _run_command(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
    )


def _normalize_extra_args(value: Any, arg_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"{arg_name} must be a list/tuple of CLI arguments, got {type(value)!r}")


def _has_checkpoint_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in _CHECKPOINT_SUFFIXES
    return any(
        child.is_file() and child.suffix.lower() in _CHECKPOINT_SUFFIXES
        for child in path.rglob("*")
    )


def _run_and_check(
    stage_name: str,
    venv_python: Path,
    script: Path,
    args: list[str],
    repo_dir: Path,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    result = run_in_venv(
        venv_python=venv_python,
        script=script,
        args=args,
        env=env,
        timeout=timeout,
        cwd=repo_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"IndexTTS2 {stage_name} 단계 실패: {script}\n"
            f"args: {' '.join(args)}\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}"
        )
    return result


@register("indextts2")
class IndexTTS2Engine(VoiceCloneEngine):
    name = "indextts2"
    train_sample_rate = 24000
    supports_language = ["ko", "en", "zh", "ja"]
    license = "Apache-2.0"
    phase = "phase_c"
    hardware_pref = "cuda"

    def health_check(self) -> tuple[bool, str]:
        paths = _indextts2_paths()
        repo_dir = paths["dir"]
        venv_python = paths["venv"]
        model_dir = paths["model_dir"]
        messages: list[str] = []
        ok = True

        if not repo_dir.exists():
            ok = False
            messages.append(
                f"INDEXTTS2_DIR 없음: {repo_dir}. "
                "예: git clone https://github.com/index-tts/index-tts ~/tools/index-tts"
            )
        if not venv_python.exists():
            ok = False
            messages.append(
                f"INDEXTTS2_VENV 없음: {venv_python}. "
                "예: python -m venv ~/tools/index-tts/.venv-indextts2"
            )
        if not model_dir.exists():
            ok = False
            messages.append(
                f"INDEXTTS2_MODEL 없음: {model_dir}. "
                "공식 checkpoints 경로를 지정하세요."
            )
        elif not _has_checkpoint_files(model_dir):
            ok = False
            messages.append(
                f"INDEXTTS2_MODEL 아래 checkpoint 파일을 찾지 못했습니다: {model_dir}"
            )

        if repo_dir.exists() and not paths["infer_script"].exists():
            ok = False
            messages.append(
                f"INDEXTTS2_INFER_SCRIPT 없음: {paths['infer_script']}"
            )

        if venv_python.exists():
            version_result = _run_command([str(venv_python), "--version"], timeout=15)
            if version_result.returncode != 0:
                ok = False
                messages.append(
                    "INDEXTTS2_VENV 실행 실패\n"
                    f"stdout:\n{version_result.stdout or '(empty)'}\n"
                    f"stderr:\n{version_result.stderr or '(empty)'}"
                )

        if not paths["finetune_script"].exists():
            messages.append(
                "INDEXTTS2_FINETUNE_SCRIPT 없음: 현재 zero-shot synthesize 중심으로 사용, "
                "train() 호출 시 NotImplementedError가 발생합니다."
            )

        if ok:
            messages.append(
                f"repo={repo_dir}; venv={venv_python}; model={model_dir}; hw_pref={self.hardware_pref}"
            )
        return ok, " / ".join(messages)

    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        manifest = load_manifest(manifest_path)
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = out_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        for split_name, filename in (("train", "train.jsonl"), ("eval", "val.jsonl")):
            lines: list[str] = []
            for sample in split_samples(manifest, split_name):
                sample_id = str(sample["id"])
                src = _resolve_audio_path(manifest_path, sample, "wav_path_24k")
                dst = wavs_dir / f"{sample_id}.wav"
                _copy_or_link(src, dst)
                normalized_text = str(sample.get("text_normalized") or normalize(str(sample.get("text", "")))).strip()
                emotion = str(sample.get("emotion") or "neutral").strip() or "neutral"
                payload = {
                    "wav": str(dst.resolve()),
                    "text": normalized_text,
                    "emotion": emotion,
                }
                lines.append(json.dumps(payload, ensure_ascii=False))

            (out_dir / filename).write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

        return out_dir

    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Path:
        paths = _indextts2_paths()
        if not paths["finetune_script"].exists():
            raise NotImplementedError(
                "IndexTTS2 파인튜닝 스크립트가 확인되지 않았습니다. "
                "INDEXTTS2_FINETUNE_SCRIPT를 지정하거나 zero-shot synthesize만 사용하세요."
            )

        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"IndexTTS2 학습 준비 실패: {message}")

        data_dir = data_dir.expanduser().resolve()
        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)

        train_manifest = data_dir / "train.jsonl"
        val_manifest = data_dir / "val.jsonl"
        if not train_manifest.exists() or not val_manifest.exists():
            raise FileNotFoundError(
                f"IndexTTS2 학습용 manifest가 없습니다: {train_manifest}, {val_manifest}"
            )

        batch_size = int(kwargs.pop("batch_size", 4))
        timeout = int(kwargs.pop("timeout", 8 * 3600))
        extra_args = _normalize_extra_args(kwargs.pop("train_args", None), "train_args")
        cli_args = [
            "--train-manifest",
            str(train_manifest),
            "--val-manifest",
            str(val_manifest),
            "--checkpoint-dir",
            str(paths["model_dir"].expanduser().resolve()),
            "--output-dir",
            str(ckpt_out),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ]

        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)
            elif value is not None:
                cli_args.extend([flag, str(value)])
        cli_args.extend(extra_args)

        _run_and_check(
            "training",
            venv_python=paths["venv"].expanduser().resolve(),
            script=paths["finetune_script"].expanduser().resolve(),
            repo_dir=paths["dir"].expanduser().resolve(),
            args=cli_args,
            env={"INDEXTTS2_MODEL": str(paths["model_dir"].expanduser().resolve())},
            timeout=timeout,
        )

        if not _has_checkpoint_files(ckpt_out):
            raise RuntimeError(
                f"IndexTTS2 학습은 끝났지만 checkpoint 산출물을 찾지 못했습니다: {ckpt_out}"
            )
        return ckpt_out

    def synthesize(
        self,
        text: str,
        ckpt: Path | None,
        ref_wav: Path | None = None,
        language: str = "ko",
        out_path: Path = Path("out.wav"),
        sample_rate: int | None = None,
        **kwargs: Any,
    ) -> Path:
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"IndexTTS2 추론 준비 실패: {message}")
        if ref_wav is None:
            raise ValueError("IndexTTS2 synthesize는 ref_wav가 필수입니다.")

        paths = _indextts2_paths()
        repo_dir = paths["dir"].expanduser().resolve()
        venv_python = paths["venv"].expanduser().resolve()
        infer_script = paths["infer_script"].expanduser().resolve()
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        target_sample_rate = sample_rate or self.train_sample_rate
        temp_out = (
            out_path
            if target_sample_rate == self.train_sample_rate
            else out_path.with_suffix(".indextts2_tmp.wav")
        )
        emotion = str(kwargs.pop("emotion", "neutral")).strip() or "neutral"
        extra_args = _normalize_extra_args(kwargs.pop("infer_args", None), "infer_args")
        model_source = ckpt.expanduser().resolve() if ckpt is not None else paths["model_dir"].expanduser().resolve()

        cli_args = [
            "--text",
            text,
            "--language",
            language,
            "--ref-wav",
            str(ref_wav.expanduser().resolve()),
            "--emotion",
            emotion,
            "--output",
            str(temp_out),
        ]
        if ckpt is not None:
            cli_args.extend(["--checkpoint", str(model_source)])
        else:
            cli_args.extend(["--model-dir", str(model_source)])

        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)
            elif value is not None:
                cli_args.extend([flag, str(value)])
        cli_args.extend(extra_args)

        _run_and_check(
            "inference",
            venv_python=venv_python,
            script=infer_script,
            repo_dir=repo_dir,
            args=cli_args,
            env={"INDEXTTS2_MODEL": str(paths["model_dir"].expanduser().resolve())},
            timeout=int(os.getenv("INDEXTTS2_INFER_TIMEOUT", "3600")),
        )

        if not temp_out.exists():
            raise RuntimeError(
                f"IndexTTS2 추론 subprocess는 성공했지만 출력 WAV를 찾지 못했습니다: {temp_out}"
            )

        if target_sample_rate != self.train_sample_rate:
            if not convert_audio(temp_out, out_path, sr=target_sample_rate, channels=1):
                raise RuntimeError(
                    f"IndexTTS2 출력 리샘플 실패: {temp_out} -> {out_path} ({target_sample_rate}Hz)"
                )
            temp_out.unlink(missing_ok=True)
        elif temp_out != out_path:
            shutil.copy2(temp_out, out_path)
        return out_path
