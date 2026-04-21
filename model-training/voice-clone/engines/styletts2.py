from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from . import register
from ._manifest import load_manifest, split_samples
from .base import VoiceCloneEngine, run_in_venv
from model_training.voice_clone._audio_utils import convert_audio
from model_training.voice_clone.scripts.normalize_korean import normalize


_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
_KOREAN_WARNING = "한국어 파인튜닝 실증 사례가 적어 StyleTTS2는 실험적 엔진으로 취급합니다."


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


def _is_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _styletts2_paths() -> dict[str, Path]:
    repo_dir = _env_path("STYLETTS2_DIR", Path("~/tools/StyleTTS2"))
    return {
        "dir": repo_dir,
        "venv": _env_path("STYLETTS2_VENV", repo_dir / ".venv-styletts2" / "bin" / "python"),
        "config": _env_path("STYLETTS2_CONFIG", repo_dir / "Configs" / "config_ft.yml"),
        "train_script": _env_path("STYLETTS2_TRAIN_SCRIPT", repo_dir / "train_second.py"),
        "infer_script": _env_path("STYLETTS2_INFER_SCRIPT", repo_dir / "inference.py"),
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


def _probe_styletts2_import(venv_python: Path) -> tuple[bool, str]:
    result = _run_command(
        [
            str(venv_python),
            "-c",
            (
                "import importlib.util\n"
                "import sys\n"
                "sys.exit(0 if importlib.util.find_spec('styletts2') else 1)\n"
            ),
        ],
        timeout=30,
    )
    if result.returncode == 0:
        return True, "styletts2 package import 확인"
    error = result.stderr.strip() or result.stdout.strip() or "styletts2 package not found"
    return False, error


def _write_pip_launcher(script_path: Path, stage: str) -> None:
    if stage == "train":
        candidates = [
            "styletts2.train_second",
            "styletts2.train",
            "styletts2.cli",
            "styletts2",
        ]
    else:
        candidates = [
            "styletts2.inference",
            "styletts2.cli",
            "styletts2",
        ]

    candidate_repr = repr(candidates)
    script_path.write_text(
        (
            "from __future__ import annotations\n"
            "import importlib\n"
            "import sys\n"
            "\n"
            f"CANDIDATES = {candidate_repr}\n"
            "\n"
            "argv = sys.argv[1:]\n"
            "errors: list[str] = []\n"
            "for module_name in CANDIDATES:\n"
            "    try:\n"
            "        module = importlib.import_module(module_name)\n"
            "    except Exception as exc:\n"
            "        errors.append(f\"{module_name}: import failed: {exc!r}\")\n"
            "        continue\n"
            "    for attr in ('main', 'run'):\n"
            "        entry = getattr(module, attr, None)\n"
            "        if callable(entry):\n"
            "            sys.argv = [module_name, *argv]\n"
            "            result = entry()\n"
            "            raise SystemExit(result if isinstance(result, int) else 0)\n"
            "    errors.append(f\"{module_name}: callable main/run entry not found\")\n"
            "\n"
            "raise SystemExit(\n"
            "    'StyleTTS2 pip mode entrypoint discovery failed. Tried: ' + '; '.join(errors)\n"
            ")\n"
        ),
        encoding="utf-8",
    )


def _run_and_check(
    stage_name: str,
    venv_python: Path,
    script: Path,
    args: list[str],
    cwd: Path | None,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    result = run_in_venv(
        venv_python=venv_python,
        script=script,
        args=args,
        env=env,
        timeout=timeout,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"StyleTTS2 {stage_name} 단계 실패: {script}\n"
            f"args: {' '.join(args)}\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}"
        )
    return result


def _run_styletts2_stage(
    stage_name: str,
    venv_python: Path,
    script: Path,
    args: list[str],
    cwd: Path | None,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    if not _is_truthy("STYLETTS2_PIP_MODE"):
        return _run_and_check(
            stage_name=stage_name,
            venv_python=venv_python,
            script=script,
            args=args,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    with tempfile.TemporaryDirectory(prefix="styletts2-pip-") as temp_dir:
        launcher = Path(temp_dir) / f"{stage_name}_launcher.py"
        _write_pip_launcher(launcher, "train" if stage_name == "training" else "infer")
        return _run_and_check(
            stage_name=stage_name,
            venv_python=venv_python,
            script=launcher,
            args=args,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )


@register("styletts2")
class StyleTTS2Engine(VoiceCloneEngine):
    name = "styletts2"
    train_sample_rate = 24000
    supports_language = ["ko", "en"]
    license = "MIT"
    phase = "phase_c"
    hardware_pref = "cuda"

    def health_check(self) -> tuple[bool, str]:
        paths = _styletts2_paths()
        venv_python = paths["venv"]
        messages: list[str] = []
        ok = True

        if not venv_python.exists():
            ok = False
            messages.append(
                f"STYLETTS2_VENV 없음: {venv_python}. "
                "예: python -m venv ~/tools/StyleTTS2/.venv-styletts2"
            )
        else:
            version_result = _run_command([str(venv_python), "--version"], timeout=15)
            if version_result.returncode != 0:
                ok = False
                messages.append(
                    "STYLETTS2_VENV 실행 실패\n"
                    f"stdout:\n{version_result.stdout or '(empty)'}\n"
                    f"stderr:\n{version_result.stderr or '(empty)'}"
                )

        if _is_truthy("STYLETTS2_PIP_MODE"):
            if venv_python.exists():
                package_ok, package_message = _probe_styletts2_import(venv_python)
                ok = ok and package_ok
                if not package_ok:
                    messages.append(
                        "STYLETTS2_PIP_MODE=1 이지만 styletts2 패키지 import에 실패했습니다: "
                        f"{package_message}"
                    )
                else:
                    messages.append("STYLETTS2_PIP_MODE=1 / styletts2 pip 패키지 import 확인")
        else:
            repo_dir = paths["dir"]
            if not repo_dir.exists():
                ok = False
                messages.append(
                    f"STYLETTS2_DIR 없음: {repo_dir}. "
                    "예: git clone https://github.com/yl4579/StyleTTS2 ~/tools/StyleTTS2"
                )
            if not paths["config"].exists():
                ok = False
                messages.append(
                    f"STYLETTS2_CONFIG 없음: {paths['config']}"
                )
            if repo_dir.exists() and not paths["train_script"].exists():
                ok = False
                messages.append(
                    f"STYLETTS2_TRAIN_SCRIPT 없음: {paths['train_script']}"
                )
            if repo_dir.exists() and not paths["infer_script"].exists():
                ok = False
                messages.append(
                    f"STYLETTS2_INFER_SCRIPT 없음: {paths['infer_script']}"
                )

        messages.append(_KOREAN_WARNING)
        return ok, " / ".join(messages)

    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        manifest = load_manifest(manifest_path)
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = out_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows: list[tuple[str, str, str]] = []
        split_lines: dict[str, list[str]] = {"train": [], "eval": []}

        for split_name in ("train", "eval"):
            for sample in split_samples(manifest, split_name):
                sample_id = str(sample["id"])
                src = _resolve_audio_path(manifest_path, sample, "wav_path_24k")
                dst = wavs_dir / f"{sample_id}.wav"
                _copy_or_link(src, dst)
                text = str(sample.get("text", "")).strip()
                normalized_text = str(sample.get("text_normalized") or normalize(text)).strip()
                rel_wav = dst.relative_to(out_dir).as_posix()
                metadata_rows.append((sample_id, text, normalized_text))
                split_lines[split_name].append(f"{rel_wav}|{normalized_text}")

        with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="|", lineterminator="\n")
            writer.writerows(metadata_rows)

        for split_name, filename in (("train", "train_list.txt"), ("eval", "val_list.txt")):
            (out_dir / filename).write_text(
                "\n".join(split_lines[split_name]) + "\n",
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
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"StyleTTS2 학습 준비 실패: {message}")

        paths = _styletts2_paths()
        data_dir = data_dir.expanduser().resolve()
        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)

        config_path = Path(kwargs.pop("config_path", paths["config"])).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"StyleTTS2 config를 찾을 수 없습니다: {config_path}")

        train_list = data_dir / "train_list.txt"
        val_list = data_dir / "val_list.txt"
        if not train_list.exists() or not val_list.exists():
            raise FileNotFoundError(
                f"StyleTTS2 학습용 목록 파일이 없습니다: {train_list}, {val_list}"
            )

        batch_size = int(kwargs.pop("batch_size", 4))
        timeout = int(kwargs.pop("timeout", 8 * 3600))
        extra_args = _normalize_extra_args(kwargs.pop("train_args", None), "train_args")
        cli_args = [
            "--config-path",
            str(config_path),
            "--train-list",
            str(train_list),
            "--val-list",
            str(val_list),
            "--data-root",
            str(data_dir),
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

        script = paths["train_script"].expanduser().resolve()
        cwd = paths["dir"].expanduser().resolve() if paths["dir"].exists() else None
        _run_styletts2_stage(
            stage_name="training",
            venv_python=paths["venv"].expanduser().resolve(),
            script=script,
            args=cli_args,
            cwd=cwd,
            env={"STYLETTS2_CONFIG": str(config_path)},
            timeout=timeout,
        )

        if not _has_checkpoint_files(ckpt_out):
            raise RuntimeError(
                f"StyleTTS2 학습은 끝났지만 checkpoint 산출물을 찾지 못했습니다: {ckpt_out}"
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
            raise RuntimeError(f"StyleTTS2 추론 준비 실패: {message}")
        if ref_wav is None:
            raise ValueError("StyleTTS2 synthesize는 ref_wav가 필수입니다.")

        paths = _styletts2_paths()
        config_path = Path(kwargs.pop("config_path", paths["config"])).expanduser().resolve()
        if config_path.exists():
            config_args = ["--config-path", str(config_path)]
        else:
            config_args = []

        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sample_rate = sample_rate or self.train_sample_rate
        temp_out = (
            out_path
            if target_sample_rate == self.train_sample_rate
            else out_path.with_suffix(".styletts2_tmp.wav")
        )
        extra_args = _normalize_extra_args(kwargs.pop("infer_args", None), "infer_args")
        ckpt_source = ckpt.expanduser().resolve() if ckpt is not None else None

        cli_args = [
            *config_args,
            "--text",
            text,
            "--ref-wav",
            str(ref_wav.expanduser().resolve()),
            "--language",
            language,
            "--output",
            str(temp_out),
        ]
        if ckpt_source is not None:
            cli_args.extend(["--checkpoint", str(ckpt_source)])

        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)
            elif value is not None:
                cli_args.extend([flag, str(value)])
        cli_args.extend(extra_args)

        script = paths["infer_script"].expanduser().resolve()
        cwd = paths["dir"].expanduser().resolve() if paths["dir"].exists() else None
        _run_styletts2_stage(
            stage_name="inference",
            venv_python=paths["venv"].expanduser().resolve(),
            script=script,
            args=cli_args,
            cwd=cwd,
            env={"STYLETTS2_CONFIG": str(config_path)} if config_path.exists() else None,
            timeout=int(os.getenv("STYLETTS2_INFER_TIMEOUT", "3600")),
        )

        if not temp_out.exists():
            raise RuntimeError(
                f"StyleTTS2 추론 subprocess는 성공했지만 출력 WAV를 찾지 못했습니다: {temp_out}"
            )

        if target_sample_rate != self.train_sample_rate:
            if not convert_audio(temp_out, out_path, sr=target_sample_rate, channels=1):
                raise RuntimeError(
                    f"StyleTTS2 출력 리샘플 실패: {temp_out} -> {out_path} ({target_sample_rate}Hz)"
                )
            temp_out.unlink(missing_ok=True)
        elif temp_out != out_path:
            shutil.copy2(temp_out, out_path)
        return out_path
