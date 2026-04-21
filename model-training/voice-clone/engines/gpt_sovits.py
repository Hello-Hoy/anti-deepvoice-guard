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


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


def _gpt_sovits_paths() -> dict[str, Path | str]:
    repo_dir = _env_path("GPT_SOVITS_DIR", Path("~/tools/GPT-SoVITS"))
    return {
        "dir": repo_dir,
        "venv": _env_path(
            "GPT_SOVITS_VENV",
            repo_dir / ".venv-gpt-sovits" / "bin" / "python",
        ),
        "commit": os.getenv("GPT_SOVITS_COMMIT", "v2-stable"),
        "feature_script": _env_path(
            "GPT_SOVITS_FEATURE_SCRIPT",
            repo_dir / "tools" / "feature_extractor" / "cnhubert.py",
        ),
        "s2_train_script": _env_path(
            "GPT_SOVITS_S2_TRAIN_SCRIPT",
            repo_dir / "s2_train.py",
        ),
        "s1_train_script": _env_path(
            "GPT_SOVITS_S1_TRAIN_SCRIPT",
            repo_dir / "s1_train.py",
        ),
        "inference_script": _env_path(
            "GPT_SOVITS_INFERENCE_SCRIPT",
            repo_dir / "GPT_SoVITS" / "inference_cli.py",
        ),
    }


def _install_hint(paths: dict[str, Path | str]) -> str:
    repo_dir = Path(paths["dir"])
    venv_python = Path(paths["venv"])
    expected_commit = str(paths["commit"])
    return (
        "설치 힌트: "
        f"git clone https://github.com/RVC-Boss/GPT-SoVITS {repo_dir} && "
        f"cd {repo_dir} && git checkout {expected_commit} && "
        f"python -m venv {repo_dir / '.venv-gpt-sovits'} && "
        f"{venv_python} -m pip install -r requirements.txt"
    )


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


def _run_command(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
    )


def _probe_torch_runtime(venv_python: Path) -> dict[str, Any]:
    probe = _run_command(
        [
            str(venv_python),
            "-c",
            (
                "import json\n"
                "payload = {}\n"
                "try:\n"
                "    import torch\n"
                "    payload['torch_import'] = True\n"
                "    payload['cuda_available'] = bool(torch.cuda.is_available())\n"
                "    mps_backend = getattr(torch.backends, 'mps', None)\n"
                "    payload['mps_available'] = bool(mps_backend and mps_backend.is_available())\n"
                "    payload['torch_version'] = getattr(torch, '__version__', 'unknown')\n"
                "except Exception as exc:\n"
                "    payload['torch_import'] = False\n"
                "    payload['error'] = repr(exc)\n"
                "print(json.dumps(payload, ensure_ascii=False))\n"
            ),
        ],
        timeout=30,
    )
    if probe.returncode != 0:
        return {
            "torch_import": False,
            "error": probe.stderr.strip() or probe.stdout.strip() or "torch runtime probe failed",
        }
    try:
        return json.loads(probe.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {
            "torch_import": False,
            "error": f"invalid torch probe output: {probe.stdout.strip()}",
        }


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
            f"GPT-SoVITS {stage_name} 단계 실패: {script}\n"
            f"args: {' '.join(args)}\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}\n"
            "힌트: GPT_SOVITS_DIR, GPT_SOVITS_VENV, 각 단계별 *_SCRIPT override와 CUDA 환경을 확인하세요."
        )
    return result


def _resolve_stage_output(stage_dir: Path, preferred_name: str, suffixes: set[str]) -> Path:
    preferred = stage_dir / preferred_name
    if preferred.exists():
        return preferred

    candidates = sorted(
        path
        for path in stage_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not candidates:
        raise FileNotFoundError(
            f"GPT-SoVITS 산출물을 찾지 못했습니다: expected={preferred_name}, stage_dir={stage_dir}"
        )
    return candidates[0]


def _copy_or_raise(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"복사할 파일이 존재하지 않습니다: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _normalize_extra_args(value: Any, arg_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"{arg_name} must be a list/tuple of CLI arguments, got {type(value)!r}")


def _resolve_checkpoint_dir(ckpt: Path) -> Path:
    ckpt_path = ckpt.expanduser().resolve()
    return ckpt_path if ckpt_path.is_dir() else ckpt_path.parent


def _resolve_synthesis_artifacts(ckpt: Path) -> tuple[Path, Path, Path]:
    ckpt_dir = _resolve_checkpoint_dir(ckpt)
    s2_ckpt = ckpt_dir / "s2G.pth"
    s1_ckpt = ckpt_dir / "s1bert.pth"
    config_path = ckpt_dir / "config.json"
    if not s2_ckpt.exists():
        raise FileNotFoundError(f"GPT-SoVITS S2 checkpoint를 찾을 수 없습니다: {s2_ckpt}")
    if not s1_ckpt.exists():
        raise FileNotFoundError(f"GPT-SoVITS S1 checkpoint를 찾을 수 없습니다: {s1_ckpt}")
    if not config_path.exists():
        raise FileNotFoundError(f"GPT-SoVITS config.json을 찾을 수 없습니다: {config_path}")
    return s2_ckpt, s1_ckpt, config_path


def _load_ref_text_map(ckpt_dir: Path) -> dict[str, str]:
    mapping_path = ckpt_dir / "ref_text_map.json"
    if not mapping_path.exists():
        return {}
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    refs = payload.get("refs")
    return refs if isinstance(refs, dict) else {}


@register("gpt_sovits")
class GPTSoVITSEngine(VoiceCloneEngine):
    name = "gpt_sovits"
    train_sample_rate = 32000
    supports_language = ["ko", "en", "zh", "ja", "yue"]
    license = "MIT"
    phase = "phase_b"
    hardware_pref = "cuda_required"

    def health_check(self) -> tuple[bool, str]:
        paths = _gpt_sovits_paths()
        repo_dir = Path(paths["dir"])
        venv_python = Path(paths["venv"])
        install_hint = _install_hint(paths)

        if not repo_dir.exists():
            return False, f"GPT_SOVITS_DIR 없음: {repo_dir}. {install_hint}"
        if not venv_python.exists():
            return False, f"GPT_SOVITS_VENV 없음: {venv_python}. {install_hint}"

        version_probe = _run_command([str(venv_python), "--version"], timeout=15)
        if version_probe.returncode != 0:
            return (
                False,
                f"GPT_SOVITS_VENV 실행 실패: {venv_python}\n"
                f"stdout:\n{version_probe.stdout or '(empty)'}\n"
                f"stderr:\n{version_probe.stderr or '(empty)'}\n"
                f"{install_hint}"
            )

        missing_scripts = [
            name
            for name in ("feature_script", "s2_train_script", "s1_train_script", "inference_script")
            if not Path(paths[name]).exists()
        ]
        if missing_scripts:
            missing_desc = ", ".join(f"{name}={paths[name]}" for name in missing_scripts)
            return False, f"GPT-SoVITS script 없음: {missing_desc}. *_SCRIPT 환경변수로 override 가능. {install_hint}"

        probe = _probe_torch_runtime(venv_python)
        if not probe.get("torch_import"):
            return False, f"GPT-SoVITS venv에서 torch import 실패: {probe.get('error', 'unknown error')}. {install_hint}"
        if not probe.get("cuda_available"):
            message = (
                "GPT-SoVITS는 CUDA 필수 경로로 가정합니다. "
                "현재 torch.cuda.is_available() == False 입니다. "
                "M4 Pro MPS 경로는 비권장/미지원에 가깝고 Linux+NVIDIA CUDA 환경을 권장합니다."
            )
            if probe.get("mps_available"):
                message += " 현재 MPS는 감지되지만 GPT-SoVITS 안정 경로로 보지 않습니다."
            return False, message

        notes = [
            f"repo={repo_dir}",
            f"venv={venv_python}",
            f"python={version_probe.stdout.strip() or version_probe.stderr.strip()}",
            f"torch={probe.get('torch_version', 'unknown')}",
            "cuda=available",
        ]
        git_probe = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_dir, timeout=15)
        if git_probe.returncode == 0:
            actual_commit = git_probe.stdout.strip()
            expected_commit = str(paths["commit"])
            if actual_commit and actual_commit != expected_commit:
                notes.append(
                    f"commit-warning actual={actual_commit} expected={expected_commit} (경고만, 실행은 계속 가능)"
                )
        else:
            notes.append("git-commit-check skipped")
        return True, "; ".join(notes)

    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        manifest = load_manifest(manifest_path)
        speaker = str(manifest["speaker"])
        raw_dir = out_dir / "raw" / speaker
        lists_dir = out_dir / "lists"
        raw_dir.mkdir(parents=True, exist_ok=True)
        lists_dir.mkdir(parents=True, exist_ok=True)

        ref_map: dict[str, str] = {}
        split_file_map = {"train": "train.list", "eval": "val.list"}

        for split_name, filename in split_file_map.items():
            lines: list[str] = []
            for sample in split_samples(manifest, split_name):
                sample_id = str(sample["id"])
                src = _resolve_audio_path(manifest_path, sample, "wav_path_24k")
                dst = raw_dir / f"{sample_id}.wav"
                if not convert_audio(src, dst, sr=self.train_sample_rate, channels=1):
                    raise RuntimeError(
                        f"GPT-SoVITS 학습용 오디오 변환 실패: {src} -> {dst} ({self.train_sample_rate}Hz)"
                    )

                language = str(sample.get("language") or "ko").strip() or "ko"
                text = str(sample.get("text") or "").strip()
                if not text:
                    raise ValueError(f"Sample '{sample_id}' is missing non-empty text.")
                lines.append(f"{dst.name}|{speaker}|{language}|{text}")
                ref_map[dst.name] = text
                ref_map[str(dst.resolve())] = text

            (lists_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

        (out_dir / "ref_text_map.json").write_text(
            json.dumps(
                {
                    "speaker": speaker,
                    "train_sample_rate": self.train_sample_rate,
                    "refs": ref_map,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return out_dir

    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 15,
        **kwargs: Any,
    ) -> Path:
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"GPT-SoVITS 학습 준비 실패: {message}")

        paths = _gpt_sovits_paths()
        repo_dir = Path(paths["dir"]).expanduser().resolve()
        venv_python = Path(paths["venv"]).expanduser().resolve()
        data_dir = data_dir.expanduser().resolve()
        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)

        train_list = data_dir / "lists" / "train.list"
        val_list = data_dir / "lists" / "val.list"
        raw_dir = data_dir / "raw"
        if not train_list.exists() or not val_list.exists():
            raise FileNotFoundError(
                f"GPT-SoVITS 데이터 리스트를 찾을 수 없습니다: train={train_list}, val={val_list}"
            )
        if not raw_dir.exists():
            raise FileNotFoundError(f"GPT-SoVITS raw 디렉토리를 찾을 수 없습니다: {raw_dir}")

        feature_args = _normalize_extra_args(kwargs.pop("feature_args", None), "feature_args")
        s2_args = _normalize_extra_args(kwargs.pop("s2_args", None), "s2_args")
        s1_args = _normalize_extra_args(kwargs.pop("s1_args", None), "s1_args")
        shared_train_args = _normalize_extra_args(kwargs.pop("train_args", None), "train_args")
        batch_size = int(kwargs.pop("batch_size", 4))
        s2_epochs = int(kwargs.pop("s2_epochs", 8))
        s1_epochs = int(kwargs.pop("s1_epochs", epochs if epochs is not None else 15))
        feature_timeout = int(kwargs.pop("feature_timeout", 2 * 3600))
        s2_timeout = int(kwargs.pop("s2_timeout", 8 * 3600))
        s1_timeout = int(kwargs.pop("s1_timeout", 8 * 3600))

        passthrough_flags: list[str] = []
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    passthrough_flags.append(flag)
            elif value is not None:
                passthrough_flags.extend([flag, str(value)])

        feature_dir = ckpt_out / "_feature_cache"
        s2_dir = ckpt_out / "_s2"
        s1_dir = ckpt_out / "_s1"
        feature_dir.mkdir(parents=True, exist_ok=True)
        s2_dir.mkdir(parents=True, exist_ok=True)
        s1_dir.mkdir(parents=True, exist_ok=True)

        _run_and_check(
            "HuBERT feature extraction",
            venv_python=venv_python,
            script=Path(paths["feature_script"]),
            repo_dir=repo_dir,
            args=[
                "--input_dir",
                str(raw_dir),
                "--train_list",
                str(train_list),
                "--val_list",
                str(val_list),
                "--output_dir",
                str(feature_dir),
                "--sample_rate",
                str(self.train_sample_rate),
                *feature_args,
            ],
            timeout=feature_timeout,
        )

        _run_and_check(
            "SoVITS S2 training",
            venv_python=venv_python,
            script=Path(paths["s2_train_script"]),
            repo_dir=repo_dir,
            args=[
                "--data_dir",
                str(data_dir),
                "--feature_dir",
                str(feature_dir),
                "--train_list",
                str(train_list),
                "--val_list",
                str(val_list),
                "--output_dir",
                str(s2_dir),
                "--epochs",
                str(s2_epochs),
                "--batch_size",
                str(batch_size),
                *shared_train_args,
                *passthrough_flags,
                *s2_args,
            ],
            timeout=s2_timeout,
        )

        s2_src = _resolve_stage_output(s2_dir, "s2G.pth", {".pth", ".pt", ".ckpt"})

        _run_and_check(
            "GPT S1 training",
            venv_python=venv_python,
            script=Path(paths["s1_train_script"]),
            repo_dir=repo_dir,
            args=[
                "--data_dir",
                str(data_dir),
                "--feature_dir",
                str(feature_dir),
                "--train_list",
                str(train_list),
                "--val_list",
                str(val_list),
                "--s2_ckpt",
                str(s2_src),
                "--output_dir",
                str(s1_dir),
                "--epochs",
                str(s1_epochs),
                "--batch_size",
                str(batch_size),
                *shared_train_args,
                *passthrough_flags,
                *s1_args,
            ],
            timeout=s1_timeout,
        )

        s1_src = _resolve_stage_output(s1_dir, "s1bert.pth", {".pth", ".pt", ".ckpt"})
        config_src = s1_dir / "config.json"
        final_s2 = ckpt_out / "s2G.pth"
        final_s1 = ckpt_out / "s1bert.pth"
        final_config = ckpt_out / "config.json"

        _copy_or_raise(s2_src, final_s2)
        _copy_or_raise(s1_src, final_s1)

        config_payload = {
            "engine": self.name,
            "license": self.license,
            "phase": self.phase,
            "train_sample_rate": self.train_sample_rate,
            "repo_dir": str(repo_dir),
            "repo_commit_expected": str(paths["commit"]),
            "scripts": {
                "feature": str(Path(paths["feature_script"])),
                "s2_train": str(Path(paths["s2_train_script"])),
                "s1_train": str(Path(paths["s1_train_script"])),
                "inference": str(Path(paths["inference_script"])),
            },
            "training": {
                "data_dir": str(data_dir),
                "feature_dir": str(feature_dir),
                "s2_epochs": s2_epochs,
                "s1_epochs": s1_epochs,
                "batch_size": batch_size,
            },
        }
        if config_src.exists():
            try:
                upstream = json.loads(config_src.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                upstream = {"upstream_config_path": str(config_src)}
            config_payload["upstream"] = upstream
        final_config.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        ref_text_map = data_dir / "ref_text_map.json"
        if ref_text_map.exists():
            _copy_or_raise(ref_text_map, ckpt_out / "ref_text_map.json")

        return ckpt_out

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
            raise RuntimeError(f"GPT-SoVITS 추론 준비 실패: {message}")
        if ckpt is None:
            raise ValueError("GPT-SoVITS 추론은 학습된 체크포인트 디렉토리 또는 파일이 필요합니다.")
        if ref_wav is None:
            raise ValueError("GPT-SoVITS 추론은 ref_wav가 필요합니다.")
        if language not in self.supports_language:
            raise ValueError(
                f"지원하지 않는 language='{language}'. supported={', '.join(self.supports_language)}"
            )

        paths = _gpt_sovits_paths()
        repo_dir = Path(paths["dir"]).expanduser().resolve()
        venv_python = Path(paths["venv"]).expanduser().resolve()
        s2_ckpt, s1_ckpt, _config_path = _resolve_synthesis_artifacts(ckpt)
        ckpt_dir = _resolve_checkpoint_dir(ckpt)

        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sample_rate = sample_rate or self.train_sample_rate
        temp_out = out_path if target_sample_rate == self.train_sample_rate else out_path.with_suffix(".gpt_sovits_tmp.wav")

        ref_text_map = _load_ref_text_map(ckpt_dir)
        resolved_ref_wav = ref_wav.expanduser().resolve()
        ref_text = ref_text_map.get(resolved_ref_wav.name) or ref_text_map.get(str(resolved_ref_wav))

        script_args = [
            "--text",
            text,
            "--ref_wav",
            str(resolved_ref_wav),
            "--ckpt_s2",
            str(s2_ckpt),
            "--ckpt_s1",
            str(s1_ckpt),
            "--language",
            language,
            "--out",
            str(temp_out),
        ]
        if ref_text:
            script_args.extend(["--ref_text", ref_text])

        _run_and_check(
            "inference",
            venv_python=venv_python,
            script=Path(paths["inference_script"]),
            repo_dir=repo_dir,
            args=script_args,
            timeout=1800,
        )

        if not temp_out.exists():
            raise RuntimeError(
                f"GPT-SoVITS inference subprocess는 성공했지만 출력 WAV를 찾지 못했습니다: {temp_out}"
            )
        if target_sample_rate != self.train_sample_rate:
            if not convert_audio(temp_out, out_path, sr=target_sample_rate, channels=1):
                raise RuntimeError(
                    f"GPT-SoVITS 출력 리샘플링 실패: {temp_out} -> {out_path} ({target_sample_rate}Hz)"
                )
            temp_out.unlink(missing_ok=True)
        return out_path
