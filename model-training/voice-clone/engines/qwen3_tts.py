from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from . import register
from .base import VoiceCloneEngine, run_in_venv
from model_training.voice_clone._audio_utils import convert_audio


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


def _qwen3_paths() -> dict[str, Path]:
    repo_dir = _env_path("QWEN3_TTS_DIR", Path("~/tools/Qwen3-TTS"))
    return {
        "dir": repo_dir,
        "venv": _env_path("QWEN3_TTS_VENV", repo_dir / ".venv-qwen3" / "bin" / "python"),
        "model_dir": _env_path("QWEN3_TTS_MODEL", repo_dir / "pretrained"),
        "train_script": _env_path("QWEN3_TTS_TRAIN_SCRIPT", repo_dir / "finetune.py"),
        "infer_script": _env_path("QWEN3_TTS_INFER_SCRIPT", repo_dir / "inference.py"),
    }


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
            f"Qwen3-TTS {stage_name} 단계 실패: {script}\n"
            f"args: {' '.join(args)}\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}"
        )
    return result


def _probe_qwen_runtime(venv_python: Path) -> dict[str, Any]:
    result = _run_command(
        [
            str(venv_python),
            "-c",
            (
                "import json\n"
                "payload = {}\n"
                "try:\n"
                "    import qwen_tts\n"
                "    payload['qwen_tts_import'] = True\n"
                "except Exception as exc:\n"
                "    payload['qwen_tts_import'] = False\n"
                "    payload['qwen_tts_error'] = repr(exc)\n"
                "try:\n"
                "    import torch\n"
                "    payload['torch_import'] = True\n"
                "    payload['cuda_available'] = bool(torch.cuda.is_available())\n"
                "except Exception as exc:\n"
                "    payload['torch_import'] = False\n"
                "    payload['torch_error'] = repr(exc)\n"
                "print(json.dumps(payload, ensure_ascii=False))\n"
            ),
        ],
        timeout=30,
    )
    if result.returncode != 0:
        return {
            "qwen_tts_import": False,
            "qwen_tts_error": result.stderr.strip() or result.stdout.strip() or "runtime probe failed",
            "torch_import": False,
        }
    try:
        import json

        return json.loads(result.stdout.strip() or "{}")
    except Exception:
        return {
            "qwen_tts_import": False,
            "qwen_tts_error": f"invalid runtime probe output: {result.stdout.strip()}",
            "torch_import": False,
        }


@register("qwen3_tts")
class Qwen3TTSEngine(VoiceCloneEngine):
    name = "qwen3_tts"
    train_sample_rate = 24000
    supports_language = ["ko", "en", "zh", "ja", "de", "fr", "ru", "pt", "es", "it"]
    license = "Apache-2.0"
    phase = "phase_c"
    hardware_pref = "cuda"

    def health_check(self) -> tuple[bool, str]:
        paths = _qwen3_paths()
        repo_dir = paths["dir"]
        venv_python = paths["venv"]
        messages: list[str] = []
        ok = True

        if not repo_dir.exists():
            ok = False
            messages.append(
                f"QWEN3_TTS_DIR 없음: {repo_dir}. "
                "예: git clone https://github.com/QwenLM/Qwen3-TTS ~/tools/Qwen3-TTS"
            )
        if not venv_python.exists():
            ok = False
            messages.append(
                f"QWEN3_TTS_VENV 없음: {venv_python}. "
                "예: python -m venv ~/tools/Qwen3-TTS/.venv-qwen3 && pip install -U qwen-tts"
            )
        if not paths["model_dir"].exists():
            ok = False
            messages.append(
                f"QWEN3_TTS_MODEL 없음: {paths['model_dir']}. "
                "공식 pretrained 디렉토리 또는 Hugging Face/ModelScope 다운로드 경로를 지정하세요."
            )
        if repo_dir.exists() and not paths["infer_script"].exists():
            ok = False
            messages.append(
                f"QWEN3_TTS_INFER_SCRIPT 없음: {paths['infer_script']}"
            )

        if venv_python.exists():
            version_result = _run_command([str(venv_python), "--version"], timeout=15)
            if version_result.returncode != 0:
                ok = False
                messages.append(
                    "QWEN3_TTS_VENV 실행 실패\n"
                    f"stdout:\n{version_result.stdout or '(empty)'}\n"
                    f"stderr:\n{version_result.stderr or '(empty)'}"
                )
            else:
                probe = _probe_qwen_runtime(venv_python)
                if not probe.get("qwen_tts_import"):
                    ok = False
                    messages.append(
                        "qwen-tts import 실패: "
                        f"{probe.get('qwen_tts_error', 'unknown error')}"
                    )

        if not paths["train_script"].exists():
            messages.append(
                "QWEN3_TTS_TRAIN_SCRIPT 부재: zero-shot만 사용 가능 "
                "(Qwen3-TTS 파인튜닝 워크플로우는 아직 미성숙)"
            )
        return ok, " / ".join(messages)

    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        raise NotImplementedError(
            "Qwen3-TTS 공식 fine-tuning recipe가 아직 안정화되지 않았습니다. "
            "현재는 zero-shot synthesize만 지원하며 prepare_data는 미구현입니다."
        )

    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 10,
        **kwargs: Any,
    ) -> Path:
        paths = _qwen3_paths()
        if not paths["train_script"].exists():
            raise NotImplementedError(
                "Qwen3-TTS 파인튜닝은 현재 미지원, zero-shot synthesize 사용"
            )

        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"Qwen3-TTS 학습 준비 실패: {message}")

        repo_dir = paths["dir"].expanduser().resolve()
        venv_python = paths["venv"].expanduser().resolve()
        data_dir = data_dir.expanduser().resolve()
        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)

        batch_size = int(kwargs.pop("batch_size", 4))
        timeout = int(kwargs.pop("timeout", 8 * 3600))
        train_args = kwargs.pop("train_args", None)
        extra_args: list[str] = []
        if train_args is not None:
            if not isinstance(train_args, (list, tuple)):
                raise TypeError(
                    f"train_args must be a list/tuple of CLI arguments, got {type(train_args)!r}"
                )
            extra_args.extend(str(item) for item in train_args)
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    extra_args.append(flag)
            elif value is not None:
                extra_args.extend([flag, str(value)])

        _run_and_check(
            "training",
            venv_python=venv_python,
            script=paths["train_script"],
            repo_dir=repo_dir,
            args=[
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(ckpt_out),
                "--model-path",
                str(paths["model_dir"]),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                *extra_args,
            ],
            timeout=timeout,
        )
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
            raise RuntimeError(f"Qwen3-TTS 추론 준비 실패: {message}")
        if ref_wav is None:
            raise ValueError("Qwen3-TTS zero-shot synthesize는 ref_wav가 필수입니다.")

        paths = _qwen3_paths()
        repo_dir = paths["dir"].expanduser().resolve()
        venv_python = paths["venv"].expanduser().resolve()
        model_source = ckpt.expanduser().resolve() if ckpt is not None else paths["model_dir"].expanduser().resolve()

        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sample_rate = sample_rate or self.train_sample_rate
        temp_out = out_path if target_sample_rate == self.train_sample_rate else out_path.with_suffix(".qwen3_tmp.wav")

        _run_and_check(
            "inference",
            venv_python=venv_python,
            script=paths["infer_script"],
            repo_dir=repo_dir,
            args=[
                "--model-dir",
                str(model_source),
                "--text",
                text,
                "--language",
                language,
                "--ref-wav",
                str(ref_wav.expanduser().resolve()),
                "--output",
                str(temp_out),
                "--x-vector-only",
            ],
            timeout=int(os.getenv("QWEN3_TTS_INFER_TIMEOUT", "3600")),
        )

        if not temp_out.exists():
            raise RuntimeError(
                f"Qwen3-TTS 추론 subprocess는 성공했지만 출력 WAV를 찾지 못했습니다: {temp_out}"
            )

        if target_sample_rate != self.train_sample_rate:
            if not convert_audio(temp_out, out_path, sr=target_sample_rate, channels=1):
                raise RuntimeError(
                    f"Qwen3-TTS 출력 리샘플 실패: {temp_out} -> {out_path} ({target_sample_rate}Hz)"
                )
            temp_out.unlink(missing_ok=True)
        elif temp_out != out_path:
            shutil.copy2(temp_out, out_path)
        return out_path
