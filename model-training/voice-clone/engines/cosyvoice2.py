from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from . import register
from ._manifest import load_manifest, split_samples
from .base import VoiceCloneEngine, run_in_venv
from model_training.voice_clone.scripts.normalize_korean import normalize


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


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


def _cosyvoice_paths() -> dict[str, Path]:
    cosy_dir = _env_path("COSYVOICE_DIR", Path("~/tools/CosyVoice"))
    return {
        "dir": cosy_dir,
        "venv": _env_path("COSYVOICE_VENV", cosy_dir / ".venv-cosy" / "bin" / "python"),
        "model_dir": _env_path(
            "COSYVOICE_MODEL_DIR",
            cosy_dir / "pretrained_models" / "CosyVoice2-0.5B",
        ),
        "train_script": _env_path(
            "COSYVOICE_TRAIN_SCRIPT",
            cosy_dir / "examples" / "cosyvoice2" / "ft_spk.py",
        ),
        "infer_script": _env_path(
            "COSYVOICE_INFER_SCRIPT",
            cosy_dir / "examples" / "cosyvoice2" / "inference_sft.py",
        ),
    }


def _run_and_check(
    venv_python: Path,
    script: Path,
    args: list[str],
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    result = run_in_venv(venv_python, script, args=args, env=env, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"CosyVoice2 subprocess 실패: {script}\n"
            f"args: {' '.join(args)}\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}\n"
            "힌트: COSYVOICE_VENV, COSYVOICE_DIR, script 경로와 codec/venv 설정을 확인하세요."
        )
    return result


@register("cosyvoice2")
class CosyVoice2Engine(VoiceCloneEngine):
    name = "cosyvoice2"
    train_sample_rate = 24000
    supports_language = ["ko", "en", "zh", "ja", "yue"]
    license = "Apache-2.0"
    phase = "phase_a"
    hardware_pref = "cuda"

    def health_check(self) -> tuple[bool, str]:
        paths = _cosyvoice_paths()
        missing: list[str] = []

        if not paths["dir"].exists():
            missing.append(
                "COSYVOICE_DIR 없음. "
                "예: git clone https://github.com/FunAudioLLM/CosyVoice ~/tools/CosyVoice"
            )
        if not paths["venv"].exists():
            missing.append(
                "COSYVOICE_VENV 없음. "
                "예: python -m venv ~/tools/CosyVoice/.venv-cosy && "
                "~/tools/CosyVoice/.venv-cosy/bin/pip install -r requirements.txt"
            )
        if not paths["model_dir"].exists():
            missing.append(
                "COSYVOICE_MODEL_DIR 없음. "
                "예: pretrained_models/CosyVoice2-0.5B 다운로드 후 환경변수 설정"
            )
        if missing:
            return False, " / ".join(missing)

        hints: list[str] = [f"repo={paths['dir']}", f"venv={paths['venv']}", f"model={paths['model_dir']}"]
        if not paths["train_script"].exists():
            hints.append(f"train script missing: {paths['train_script']} (COSYVOICE_TRAIN_SCRIPT로 override 가능)")
        if not paths["infer_script"].exists():
            hints.append(f"infer script missing: {paths['infer_script']} (COSYVOICE_INFER_SCRIPT로 override 가능)")
        return True, "; ".join(hints)

    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        manifest = load_manifest(manifest_path)
        speaker = str(manifest["speaker"])
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = out_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "eval"):
            split_dir = out_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            wav_scp_lines: list[str] = []
            utt2text_lines: list[str] = []
            utt2spk_lines: list[str] = []
            utt_ids: list[str] = []

            for sample in split_samples(manifest, split_name):
                utt_id = str(sample["id"])
                src = _resolve_audio_path(manifest_path, sample, "wav_path_24k")
                dst = wavs_dir / f"{utt_id}.wav"
                _copy_or_link(src, dst)
                normalized_text = str(sample.get("text_normalized") or sample.get("text") or "").strip()
                wav_scp_lines.append(f"{utt_id} {dst}")
                utt2text_lines.append(f"{utt_id} {normalized_text}")
                utt2spk_lines.append(f"{utt_id} {speaker}")
                utt_ids.append(utt_id)

            (split_dir / "wav.scp").write_text("\n".join(wav_scp_lines) + "\n", encoding="utf-8")
            (split_dir / "utt2text").write_text("\n".join(utt2text_lines) + "\n", encoding="utf-8")
            (split_dir / "utt2spk").write_text("\n".join(utt2spk_lines) + "\n", encoding="utf-8")
            (split_dir / "spk2utt").write_text(f"{speaker} {' '.join(utt_ids)}\n", encoding="utf-8")

        return out_dir

    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 50,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> Path:
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"CosyVoice2 학습 준비 실패: {message}")

        paths = _cosyvoice_paths()
        if not paths["train_script"].exists():
            raise FileNotFoundError(
                f"CosyVoice2 train script를 찾을 수 없습니다: {paths['train_script']} "
                "(COSYVOICE_TRAIN_SCRIPT로 override 가능)"
            )

        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)
        script_args = [
            "--train-data",
            str((data_dir / "train").resolve()),
            "--eval-data",
            str((data_dir / "eval").resolve()),
            "--pretrained-model-dir",
            str(paths["model_dir"]),
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
                    script_args.append(flag)
            else:
                script_args.extend([flag, str(value)])

        _run_and_check(
            paths["venv"],
            paths["train_script"],
            args=script_args,
            env={"COSYVOICE_MODEL_DIR": str(paths["model_dir"])},
            timeout=int(kwargs.get("timeout", 6 * 3600)),
        )

        checkpoints = sorted(
            path
            for path in ckpt_out.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
        )
        if not checkpoints:
            raise RuntimeError(f"CosyVoice2 학습은 끝났지만 checkpoint를 찾지 못했습니다: {ckpt_out}")
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
            raise RuntimeError(f"CosyVoice2 추론 준비 실패: {message}")

        paths = _cosyvoice_paths()
        if not paths["infer_script"].exists():
            raise FileNotFoundError(
                f"CosyVoice2 inference script를 찾을 수 없습니다: {paths['infer_script']} "
                "(COSYVOICE_INFER_SCRIPT로 override 가능)"
            )

        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sample_rate = sample_rate or self.train_sample_rate
        temp_out = out_path if target_sample_rate == self.train_sample_rate else out_path.with_suffix(".cosyvoice_tmp.wav")
        normalized_text = normalize(text)

        model_source = ckpt.expanduser().resolve() if ckpt is not None else paths["model_dir"]
        if ckpt is None and ref_wav is None:
            raise ValueError("CosyVoice2 zero-shot/베이스라인 추론은 ref_wav가 필요합니다.")

        script_args = [
            "--model-dir",
            str(paths["model_dir"]),
            "--checkpoint",
            str(model_source),
            "--text",
            normalized_text,
            "--language",
            language,
            "--output",
            str(temp_out),
        ]
        if ref_wav is not None:
            script_args.extend(["--prompt-wav", str(ref_wav)])

        _run_and_check(
            paths["venv"],
            paths["infer_script"],
            args=script_args,
            env={"COSYVOICE_MODEL_DIR": str(paths["model_dir"])},
            timeout=1800,
        )

        if not temp_out.exists():
            raise RuntimeError(
                f"CosyVoice2 추론 subprocess는 성공했지만 출력 WAV를 찾지 못했습니다: {temp_out}"
            )
        if target_sample_rate != self.train_sample_rate:
            _convert_audio_or_raise(temp_out, out_path, sample_rate=target_sample_rate)
            temp_out.unlink(missing_ok=True)
        return out_path
