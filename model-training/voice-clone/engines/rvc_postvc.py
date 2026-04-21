from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from uuid import uuid4

from . import load_engine, register
from ._manifest import load_manifest
from .base import VoiceCloneEngine, run_in_venv
from model_training.voice_clone._audio_utils import convert_audio


_SUPPORTED_UPSTREAMS = {"edge_tts", "cosyvoice2", "xtts"}
_EDGE_TTS_DEFAULT_VOICE = {
    "ko": "ko-KR-SunHiNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


def _rvc_paths() -> dict[str, Path]:
    repo_dir = _env_path("RVC_DIR", Path("~/tools/Retrieval-based-Voice-Conversion-WebUI"))
    return {
        "dir": repo_dir,
        "venv": _env_path("RVC_VENV", repo_dir / ".venv-rvc" / "bin" / "python"),
        "train_script": _env_path(
            "RVC_TRAIN_SCRIPT",
            repo_dir / "train_nsf_sim_cache_sid_load_pretrain.py",
        ),
        "infer_script": _env_path(
            "RVC_INFER_SCRIPT",
            repo_dir / "tools" / "infer_cli.py",
        ),
        "weight_root": repo_dir / "assets" / "weight_root",
    }


def _run_command(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
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
            f"RVC post-VC {stage_name} 단계 실패: {script}\n"
            f"args: {' '.join(args)}\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}"
        )
    return result


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


def _normalize_upstream(name: str | None = None) -> str:
    candidate = (name or os.getenv("RVC_UPSTREAM_ENGINE", "edge_tts")).strip().lower()
    if candidate not in _SUPPORTED_UPSTREAMS:
        supported = ", ".join(sorted(_SUPPORTED_UPSTREAMS))
        raise ValueError(
            f"Unsupported RVC_UPSTREAM_ENGINE '{candidate}'. Supported values: {supported}"
        )
    return candidate


def _resolve_edge_tts_voice(language: str) -> str:
    override = os.getenv("RVC_UPSTREAM_VOICE")
    if override:
        return override
    lang_key = language.strip().lower().split("-", 1)[0]
    return _EDGE_TTS_DEFAULT_VOICE.get(lang_key, _EDGE_TTS_DEFAULT_VOICE["ko"])


def _has_edge_tts() -> bool:
    return importlib.util.find_spec("edge_tts") is not None


def _resolve_rvc_artifacts(ckpt: Path) -> tuple[Path, Path | None]:
    ckpt_path = ckpt.expanduser().resolve()
    search_root = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent

    if ckpt_path.is_file() and ckpt_path.suffix.lower() == ".pth":
        model_path = ckpt_path
    else:
        model_candidates = sorted(search_root.rglob("G_*.pth"))
        if not model_candidates:
            model_candidates = sorted(
                path
                for path in search_root.rglob("*.pth")
                if path.is_file()
            )
        if not model_candidates:
            raise FileNotFoundError(f"RVC checkpoint(.pth)를 찾을 수 없습니다: {ckpt_path}")
        model_path = model_candidates[0]

    index_candidates = sorted(search_root.rglob("index_*.index"))
    if not index_candidates:
        index_candidates = sorted(search_root.rglob("*.index"))
    return model_path, index_candidates[0] if index_candidates else None


def _find_training_outputs(ckpt_out: Path, speaker: str) -> tuple[Path, Path | None]:
    preferred_model = ckpt_out / f"G_{speaker}.pth"
    if preferred_model.exists():
        preferred_index = ckpt_out / f"index_{speaker}.index"
        return preferred_model, preferred_index if preferred_index.exists() else None

    model_candidates = sorted(ckpt_out.rglob("G_*.pth"))
    if not model_candidates:
        model_candidates = sorted(ckpt_out.rglob("*.pth"))
    if not model_candidates:
        raise FileNotFoundError(f"RVC 학습 산출물(.pth)을 찾지 못했습니다: {ckpt_out}")

    index_candidates = sorted(ckpt_out.rglob("index_*.index"))
    if not index_candidates:
        index_candidates = sorted(ckpt_out.rglob("*.index"))
    return model_candidates[0], index_candidates[0] if index_candidates else None


def _normalize_extra_args(value: Any, arg_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"{arg_name} must be a list/tuple of CLI arguments, got {type(value)!r}")


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@register("rvc_postvc")
class RVCPostVCEngine(VoiceCloneEngine):
    name = "rvc_postvc"
    train_sample_rate = 40000
    supports_language = ["ko", "en", "ja", "zh"]
    license = "MIT"
    phase = "phase_c"
    hardware_pref = "cuda"

    def health_check(self) -> tuple[bool, str]:
        paths = _rvc_paths()
        repo_dir = paths["dir"]
        venv_python = paths["venv"]
        upstream = _normalize_upstream()
        messages: list[str] = []
        ok = True

        if not repo_dir.exists():
            ok = False
            messages.append(
                f"RVC_DIR 없음: {repo_dir}. "
                "예: git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI ~/tools/Retrieval-based-Voice-Conversion-WebUI"
            )
        if not venv_python.exists():
            ok = False
            messages.append(
                f"RVC_VENV 없음: {venv_python}. "
                "예: python -m venv ~/tools/Retrieval-based-Voice-Conversion-WebUI/.venv-rvc"
            )

        for key in ("train_script", "infer_script"):
            if repo_dir.exists() and not paths[key].exists():
                ok = False
                env_name = "RVC_TRAIN_SCRIPT" if key == "train_script" else "RVC_INFER_SCRIPT"
                messages.append(f"{env_name} 없음: {paths[key]}")

        if upstream == "edge_tts":
            if not _has_edge_tts():
                ok = False
                messages.append(
                    "edge-tts 패키지 없음. "
                    "예: pip install edge-tts 또는 RVC_UPSTREAM_ENGINE=cosyvoice2/xtts"
                )
        else:
            try:
                upstream_engine = load_engine(upstream)
                upstream_ok, upstream_message = upstream_engine.health_check()
            except Exception as exc:
                upstream_ok = False
                upstream_message = str(exc)
            if not upstream_ok:
                ok = False
                messages.append(f"upstream {upstream} 준비 안 됨: {upstream_message}")

        messages.append(f"upstream={upstream}")
        return ok, " / ".join(messages)

    def prepare_data(self, manifest_path: Path, out_dir: Path) -> Path:
        manifest = load_manifest(manifest_path)
        speaker = str(manifest["speaker"])
        out_dir = out_dir.expanduser().resolve()
        wavs_dir = out_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        filelist_lines: list[str] = []
        sample_ids: list[str] = []
        for sample in manifest.get("samples", []):
            sample_id = str(sample["id"])
            src = _resolve_audio_path(manifest_path, sample, "wav_path_24k")
            dst = wavs_dir / f"{sample_id}.wav"
            if not convert_audio(src, dst, sr=self.train_sample_rate, channels=1):
                raise RuntimeError(
                    f"RVC 학습용 오디오 변환 실패: {src} -> {dst} ({self.train_sample_rate}Hz)"
                )
            filelist_lines.append(f"{dst.resolve()}|{speaker}|{sample_id}")
            sample_ids.append(sample_id)

        filelist_path = out_dir / "filelist.txt"
        filelist_path.write_text("\n".join(filelist_lines) + "\n", encoding="utf-8")
        _write_metadata(
            out_dir / "metadata.json",
            {
                "speaker": speaker,
                "train_sample_rate": self.train_sample_rate,
                "filelist": str(filelist_path),
                "sample_ids": sample_ids,
            },
        )
        return out_dir

    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 20,
        **kwargs: Any,
    ) -> Path:
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"RVC 학습 준비 실패: {message}")

        paths = _rvc_paths()
        repo_dir = paths["dir"].expanduser().resolve()
        venv_python = paths["venv"].expanduser().resolve()
        data_dir = data_dir.expanduser().resolve()
        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)

        metadata_path = data_dir / "metadata.json"
        speaker = str(kwargs.pop("speaker", "speaker"))
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                speaker = str(payload.get("speaker") or speaker)
            except json.JSONDecodeError:
                pass

        filelist_path = data_dir / "filelist.txt"
        if not filelist_path.exists():
            raise FileNotFoundError(f"RVC filelist를 찾을 수 없습니다: {filelist_path}")

        # RVC 공식 train script는 logs/<exp>/... 구조를 기대하는 경우가 많아 실험 디렉토리를 맞춘다.
        exp_dir = repo_dir / "logs" / speaker
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        gt_wavs_dir.mkdir(parents=True, exist_ok=True)
        for wav_path in sorted((data_dir / "wavs").glob("*.wav")):
            _copy_or_link(wav_path, gt_wavs_dir / wav_path.name)
        shutil.copy2(filelist_path, exp_dir / "filelist.txt")

        batch_size = int(kwargs.pop("batch_size", 4))
        save_every_epoch = int(kwargs.pop("save_every_epoch", 5))
        f0 = int(kwargs.pop("f0", 1))
        g_pretrained = kwargs.pop(
            "g_pretrained",
            repo_dir / "pretrained_v2" / "f0G40k.pth",
        )
        d_pretrained = kwargs.pop(
            "d_pretrained",
            repo_dir / "pretrained_v2" / "f0D40k.pth",
        )
        version = str(kwargs.pop("version", "v2"))
        timeout = int(kwargs.pop("timeout", 8 * 3600))
        train_args = _normalize_extra_args(kwargs.pop("train_args", None), "train_args")

        passthrough_flags: list[str] = []
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    passthrough_flags.append(flag)
            elif value is not None:
                passthrough_flags.extend([flag, str(value)])

        _run_and_check(
            "training",
            venv_python=venv_python,
            script=paths["train_script"],
            repo_dir=repo_dir,
            args=[
                "-e",
                speaker,
                "-sr",
                "40k",
                "-f0",
                str(f0),
                "-bs",
                str(batch_size),
                "-te",
                str(epochs),
                "-se",
                str(save_every_epoch),
                "-pg",
                str(Path(g_pretrained).expanduser().resolve()),
                "-pd",
                str(Path(d_pretrained).expanduser().resolve()),
                "-l",
                "0",
                "-c",
                "0",
                "-sw",
                "0",
                "-v",
                version,
                *train_args,
                *passthrough_flags,
            ],
            timeout=timeout,
        )

        model_path, index_path = _find_training_outputs(ckpt_out if any(ckpt_out.iterdir()) else exp_dir, speaker)
        target_model = ckpt_out / f"G_{speaker}.pth"
        target_index = ckpt_out / f"index_{speaker}.index"
        if model_path != target_model:
            shutil.copy2(model_path, target_model)
        if index_path is not None and index_path != target_index:
            shutil.copy2(index_path, target_index)
        return ckpt_out

    def synthesize(
        self,
        text: str,
        ckpt: Path | None,
        ref_wav: Path | None = None,
        language: str = "ko",
        out_path: Path = Path("out.wav"),
        sample_rate: int | None = None,
    ) -> Path:
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"RVC post-VC 추론 준비 실패: {message}")
        if ckpt is None:
            raise ValueError("RVC post-VC는 학습된 RVC checkpoint가 필요합니다.")

        paths = _rvc_paths()
        repo_dir = paths["dir"].expanduser().resolve()
        venv_python = paths["venv"].expanduser().resolve()
        model_path, index_path = _resolve_rvc_artifacts(ckpt)

        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sample_rate = sample_rate or self.train_sample_rate

        with tempfile.TemporaryDirectory(prefix="rvc-postvc-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            upstream_wav = temp_dir / "upstream.wav"
            converted_wav = out_path if target_sample_rate == self.train_sample_rate else temp_dir / "converted.wav"
            self._synthesize_upstream(
                text=text,
                ref_wav=ref_wav,
                language=language,
                out_path=upstream_wav,
            )

            if not upstream_wav.exists():
                raise RuntimeError(
                    f"RVC upstream TTS는 성공했지만 출력 WAV를 찾지 못했습니다: {upstream_wav}"
                )

            temp_model_name = f"rvc-postvc-{uuid4().hex}.pth"
            temp_model_path = paths["weight_root"] / temp_model_name
            _copy_or_link(model_path, temp_model_path)
            try:
                infer_args = [
                    "--input_path",
                    str(upstream_wav),
                    "--opt_path",
                    str(converted_wav),
                    "--model_name",
                    temp_model_name,
                    "--index_path",
                    str(index_path) if index_path is not None else "",
                    "--f0up_key",
                    str(0),
                    "--f0method",
                    os.getenv("RVC_F0_METHOD", "rmvpe"),
                    "--index_rate",
                    str(float(os.getenv("RVC_INDEX_RATE", "0.66"))),
                    "--filter_radius",
                    str(int(os.getenv("RVC_FILTER_RADIUS", "3"))),
                    "--resample_sr",
                    str(0),
                    "--rms_mix_rate",
                    str(float(os.getenv("RVC_RMS_MIX_RATE", "1.0"))),
                    "--protect",
                    str(float(os.getenv("RVC_PROTECT", "0.33"))),
                    "--device",
                    os.getenv("RVC_DEVICE", self.hardware_pref),
                    "--is_half",
                    os.getenv("RVC_IS_HALF", "False"),
                ]
                _run_and_check(
                    "inference",
                    venv_python=venv_python,
                    script=paths["infer_script"],
                    repo_dir=repo_dir,
                    args=infer_args,
                    timeout=int(os.getenv("RVC_INFER_TIMEOUT", "1800")),
                )
            finally:
                temp_model_path.unlink(missing_ok=True)

            if not converted_wav.exists():
                raise RuntimeError(
                    f"RVC infer subprocess는 성공했지만 출력 WAV를 찾지 못했습니다: {converted_wav}"
                )
            if target_sample_rate != self.train_sample_rate:
                if not convert_audio(converted_wav, out_path, sr=target_sample_rate, channels=1):
                    raise RuntimeError(
                        f"RVC 출력 리샘플 실패: {converted_wav} -> {out_path} ({target_sample_rate}Hz)"
                    )
            return out_path

    def _synthesize_upstream(
        self,
        text: str,
        ref_wav: Path | None,
        language: str,
        out_path: Path,
    ) -> Path:
        upstream = _normalize_upstream()
        if upstream == "edge_tts":
            return self._synthesize_with_edge_tts(text=text, language=language, out_path=out_path)

        upstream_engine = load_engine(upstream)
        if ref_wav is None:
            raise ValueError(
                f"RVC_UPSTREAM_ENGINE={upstream} 사용 시 ref_wav가 필요합니다."
            )
        return upstream_engine.synthesize(
            text=text,
            ckpt=None,
            ref_wav=ref_wav,
            language=language,
            out_path=out_path,
            sample_rate=self.train_sample_rate,
        )

    def _synthesize_with_edge_tts(
        self,
        text: str,
        language: str,
        out_path: Path,
    ) -> Path:
        voice = _resolve_edge_tts_voice(language)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            temp_media = Path(handle.name)
        try:
            result = _run_command(
                [
                    sys.executable,
                    "-m",
                    "edge_tts",
                    "--text",
                    text,
                    "--voice",
                    voice,
                    "--write-media",
                    str(temp_media),
                ],
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "edge-tts 합성 실패\n"
                    f"voice: {voice}\n"
                    f"stdout:\n{result.stdout or '(empty)'}\n"
                    f"stderr:\n{result.stderr or '(empty)'}"
                )
            if not temp_media.exists():
                raise RuntimeError(f"edge-tts 출력 파일을 찾지 못했습니다: {temp_media}")
            if not convert_audio(temp_media, out_path, sr=self.train_sample_rate, channels=1):
                raise RuntimeError(
                    f"edge-tts 출력 WAV 변환 실패: {temp_media} -> {out_path}"
                )
            return out_path
        finally:
            temp_media.unlink(missing_ok=True)
