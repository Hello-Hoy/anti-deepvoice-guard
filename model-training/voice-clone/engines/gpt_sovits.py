"""GPT-SoVITS engine adapter.

Drives upstream `RVC-Boss/GPT-SoVITS` via its actual interface:

- Prep: 3 env-var-driven scripts (1-get-text, 2-get-hubert-wav32k, 3-get-semantic).
- S2 (SoVITS) training: `GPT_SoVITS/s2_train.py --config <tmp>.json` built from
  `GPT_SoVITS/configs/s2.json` with our per-run overrides applied.
- S1 (GPT) training: `GPT_SoVITS/s1_train.py --config_file <tmp>.yaml` built from
  `GPT_SoVITS/configs/s1longer-v2.yaml` with our per-run overrides applied.
- Inference: our own Korean-capable runner (upstream `inference_cli.py` only
  knows zh/en/ja) in `scripts/gpt_sovits_inference_runner.py`.

Environment overrides:
- `GPT_SOVITS_DIR`: upstream repo root.
- `GPT_SOVITS_VENV`: python inside the repo's venv (CUDA+upstream deps).
- `GPT_SOVITS_VERSION`: v1 | v2 | v2Pro | v2ProPlus | v3 | v4. Default `v2`
  (Korean-capable, stable).
- `GPT_SOVITS_PRETRAINED_DIR`: override for pretrained model root; defaults to
  `{GPT_SOVITS_DIR}/GPT_SoVITS/pretrained_models`.
- `GPT_SOVITS_*_SCRIPT`: override individual script paths.
- `GPT_SOVITS_*_CONFIG_TEMPLATE`: override s1/s2 config YAML/JSON.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # optional at import time; required at train() time

from . import register
from ._manifest import load_manifest, split_samples
from .base import VoiceCloneEngine, run_in_venv


def convert_audio(*args: Any, **kwargs: Any) -> bool:
    from model_training.voice_clone._audio_utils import convert_audio as _convert_audio

    return _convert_audio(*args, **kwargs)


_PRETRAINED_V2 = {
    "bert": "chinese-roberta-wwm-ext-large",
    "cnhubert": "chinese-hubert-base",
    "s2G": "gsv-v2final-pretrained/s2G2333k.pth",
    "s2D": "gsv-v2final-pretrained/s2D2333k.pth",
    "s1": "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "s2_config": "configs/s2.json",
    "s1_config": "configs/s1longer-v2.yaml",
}

_PRETRAINED_V2PRO = {
    **_PRETRAINED_V2,
    "s2G": "v2Pro/s2Gv2Pro.pth",
    "s2D": "v2Pro/s2Dv2Pro.pth",
    "s2_config": "configs/s2v2Pro.json",
    "s1": "s1v3.ckpt",
    "sv": "sv/pretrained_eres2netv2w24s4ep4.ckpt",
}

_PRETRAINED_V2PROPLUS = {
    **_PRETRAINED_V2PRO,
    "s2G": "v2Pro/s2Gv2ProPlus.pth",
    "s2D": "v2Pro/s2Dv2ProPlus.pth",
    "s2_config": "configs/s2v2ProPlus.json",
    "bigvgan": "models--nvidia--bigvgan_v2_24khz_100band_256x",
}

_PRETRAINED_V1 = {
    **_PRETRAINED_V2,
    "s2G": "s2G488k.pth",
    "s2D": "s2D488k.pth",
    "s1": "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "s1_config": "configs/s1longer.yaml",
}

_PRETRAINED_BY_VERSION = {
    "v1": _PRETRAINED_V1,
    "v2": _PRETRAINED_V2,
    "v2Pro": _PRETRAINED_V2PRO,
    "v2ProPlus": _PRETRAINED_V2PROPLUS,
}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


def _gpt_sovits_paths() -> dict[str, Any]:
    repo_dir = _env_path("GPT_SOVITS_DIR", Path("~/tools/GPT-SoVITS"))
    version = os.getenv("GPT_SOVITS_VERSION", "v2")
    pretrained_root = _env_path(
        "GPT_SOVITS_PRETRAINED_DIR",
        repo_dir / "GPT_SoVITS" / "pretrained_models",
    )
    defaults = _PRETRAINED_BY_VERSION.get(version, _PRETRAINED_V2)

    gpt_sovits_pkg = repo_dir / "GPT_SoVITS"
    return {
        "dir": repo_dir,
        "venv": _env_path(
            "GPT_SOVITS_VENV",
            repo_dir / ".venv-gpt-sovits" / "Scripts" / "python.exe",
        ),
        "version": version,
        "commit": os.getenv("GPT_SOVITS_COMMIT", ""),
        "pretrained_root": pretrained_root,
        "bert_dir": _env_path(
            "GPT_SOVITS_BERT_DIR",
            pretrained_root / defaults["bert"],
        ),
        "cnhubert_dir": _env_path(
            "GPT_SOVITS_CNHUBERT_DIR",
            pretrained_root / defaults["cnhubert"],
        ),
        "pretrained_s2G": _env_path(
            "GPT_SOVITS_PRETRAINED_S2G",
            pretrained_root / defaults["s2G"],
        ),
        "pretrained_s2D": _env_path(
            "GPT_SOVITS_PRETRAINED_S2D",
            pretrained_root / defaults["s2D"],
        ),
        "pretrained_s1": _env_path(
            "GPT_SOVITS_PRETRAINED_S1",
            pretrained_root / defaults["s1"],
        ),
        "sv_ckpt": (
            _env_path("GPT_SOVITS_SV_CKPT", pretrained_root / defaults["sv"])
            if "sv" in defaults
            else None
        ),
        "bigvgan_dir": (
            _env_path("GPT_SOVITS_BIGVGAN_DIR", pretrained_root / defaults["bigvgan"])
            if "bigvgan" in defaults
            else None
        ),
        "s2_config_template": _env_path(
            "GPT_SOVITS_S2_CONFIG_TEMPLATE",
            gpt_sovits_pkg / defaults["s2_config"],
        ),
        "s1_config_template": _env_path(
            "GPT_SOVITS_S1_CONFIG_TEMPLATE",
            gpt_sovits_pkg / defaults["s1_config"],
        ),
        "script_1a": _env_path(
            "GPT_SOVITS_SCRIPT_1A",
            gpt_sovits_pkg / "prepare_datasets" / "1-get-text.py",
        ),
        "script_1b": _env_path(
            "GPT_SOVITS_SCRIPT_1B",
            gpt_sovits_pkg / "prepare_datasets" / "2-get-hubert-wav32k.py",
        ),
        "script_1sv": (
            _env_path(
                "GPT_SOVITS_SCRIPT_1SV",
                gpt_sovits_pkg / "prepare_datasets" / "2-get-sv.py",
            )
            if "sv" in defaults
            else None
        ),
        "script_1c": _env_path(
            "GPT_SOVITS_SCRIPT_1C",
            gpt_sovits_pkg / "prepare_datasets" / "3-get-semantic.py",
        ),
        "s2_train_script": _env_path(
            "GPT_SOVITS_S2_TRAIN_SCRIPT",
            gpt_sovits_pkg / "s2_train.py",
        ),
        "s1_train_script": _env_path(
            "GPT_SOVITS_S1_TRAIN_SCRIPT",
            gpt_sovits_pkg / "s1_train.py",
        ),
        "inference_runner": _env_path(
            "GPT_SOVITS_INFERENCE_RUNNER",
            Path(__file__).resolve().parents[1] / "scripts" / "gpt_sovits_inference_runner.py",
        ),
    }


def _install_hint(paths: dict[str, Any]) -> str:
    repo_dir = paths["dir"]
    venv_python = paths["venv"]
    return (
        "설치 힌트: "
        f"git clone https://github.com/RVC-Boss/GPT-SoVITS {repo_dir} && "
        f"cd {repo_dir} && python -m venv .venv-gpt-sovits && "
        f"{venv_python} -m pip install -r requirements.txt 후 "
        f"pretrained_models.zip 을 {paths['pretrained_root']} 에 풀어두세요."
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


def _probe_torch_runtime(venv_python: Path) -> dict[str, Any]:
    probe = subprocess.run(
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
                "    payload['torch_version'] = getattr(torch, '__version__', 'unknown')\n"
                "except Exception as exc:\n"
                "    payload['torch_import'] = False\n"
                "    payload['error'] = repr(exc)\n"
                "print(json.dumps(payload, ensure_ascii=False))\n"
            ),
        ],
        capture_output=True,
        text=True,
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


_WINDOWS_UTF8_ENV = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONLEGACYWINDOWSSTDIO": "utf-8",
}


def _stream_subprocess(
    args: list[str],
    env: dict[str, str] | None,
    cwd: Path,
    timeout: int,
    stage_name: str,
    success_paths: list[Path] | None = None,
) -> None:
    """Run a subprocess with inherited stdout/stderr so long-running train logs stream live.

    If `success_paths` are given and all exist after the process exits, a non-zero
    exit is treated as a recoverable cleanup-error (e.g. pytorch_lightning + rich
    on Windows can throw `UnicodeEncodeError` from its progress-bar teardown
    AFTER checkpoints are saved — we don't want to fail the whole pipeline for
    that cosmetic issue).
    """
    merged_env = os.environ.copy()
    merged_env.update(_WINDOWS_UTF8_ENV)
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    try:
        result = subprocess.run(
            args,
            env=merged_env,
            cwd=str(cwd),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"GPT-SoVITS {stage_name} timed out after {timeout}s: {' '.join(args)}"
        ) from exc
    if result.returncode == 0:
        return
    if success_paths and all(p.exists() for p in success_paths):
        print(
            f"WARNING: GPT-SoVITS {stage_name} exited with code {result.returncode} "
            f"but expected artefacts are present: {[str(p) for p in success_paths]}. "
            "Treating as success (likely progress-bar/console cleanup error)."
        )
        return
    raise RuntimeError(
        f"GPT-SoVITS {stage_name} failed (exit={result.returncode}): {' '.join(args)}"
    )


def _copy_or_raise(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"복사할 파일이 존재하지 않습니다: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _find_latest(pattern_dir: Path, glob: str) -> Path:
    candidates = sorted(pattern_dir.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No file matching '{glob}' under {pattern_dir}")
    return candidates[0]


@register("gpt_sovits")
class GPTSoVITSEngine(VoiceCloneEngine):
    name = "gpt_sovits"
    train_sample_rate = 32000
    supports_language = ["ko", "en", "zh", "ja", "yue"]
    license = "MIT"
    phase = "phase_b"
    hardware_pref = "cuda_required"

    # --- health_check ---------------------------------------------------

    def health_check(self) -> tuple[bool, str]:
        paths = _gpt_sovits_paths()
        install_hint = _install_hint(paths)
        repo_dir = paths["dir"]
        venv_python = paths["venv"]

        if not repo_dir.exists():
            return False, f"GPT_SOVITS_DIR 없음: {repo_dir}. {install_hint}"
        if not venv_python.exists():
            return False, f"GPT_SOVITS_VENV 없음: {venv_python}. {install_hint}"

        version_probe = subprocess.run(
            [str(venv_python), "--version"],
            capture_output=True, text=True, timeout=15,
        )
        if version_probe.returncode != 0:
            return (
                False,
                f"GPT_SOVITS_VENV 실행 실패: {venv_python}\n"
                f"stderr: {version_probe.stderr or '(empty)'}\n{install_hint}",
            )

        script_keys = [
            "script_1a", "script_1b", "script_1c",
            "s2_train_script", "s1_train_script",
            "s2_config_template", "s1_config_template",
            "inference_runner",
        ]
        if paths["version"] in {"v2Pro", "v2ProPlus"}:
            script_keys.append("script_1sv")
        missing_scripts = [k for k in script_keys if not Path(paths[k]).exists()]
        if missing_scripts:
            desc = ", ".join(f"{k}={paths[k]}" for k in missing_scripts)
            return False, f"GPT-SoVITS 파일 없음: {desc}. {install_hint}"

        pretrained_keys = ("bert_dir", "cnhubert_dir", "pretrained_s2G", "pretrained_s2D", "pretrained_s1")
        missing_pretrained = [k for k in pretrained_keys if not Path(paths[k]).exists()]
        if missing_pretrained:
            desc = ", ".join(f"{k}={paths[k]}" for k in missing_pretrained)
            return False, (
                f"GPT-SoVITS 사전학습 모델 없음: {desc}. "
                f"pretrained_models.zip 을 {paths['pretrained_root']} 에 풀어두세요. {install_hint}"
            )

        if paths["version"] in {"v2Pro", "v2ProPlus"}:
            sv_ckpt: Optional[Path] = paths["sv_ckpt"]
            if sv_ckpt is None or not sv_ckpt.is_file():
                return False, f"GPT-SoVITS SV ckpt missing: {sv_ckpt}. {install_hint}"
        if paths["version"] == "v2ProPlus":
            bigvgan_dir: Optional[Path] = paths["bigvgan_dir"]
            if bigvgan_dir is None or not bigvgan_dir.is_dir():
                return False, f"GPT-SoVITS BigVGAN dir missing: {bigvgan_dir}. {install_hint}"

        probe = _probe_torch_runtime(venv_python)
        if not probe.get("torch_import"):
            return False, f"GPT-SoVITS venv torch import 실패: {probe.get('error', 'unknown')}. {install_hint}"
        if not probe.get("cuda_available"):
            return False, (
                "GPT-SoVITS는 CUDA 필수 경로입니다. torch.cuda.is_available()=False."
            )

        return True, (
            f"repo={repo_dir}; venv={venv_python}; "
            f"python={version_probe.stdout.strip() or version_probe.stderr.strip()}; "
            f"torch={probe.get('torch_version', 'unknown')}; cuda=available; "
            f"version={paths['version']}"
        )

    # --- prepare_data ---------------------------------------------------

    def prepare_data(
        self,
        manifest_path: Path,
        out_dir: Path,
        min_snr_db: float | None = 25.0,
    ) -> Path:
        manifest = load_manifest(manifest_path)
        speaker = str(manifest["speaker"])
        out_dir = out_dir.expanduser().resolve()
        raw_dir = out_dir / "raw" / speaker
        lists_dir = out_dir / "lists"
        raw_dir.mkdir(parents=True, exist_ok=True)
        lists_dir.mkdir(parents=True, exist_ok=True)

        ref_map: dict[str, str] = {}
        train_lines: list[str] = []
        val_lines: list[str] = []
        split_buckets = {"train": train_lines, "eval": val_lines}
        skipped_low_snr: list[str] = []
        skipped_missing_snr: list[str] = []

        for split_name, lines in split_buckets.items():
            for sample in split_samples(manifest, split_name):
                sample_id = str(sample["id"])
                snr_db = sample.get("snr_db")
                if min_snr_db is not None:
                    if snr_db is None:
                        skipped_missing_snr.append(sample_id)
                    else:
                        try:
                            snr_value = float(snr_db)
                        except (TypeError, ValueError):
                            skipped_missing_snr.append(sample_id)
                        else:
                            if snr_value < min_snr_db:
                                skipped_low_snr.append(sample_id)
                                continue
                src = _resolve_audio_path(manifest_path, sample, "wav_path_24k")
                dst = raw_dir / f"{sample_id}.wav"
                if not convert_audio(src, dst, sr=self.train_sample_rate, channels=1):
                    raise RuntimeError(
                        f"GPT-SoVITS 학습용 오디오 변환 실패: {src} -> {dst} "
                        f"({self.train_sample_rate}Hz)"
                    )
                language = str(sample.get("language") or "ko").strip() or "ko"
                text = str(sample.get("text") or "").strip()
                if not text:
                    raise ValueError(f"Sample '{sample_id}' is missing non-empty text.")
                # Upstream prep scripts parse `wav_name|spk|lang|text` and
                # use `os.path.basename(wav_name)` against inp_wav_dir.
                lines.append(f"{dst.name}|{speaker}|{language}|{text}")
                ref_map[dst.name] = text
                ref_map[str(dst.resolve())] = text

        (lists_dir / "train.list").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        (lists_dir / "val.list").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
        # Upstream trains on a single list; we feed the union (eval samples do
        # not hurt single-speaker adaptation and broaden phoneme coverage).
        combined = train_lines + val_lines
        (out_dir / "list.txt").write_text("\n".join(combined) + "\n", encoding="utf-8")

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
        (out_dir / "_meta.json").write_text(
            json.dumps(
                {
                    "speaker": speaker,
                    "train_count": len(train_lines),
                    "eval_count": len(val_lines),
                    "raw_wav_dir": str(raw_dir),
                    "list_file": str((out_dir / "list.txt").resolve()),
                    "min_snr_db": min_snr_db,
                    "skipped_low_snr": skipped_low_snr,
                    "skipped_missing_snr": skipped_missing_snr,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return out_dir

    # --- train ----------------------------------------------------------

    def train(
        self,
        data_dir: Path,
        ckpt_out: Path,
        epochs: int = 15,
        **kwargs: Any,
    ) -> Path:
        if yaml is None:
            raise RuntimeError("PyYAML이 필요합니다: pip install pyyaml")
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(f"GPT-SoVITS 학습 준비 실패: {message}")

        paths = _gpt_sovits_paths()
        repo_dir: Path = paths["dir"]
        venv_python: Path = paths["venv"]
        version: str = paths["version"]
        data_dir = data_dir.expanduser().resolve()
        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)

        list_file = data_dir / "list.txt"
        meta_path = data_dir / "_meta.json"
        if not list_file.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"prepare_data 출력이 누락: {list_file} 또는 {meta_path}. "
                "먼저 eng.prepare_data(manifest_path, data_dir) 를 실행하세요."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        speaker: str = meta["speaker"]
        raw_wav_dir = Path(meta["raw_wav_dir"])
        exp_name = kwargs.pop("exp_name", speaker)

        # Upstream writes prep artefacts and train logs into opt_dir.
        opt_dir = (ckpt_out / "_work").resolve()
        opt_dir.mkdir(parents=True, exist_ok=True)

        batch_size = int(kwargs.pop("batch_size", 8))
        s2_epochs = int(kwargs.pop("s2_epochs", 20))
        s1_epochs = int(kwargs.pop("s1_epochs", 40))
        text_low_lr_rate = float(kwargs.pop("text_low_lr_rate", 0.7))
        save_every_epoch = int(
            kwargs.pop(
                "save_every_epoch",
                # prevent losing final epochs — prior run lost e13-e15 with save_every_epoch=4 on 15-epoch run.
                max(1, min(s1_epochs, s2_epochs) // 10),
            )
        )
        if_save_latest = bool(kwargs.pop("if_save_latest", True))
        if_save_every_weights = bool(kwargs.pop("if_save_every_weights", True))
        if_grad_ckpt = bool(kwargs.pop("if_grad_ckpt", False))
        lora_rank = kwargs.pop("lora_rank", 0)
        gpu_numbers = str(kwargs.pop("gpu_numbers", "0"))
        is_half_flag = str(kwargs.pop("is_half", True))
        feature_timeout = int(kwargs.pop("feature_timeout", 2 * 3600))
        s2_timeout = int(kwargs.pop("s2_timeout", 10 * 3600))
        s1_timeout = int(kwargs.pop("s1_timeout", 10 * 3600))
        if kwargs:
            raise TypeError(f"Unknown train kwargs: {sorted(kwargs)}")

        # 1A — phoneme / BERT feature extraction. env-var driven.
        self._run_prep_stage(
            stage="1a",
            venv_python=venv_python,
            repo_dir=repo_dir,
            script=paths["script_1a"],
            env={
                "inp_text": str(list_file),
                "inp_wav_dir": str(raw_wav_dir),
                "exp_name": exp_name,
                "opt_dir": str(opt_dir),
                "bert_pretrained_dir": str(paths["bert_dir"]),
                "i_part": "0",
                "all_parts": "1",
                "_CUDA_VISIBLE_DEVICES": gpu_numbers,
                "is_half": is_half_flag,
                "version": version,
            },
            timeout=feature_timeout,
        )
        self._consolidate_part_file(
            part_dir=opt_dir,
            part_glob="2-name2text-*.txt",
            consolidated=opt_dir / "2-name2text.txt",
            header=None,
        )

        # 1B — cnHuBERT feature + 32kHz wav.
        self._run_prep_stage(
            stage="1b",
            venv_python=venv_python,
            repo_dir=repo_dir,
            script=paths["script_1b"],
            env={
                "inp_text": str(list_file),
                "inp_wav_dir": str(raw_wav_dir),
                "exp_name": exp_name,
                "opt_dir": str(opt_dir),
                "cnhubert_base_dir": str(paths["cnhubert_dir"]),
                "i_part": "0",
                "all_parts": "1",
                "_CUDA_VISIBLE_DEVICES": gpu_numbers,
                "is_half": is_half_flag,
                "version": version,
            },
            timeout=feature_timeout,
        )

        # 1C — semantic token extraction.
        # 1SV - speaker verification embeddings for v2Pro/v2ProPlus.
        if version in {"v2Pro", "v2ProPlus"}:
            self._run_prep_stage(
                stage="1sv",
                venv_python=venv_python,
                repo_dir=repo_dir,
                script=paths["script_1sv"],
                env={
                    "inp_text": str(list_file),
                    "inp_wav_dir": str(raw_wav_dir),
                    "exp_name": exp_name,
                    "opt_dir": str(opt_dir),
                    "sv_path": str(paths["sv_ckpt"]),
                    "i_part": "0",
                    "all_parts": "1",
                    "_CUDA_VISIBLE_DEVICES": gpu_numbers,
                    "is_half": is_half_flag,
                    "version": version,
                },
                timeout=feature_timeout,
            )

        # 1C - semantic token extraction.
        self._run_prep_stage(
            stage="1c",
            venv_python=venv_python,
            repo_dir=repo_dir,
            script=paths["script_1c"],
            env={
                "inp_text": str(list_file),
                "exp_name": exp_name,
                "opt_dir": str(opt_dir),
                "pretrained_s2G": str(paths["pretrained_s2G"]),
                "s2config_path": str(paths["s2_config_template"]),
                "i_part": "0",
                "all_parts": "1",
                "_CUDA_VISIBLE_DEVICES": gpu_numbers,
                "is_half": is_half_flag,
                "version": version,
            },
            timeout=feature_timeout,
        )
        self._consolidate_part_file(
            part_dir=opt_dir,
            part_glob="6-name2semantic-*.tsv",
            consolidated=opt_dir / "6-name2semantic.tsv",
            header="item_name\tsemantic_audio",
        )

        # S2 (SoVITS) training — build JSON config then invoke s2_train.py -c
        s2_logs_dir = opt_dir / f"logs_s2_{version}"
        s2_logs_dir.mkdir(parents=True, exist_ok=True)
        sovits_weights_dir = ckpt_out / "sovits_weights"
        sovits_weights_dir.mkdir(parents=True, exist_ok=True)

        s2_config = json.loads(paths["s2_config_template"].read_text(encoding="utf-8"))
        s2_config["train"]["batch_size"] = batch_size
        s2_config["train"]["epochs"] = s2_epochs
        s2_config["train"]["text_low_lr_rate"] = text_low_lr_rate
        s2_config["train"]["pretrained_s2G"] = str(paths["pretrained_s2G"])
        s2_config["train"]["pretrained_s2D"] = str(paths["pretrained_s2D"])
        s2_config["train"]["if_save_latest"] = if_save_latest
        s2_config["train"]["if_save_every_weights"] = if_save_every_weights
        s2_config["train"]["save_every_epoch"] = save_every_epoch
        s2_config["train"]["gpu_numbers"] = gpu_numbers
        s2_config["train"]["grad_ckpt"] = if_grad_ckpt
        s2_config["train"]["lora_rank"] = lora_rank
        if version in {"v2Pro", "v2ProPlus"}:
            # Upstream s2v2Pro*.json currently carries no train key for SV.
            if "sv_path" in s2_config["train"]:
                s2_config["train"]["sv_path"] = str(paths["sv_ckpt"])
        if version == "v2ProPlus":
            # Upstream s2v2ProPlus.json currently carries no train key for BigVGAN.
            if "bigvgan_dir" in s2_config["train"]:
                s2_config["train"]["bigvgan_dir"] = str(paths["bigvgan_dir"])
        s2_config["model"]["version"] = version
        s2_config["data"]["exp_dir"] = str(opt_dir)
        s2_config["s2_ckpt_dir"] = str(opt_dir)
        s2_config["save_weight_dir"] = str(sovits_weights_dir)
        s2_config["name"] = exp_name
        s2_config["version"] = version

        s2_config_path = opt_dir / "tmp_s2.json"
        s2_config_path.write_text(json.dumps(s2_config, ensure_ascii=False, indent=2), encoding="utf-8")

        _stream_subprocess(
            args=[str(venv_python), "-s", str(paths["s2_train_script"]), "--config", str(s2_config_path)],
            env={"PYTHONPATH": str(repo_dir / "GPT_SoVITS")},
            cwd=repo_dir,
            timeout=s2_timeout,
            stage_name="S2 training",
            success_paths=[sovits_weights_dir],
        )
        # sovits_weights_dir existence check is weak; follow up with a real pattern match
        if not any(sovits_weights_dir.glob("*.pth")):
            raise RuntimeError(
                f"S2 training finished but no .pth checkpoints were written to {sovits_weights_dir}."
            )

        # S1 (GPT) training — build YAML config then invoke s1_train.py --config_file
        s1_logs_dir = opt_dir / f"logs_s1_{version}"
        s1_logs_dir.mkdir(parents=True, exist_ok=True)
        gpt_weights_dir = ckpt_out / "gpt_weights"
        gpt_weights_dir.mkdir(parents=True, exist_ok=True)

        s1_config_raw = paths["s1_config_template"].read_text(encoding="utf-8")
        s1_config = yaml.safe_load(s1_config_raw)
        s1_config["train"]["batch_size"] = batch_size
        s1_config["train"]["epochs"] = s1_epochs
        s1_config["train"]["save_every_n_epoch"] = save_every_epoch
        s1_config["train"]["if_save_every_weights"] = if_save_every_weights
        s1_config["train"]["if_save_latest"] = if_save_latest
        s1_config["train"]["if_dpo"] = False
        s1_config["train"]["half_weights_save_dir"] = str(gpt_weights_dir)
        s1_config["train"]["exp_name"] = exp_name
        s1_config["pretrained_s1"] = str(paths["pretrained_s1"])
        s1_config["train_semantic_path"] = str(opt_dir / "6-name2semantic.tsv")
        s1_config["train_phoneme_path"] = str(opt_dir / "2-name2text.txt")
        s1_config["output_dir"] = str(s1_logs_dir)

        s1_config_path = opt_dir / "tmp_s1.yaml"
        s1_config_path.write_text(yaml.safe_dump(s1_config, allow_unicode=True), encoding="utf-8")

        _stream_subprocess(
            args=[str(venv_python), "-s", str(paths["s1_train_script"]), "--config_file", str(s1_config_path)],
            env={
                "PYTHONPATH": str(repo_dir / "GPT_SoVITS"),
                "_CUDA_VISIBLE_DEVICES": gpu_numbers,
                "hz": "25hz",
            },
            cwd=repo_dir,
            timeout=s1_timeout,
            stage_name="S1 training",
            success_paths=[gpt_weights_dir],
        )
        if not any(gpt_weights_dir.glob("*.ckpt")):
            raise RuntimeError(
                f"S1 training finished but no .ckpt checkpoints were written to {gpt_weights_dir}."
            )

        # Collect final weights
        final_s2 = ckpt_out / "s2G.pth"
        final_s1 = ckpt_out / "s1bert.pth"
        final_config = ckpt_out / "config.json"

        s2_src = _find_latest(sovits_weights_dir, "*.pth")
        s1_src = _find_latest(gpt_weights_dir, "*.ckpt")
        _copy_or_raise(s2_src, final_s2)
        _copy_or_raise(s1_src, final_s1)

        final_config.write_text(
            json.dumps(
                {
                    "engine": self.name,
                    "license": self.license,
                    "phase": self.phase,
                    "version": version,
                    "train_sample_rate": self.train_sample_rate,
                    "repo_dir": str(repo_dir),
                    "scripts": {
                        "s1_train": str(paths["s1_train_script"]),
                        "s2_train": str(paths["s2_train_script"]),
                        "inference_runner": str(paths["inference_runner"]),
                    },
                    "training": {
                        "data_dir": str(data_dir),
                        "opt_dir": str(opt_dir),
                        "s2_epochs": s2_epochs,
                        "s1_epochs": s1_epochs,
                        "batch_size": batch_size,
                        "save_every_epoch": save_every_epoch,
                    },
                    "sources": {
                        "s2G": str(s2_src),
                        "s1bert": str(s1_src),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        ref_text_map = data_dir / "ref_text_map.json"
        if ref_text_map.exists():
            _copy_or_raise(ref_text_map, ckpt_out / "ref_text_map.json")

        return ckpt_out

    def _run_prep_stage(
        self,
        stage: str,
        venv_python: Path,
        repo_dir: Path,
        script: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        _stream_subprocess(
            args=[str(venv_python), "-s", str(script)],
            env={**env, "PYTHONPATH": str(repo_dir / "GPT_SoVITS")},
            cwd=repo_dir,
            timeout=timeout,
            stage_name=f"prep {stage}",
        )

    @staticmethod
    def _consolidate_part_file(
        part_dir: Path, part_glob: str, consolidated: Path, header: str | None
    ) -> None:
        parts = sorted(part_dir.glob(part_glob))
        if not parts:
            raise FileNotFoundError(f"Prep output 없음: {part_dir / part_glob}")
        lines: list[str] = []
        if header:
            lines.append(header)
        for part in parts:
            content = part.read_text(encoding="utf-8").strip("\n")
            if content:
                lines.extend(content.split("\n"))
        consolidated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for part in parts:
            part.unlink(missing_ok=True)

    # --- synthesize -----------------------------------------------------

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
        repo_dir: Path = paths["dir"]
        venv_python: Path = paths["venv"]
        version: str = paths["version"]

        ckpt = ckpt.expanduser().resolve()
        ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent
        s2_ckpt = ckpt_dir / "s2G.pth"
        s1_ckpt = ckpt_dir / "s1bert.pth"
        if not s2_ckpt.exists() or not s1_ckpt.exists():
            raise FileNotFoundError(f"ckpt_dir 내 s2G.pth / s1bert.pth 없음: {ckpt_dir}")

        resolved_ref_wav = ref_wav.expanduser().resolve()
        ref_text = ""
        ref_text_map_path = ckpt_dir / "ref_text_map.json"
        if ref_text_map_path.exists():
            try:
                mapping = json.loads(ref_text_map_path.read_text(encoding="utf-8"))
                refs = mapping.get("refs", {}) if isinstance(mapping, dict) else {}
                ref_text = refs.get(resolved_ref_wav.name, "") or refs.get(str(resolved_ref_wav), "")
            except json.JSONDecodeError:
                ref_text = ""

        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_sample_rate = sample_rate or self.train_sample_rate
        temp_out = (
            out_path
            if target_sample_rate == self.train_sample_rate
            else out_path.with_suffix(".gpt_sovits_tmp.wav")
        )

        runner = paths["inference_runner"]
        runner_args = [
            "--ckpt_s2", str(s2_ckpt),
            "--ckpt_s1", str(s1_ckpt),
            "--ref_wav", str(resolved_ref_wav),
            "--text", text,
            "--language", language,
            "--out", str(temp_out),
        ]
        if ref_text:
            runner_args.extend(["--ref_text", ref_text])

        env = {
            "GPT_SOVITS_DIR": str(repo_dir),
            "GPT_SOVITS_VERSION": version,
            "PYTHONPATH": str(repo_dir),
        }
        result = run_in_venv(
            venv_python=venv_python,
            script=runner,
            args=runner_args,
            env=env,
            timeout=1800,
            cwd=repo_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"GPT-SoVITS inference failed (exit={result.returncode}).\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if not temp_out.exists():
            raise RuntimeError(
                f"GPT-SoVITS inference succeeded but output missing: {temp_out}"
            )
        if target_sample_rate != self.train_sample_rate:
            if not convert_audio(temp_out, out_path, sr=target_sample_rate, channels=1):
                raise RuntimeError(
                    f"출력 리샘플링 실패: {temp_out} -> {out_path} ({target_sample_rate}Hz)"
                )
            temp_out.unlink(missing_ok=True)
        return out_path
