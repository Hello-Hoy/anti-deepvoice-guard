"""OpenVoice v2 + MeloTTS Korean zero-shot adapter.

The upstream OpenVoice and MeloTTS packages have changed import paths across
commits. This adapter keeps the project interface complete and fail-fast while
leaving precise inference calls behind TODO markers that must be checked
against the pinned upstream commits before production synthesis.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Literal

from . import register
from ._manifest import load_manifest, split_samples
from .base import VoiceCloneEngine


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default.expanduser()


@register("openvoice2")
class OpenVoice2Engine(VoiceCloneEngine):
    name = "openvoice2"
    train_sample_rate = 24000
    supports_language = ["ko"]
    license = "MIT/Apache-2.0 upstream mix; verify pinned repos"
    phase = "zero_shot"
    hardware_pref = "cpu_or_cuda"

    def health_check(self) -> tuple[bool, str]:
        missing: list[str] = []
        if importlib.util.find_spec("openvoice") is None:
            missing.append(
                "Python package 'openvoice' is missing. On Windows, install from the pinned "
                "OpenVoice v2 git commit in requirements-cloning.txt; avoid unpinned pip mirrors."
            )
        if importlib.util.find_spec("melo") is None and importlib.util.find_spec("melotts") is None:
            missing.append(
                "MeloTTS package is missing. Install the pinned MeloTTS git dependency and "
                "confirm the Korean checkpoint path below."
            )

        melo_ckpt = _env_path("MELOTTS_KR_CKPT", Path("~/tools/MeloTTS/checkpoints/korean"))
        tone_ckpt = _env_path("OPENVOICE2_TONE_CONVERTER", Path("~/tools/OpenVoice/checkpoints_v2/converter"))
        se_extractor = _env_path("OPENVOICE2_SE_EXTRACTOR", Path("~/tools/OpenVoice/checkpoints_v2/base_speakers/ses"))
        for label, path in (
            ("MeloTTS-KR checkpoint", melo_ckpt),
            ("OpenVoice v2 tone converter weights", tone_ckpt),
            ("OpenVoice v2 speaker embedding/se_extractor weights", se_extractor),
        ):
            if not path.exists():
                missing.append(f"{label} not found: {path}")

        if missing:
            return False, "\n".join(missing)
        return True, f"melo={melo_ckpt}; tone_converter={tone_ckpt}; se_extractor={se_extractor}"

    def prepare_data(
        self,
        manifest_path: Path,
        out_dir: Path,
        min_snr_db: float | None = 35.0,
    ) -> Path:
        manifest = load_manifest(manifest_path)
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        refs: list[dict[str, Any]] = []
        for sample in split_samples(manifest, "train"):
            try:
                duration_sec = float(sample.get("duration_sec", 0.0))
                snr_db = float(sample.get("snr_db", 0.0))
            except (TypeError, ValueError):
                continue
            if min_snr_db is not None and snr_db < min_snr_db:
                continue
            if 6.0 <= duration_sec <= 10.0:
                wav = Path(str(sample["wav_path_24k"]))
                if not wav.is_absolute():
                    wav = (manifest_path.parent / wav).resolve()
                refs.append(
                    {
                        "id": str(sample["id"]),
                        "wav": str(wav),
                        "text": str(sample.get("text") or ""),
                        "duration_sec": duration_sec,
                        "snr_db": snr_db,
                        "style": str(sample.get("style") or sample.get("metadata", {}).get("style") or ""),
                    }
                )
        refs.sort(key=lambda item: float(item["snr_db"]), reverse=True)
        payload = {
            "speaker": manifest["speaker"],
            "engine": self.name,
            "strategy": "train split, SNR top-5, 6-10 second clips",
            "rotate_index": 0,
            "refs": refs[:5],
        }
        (out_dir / "ref_selection.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return out_dir

    def train(self, data_dir: Path, ckpt_out: Path, epochs: int = 0, **kwargs: Any) -> Path:
        ckpt_out = ckpt_out.expanduser().resolve()
        ckpt_out.mkdir(parents=True, exist_ok=True)
        selection = data_dir / "ref_selection.json"
        if selection.exists():
            (ckpt_out / "ref_selection.json").write_text(selection.read_text(encoding="utf-8"), encoding="utf-8")
        (ckpt_out / "config.json").write_text(
            json.dumps(
                {"engine": self.name, "mode": "zero-shot", "message": "OpenVoice v2 training is a no-op."},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("OpenVoice v2 is zero-shot; train() only persisted reference selection.")
        return ckpt_out

    def synthesize(
        self,
        text: str,
        ckpt: Path | None,
        ref_wav: Path | None = None,
        ref_wav_idx: int | None = None,
        ref_mode: Literal["rotate", "best"] = "rotate",
        language: str = "ko",
        out_path: Path = Path("out.wav"),
        sample_rate: int | None = None,
    ) -> Path:
        ok, message = self.health_check()
        if not ok:
            raise RuntimeError(message)
        if language != "ko":
            raise ValueError("OpenVoice2Engine currently supports Korean only.")
        ckpt_dir = (ckpt.expanduser().resolve() if ckpt else Path.cwd())
        selected_ref = ref_wav.expanduser().resolve() if ref_wav else self._select_ref(ckpt_dir, text, ref_mode, ref_wav_idx)
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO(openvoice2-api): after pinning OpenVoice v2 and MeloTTS commits,
        # replace this placeholder with:
        # 1. MeloTTS Korean base synthesis to a temporary WAV.
        # 2. se_extractor.get_se(selected_ref, tone_converter, vad=False/True per upstream).
        # 3. ToneColorConverter.convert(base_wav, src_se, tgt_se, output_path=out_path).
        raise NotImplementedError(
            "OpenVoice v2 inference API must be wired after upstream commit pinning. "
            f"Selected reference: {selected_ref}; output target: {out_path}"
        )

    @staticmethod
    def _select_ref(
        ckpt_dir: Path,
        text: str,
        ref_mode: Literal["rotate", "best"],
        ref_wav_idx: int | None,
    ) -> Path:
        selection_path = ckpt_dir / "ref_selection.json"
        if not selection_path.exists():
            raise ValueError("OpenVoice2 synthesize needs --ref-wav or ref_selection.json.")
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        refs = payload.get("refs", [])
        if not refs:
            raise ValueError(f"No references in {selection_path}")
        if ref_wav_idx is not None:
            chosen = refs[int(ref_wav_idx) % len(refs)]
        elif ref_mode == "rotate":
            index = int(payload.get("rotate_index", 0)) % len(refs)
            chosen = refs[index]
            payload["rotate_index"] = index + 1
            selection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif ref_mode == "best":
            target_len = len(text)
            chosen = min(
                refs,
                key=lambda ref: (
                    abs(len(str(ref.get("text", ""))) - target_len) - 0.05 * float(ref.get("snr_db", 0.0)),
                    -float(ref.get("snr_db", 0.0)),
                ),
            )
        else:
            raise ValueError("ref_mode must be 'rotate' or 'best'")
        wav = Path(str(chosen["wav"])).expanduser().resolve()
        if not wav.exists():
            raise FileNotFoundError(f"Selected reference WAV does not exist: {wav}")
        return wav
