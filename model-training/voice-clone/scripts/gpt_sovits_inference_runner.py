"""Standalone inference runner for GPT-SoVITS that supports Korean.

Upstream's `GPT_SoVITS/inference_cli.py` only exposes zh/en/ja.
This runner reaches below that CLI and drives the actual
`change_sovits_weights` / `change_gpt_weights` / `get_tts_wav` pipeline
from `GPT_SoVITS.inference_webui`, which supports `all_ko` / `ko` / etc.

Invoke inside the GPT-SoVITS venv with cwd=GPT-SoVITS repo root so the
package layout resolves (`GPT_SoVITS.*`, `tools.*`).

Example:
    python model-training/voice-clone/scripts/gpt_sovits_inference_runner.py \
        --ckpt_s2 /ckpts/s2G.pth \
        --ckpt_s1 /ckpts/s1bert.pth \
        --ref_wav /refs/phase0.wav \
        --ref_text "여보세요" \
        --text "여보세요, 저 지금 급해서요." \
        --language ko \
        --out /out/clone.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_LANGUAGE_ALIASES = {
    "ko": "all_ko",
    "all_ko": "all_ko",
    "ko_en": "ko",
    "zh": "all_zh",
    "all_zh": "all_zh",
    "zh_en": "zh",
    "en": "en",
    "ja": "all_ja",
    "all_ja": "all_ja",
    "ja_en": "ja",
    "yue": "all_yue",
    "all_yue": "all_yue",
    "yue_en": "yue",
    "auto": "auto",
    "auto_yue": "auto_yue",
}


def _resolve_language(code: str) -> str:
    normalized = code.strip().lower()
    if normalized not in _LANGUAGE_ALIASES:
        raise SystemExit(
            f"Unsupported language '{code}'. Known: {sorted(_LANGUAGE_ALIASES)}"
        )
    return _LANGUAGE_ALIASES[normalized]


def main() -> int:
    parser = argparse.ArgumentParser(description="GPT-SoVITS Korean-capable inference runner")
    parser.add_argument("--ckpt_s2", required=True, help="Path to SoVITS (s2G) .pth checkpoint")
    parser.add_argument("--ckpt_s1", required=True, help="Path to GPT (s1bert) .ckpt/.pth checkpoint")
    parser.add_argument("--ref_wav", required=True, help="Reference audio WAV path")
    parser.add_argument("--ref_text", default="", help="Reference text transcript (optional but recommended)")
    parser.add_argument("--text", required=True, help="Target text to synthesize")
    parser.add_argument("--language", default="ko", help="Target language code (ko/en/zh/ja/yue/...)")
    parser.add_argument("--ref_language", default=None, help="Reference language (defaults to --language)")
    parser.add_argument(
        "--version",
        default=os.environ.get("GPT_SOVITS_VERSION", "v2"),
        help="GPT-SoVITS version hint used before importing inference_webui",
    )
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    repo_dir = Path(os.environ.get("GPT_SOVITS_DIR") or os.getcwd()).resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # Upstream's inference_webui depends on these env vars being set for weights discovery.
    os.environ["version"] = args.version
    os.environ.setdefault("GPT_SOVITS_VERSION", args.version)

    from GPT_SoVITS import inference_webui as iw  # type: ignore
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav  # type: ignore

    try:
        next(change_sovits_weights(sovits_path=args.ckpt_s2))
    except UnboundLocalError:
        # Upstream's generator loads weights before touching UI update locals.
        pass
    change_gpt_weights(gpt_path=args.ckpt_s1)
    if getattr(iw, "model_version", args.version) in {"v2Pro", "v2ProPlus"} and getattr(iw, "sv_cn_model", None) is None:
        iw.init_sv_cn()

    def _map_to_dict_language_key(internal_code: str) -> str:
        """`get_tts_wav` expects keys from `dict_language` (i18n-translated labels),
        not the internal codes like `all_ko`. Look up the active `dict_language`
        (v1 or v2 depending on loaded SoVITS weights) and reverse-map."""
        mapping = getattr(iw, "dict_language", None)
        if not isinstance(mapping, dict):
            raise SystemExit("inference_webui.dict_language is not available")
        for key, value in mapping.items():
            if value == internal_code:
                return key
        available = sorted({v for v in mapping.values() if isinstance(v, str)})
        raise SystemExit(
            f"internal language code '{internal_code}' not present in dict_language. "
            f"Available codes: {available}"
        )

    text_language = _map_to_dict_language_key(_resolve_language(args.language))
    ref_language = _map_to_dict_language_key(
        _resolve_language(args.ref_language or args.language)
    )

    import soundfile as sf  # provided by GPT-SoVITS venv

    synthesis = get_tts_wav(
        ref_wav_path=args.ref_wav,
        prompt_text=args.ref_text or "",
        prompt_language=ref_language,
        text=args.text,
        text_language=text_language,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
    )
    segments = list(synthesis)
    if not segments:
        raise SystemExit("Inference produced no audio segments.")

    sample_rate, audio = segments[-1]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate)
    print(f"wrote: {out_path} ({sample_rate} Hz, {len(audio)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
