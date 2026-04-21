"""Pronunciation probes with parameterized GPT-SoVITS version and sampling."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _map_key(iw: object, internal: str) -> str:
    for key, value in iw.dict_language.items():
        if value == internal:
            return str(key)
    raise SystemExit(f"{internal} not in dict_language")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-SoVITS pronunciation probes")
    parser.add_argument("--speaker", default="hyohee")
    parser.add_argument("--version", default="v4")
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GPT_SOVITS_DIR", "D:/Claude_Projects/GPT-SoVITS")))
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("test-samples/Demo"))
    parser.add_argument("--ref-wav", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--alt-ref-wav", type=Path)
    parser.add_argument("--alt-ref-text", default="")
    parser.add_argument("--text", default="여보세요, 저 지금 급해서요.")
    parser.add_argument("--s1-epoch", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--temp", type=float, default=0.6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_dir = args.ckpt_dir or Path("model-training/weights/voice-clone") / f"user-{args.speaker}" / "gpt_sovits"
    os.environ["version"] = args.version
    os.environ["GPT_SOVITS_VERSION"] = args.version
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if str(args.repo_dir) not in sys.path:
        sys.path.insert(0, str(args.repo_dir))

    from GPT_SoVITS import inference_webui as iw  # type: ignore
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav  # type: ignore
    import soundfile as sf  # type: ignore

    s1_path = ckpt_dir / "gpt_weights" / f"{args.speaker}-e{args.s1_epoch}.ckpt"
    s2_path = ckpt_dir / "s2G.pth"
    change_gpt_weights(gpt_path=str(s1_path))
    try:
        next(change_sovits_weights(sovits_path=str(s2_path)))
    except UnboundLocalError:
        pass
    if args.version in {"v2Pro", "v2ProPlus", "v4"} and getattr(iw, "sv_cn_model", None) is None:
        iw.init_sv_cn()
    all_ko = _map_key(iw, "all_ko")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes = [("default", args.ref_wav, args.ref_text, args.top_k, args.top_p, args.temp)]
    if args.alt_ref_wav:
        probes.append(("altref", args.alt_ref_wav, args.alt_ref_text, args.top_k, args.top_p, args.temp))
    probes.append(("extra_conservative", args.ref_wav, args.ref_text, 3, 0.5, 0.3))

    for name, ref_wav, ref_text, top_k, top_p, temp in probes:
        out = args.out_dir / f"{args.version}_probe_{name}.wav"
        segments = list(get_tts_wav(
            ref_wav_path=str(ref_wav),
            prompt_text=ref_text,
            prompt_language=all_ko,
            text=args.text,
            text_language=all_ko,
            top_k=top_k,
            top_p=top_p,
            temperature=temp,
            speed=1.0,
        ))
        if segments:
            sr, audio = segments[-1]
            sf.write(str(out), audio, sr)
            print(f"[{name}] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
