"""Ad-hoc S1 checkpoint bracket evaluator for GPT-SoVITS."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


TARGETS = [
    ("short", "여보세요, 저 지금 급해서요."),
    ("mid", "오늘 아침에 커피 마시면서 뉴스를 보았어요."),
    ("long", "이번 주말에는 부모님과 같이 영화를 보러 가려고 했는데 시간이 맞지 않아 다음 주로 미뤘어요."),
]
DEFAULT_EPOCHS = [2, 4, 6, 8, 12]


def _map_to_dict_language_key(iw: object, internal_code: str) -> str:
    mapping = getattr(iw, "dict_language", None)
    if not isinstance(mapping, dict):
        raise SystemExit("inference_webui.dict_language is not available")
    for key, value in mapping.items():
        if value == internal_code:
            return str(key)
    raise SystemExit(f"internal language code '{internal_code}' not in dict_language")


def _parse_epochs(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-SoVITS S1 bracket evaluator")
    parser.add_argument("--speaker", default="hyohee")
    parser.add_argument("--version", default="v4")
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("GPT_SOVITS_DIR", "D:/Claude_Projects/GPT-SoVITS")))
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("test-samples/Demo"))
    parser.add_argument("--ref-wav", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--epochs", type=_parse_epochs, default=DEFAULT_EPOCHS)
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

    s2_path = ckpt_dir / "s2G.pth"
    print(f"[setup] loading S2: {s2_path}")
    try:
        next(change_sovits_weights(sovits_path=str(s2_path)))
    except UnboundLocalError:
        pass
    if args.version in {"v2Pro", "v2ProPlus", "v4"} and getattr(iw, "sv_cn_model", None) is None:
        iw.init_sv_cn()
    all_ko = _map_to_dict_language_key(iw, "all_ko")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in args.epochs:
        s1_path = ckpt_dir / "gpt_weights" / f"{args.speaker}-e{epoch}.ckpt"
        if not s1_path.exists():
            print(f"[e{epoch}] SKIP missing {s1_path}")
            continue
        change_gpt_weights(gpt_path=str(s1_path))
        for tag, text in TARGETS:
            out = args.out_dir / f"{args.version}_bracket_e{epoch}_{tag}.wav"
            synthesis = get_tts_wav(
                ref_wav_path=str(args.ref_wav),
                prompt_text=args.ref_text,
                prompt_language=all_ko,
                text=text,
                text_language=all_ko,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temp,
                speed=1.0,
            )
            segments = list(synthesis)
            if not segments:
                print(f"[e{epoch}/{tag}] FAIL no segments")
                continue
            sr, audio = segments[-1]
            sf.write(str(out), audio, sr)
            print(f"[e{epoch}/{tag}] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
