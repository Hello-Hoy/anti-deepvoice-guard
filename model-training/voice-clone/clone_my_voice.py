#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.voice_clone.engines import list_engines, load_engine
from model_training.voice_clone.engines.base import resolve_checkpoint


def synthesize(
    text: str,
    speaker: str,
    engine: str = "cosyvoice2",
    ref_wav: Path | None = None,
    ref_wav_idx: int | None = None,
    ref_mode: Literal["rotate", "best"] = "rotate",
    out_path: Path = Path("out.wav"),
    language: str = "ko",
    sample_rate: int = 24000,
    checkpoint_override: Path | None = None,
) -> Path:
    eng = load_engine(engine)
    ok, msg = eng.health_check()
    if not ok:
        raise RuntimeError(f"Engine {engine} is not ready: {msg}")
    if checkpoint_override is not None and not checkpoint_override.exists():
        raise FileNotFoundError(f"--checkpoint does not exist: {checkpoint_override}")
    ckpt = checkpoint_override or resolve_checkpoint(speaker, engine)
    kwargs = {
        "text": text,
        "ckpt": ckpt,
        "ref_wav": ref_wav,
        "language": language,
        "out_path": out_path,
        "sample_rate": sample_rate,
    }
    if engine in {"gpt_sovits", "openvoice2"}:
        kwargs["ref_wav_idx"] = ref_wav_idx
        kwargs["ref_mode"] = ref_mode
    return eng.synthesize(**kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Common voice-clone synthesis CLI")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--text-file", type=Path, help="UTF-8 file with one sentence per line")
    parser.add_argument("--speaker", default="hyohee")
    parser.add_argument("--engine", default="cosyvoice2", choices=list_engines())
    parser.add_argument("--ref-wav", type=Path)
    parser.add_argument("--ref-wav-idx", type=int)
    parser.add_argument("--ref-mode", choices=("rotate", "best"), default="rotate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out.wav"))
    parser.add_argument("--language", default="ko")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--list-engines", action="store_true")
    parser.add_argument("--health-check", action="store_true")
    return parser


def _print_engine_status() -> int:
    for name in list_engines():
        try:
            engine = load_engine(name)
            ok, message = engine.health_check()
            status = "OK" if ok else "NOT_READY"
        except Exception as exc:
            status = "IMPORT_ERROR"
            message = str(exc)
        print(f"{name}\t{status}\t{message}")
    return 0


def _read_texts(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        return [line.strip() for line in args.text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.text:
        return [args.text]
    raise SystemExit("--text or --text-file is required unless --list-engines/--health-check is used.")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list_engines:
        return _print_engine_status()
    if args.health_check:
        engine = load_engine(args.engine)
        ok, message = engine.health_check()
        print("OK" if ok else "NOT_READY")
        print(message)
        return 0 if ok else 1

    texts = _read_texts(args)
    for index, text in enumerate(texts, start=1):
        out = args.out
        if len(texts) > 1:
            out = args.out.parent / f"{args.out.stem}_{index:04d}{args.out.suffix or '.wav'}"
        result = synthesize(
            text=text,
            speaker=args.speaker,
            engine=args.engine,
            ref_wav=args.ref_wav,
            ref_wav_idx=args.ref_wav_idx,
            ref_mode=args.ref_mode,
            out_path=out,
            language=args.language,
            sample_rate=args.sample_rate,
            checkpoint_override=args.checkpoint,
        )
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
