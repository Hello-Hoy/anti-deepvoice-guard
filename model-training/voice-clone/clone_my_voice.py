#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


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
    out_path: Path = Path("out.wav"),
    language: str = "ko",
    sample_rate: int = 24000,
    checkpoint_override: Path | None = None,
) -> Path:
    eng = load_engine(engine)
    ok, msg = eng.health_check()
    if not ok:
        raise RuntimeError(f"엔진 {engine} 준비 안 됨: {msg}")

    if checkpoint_override is not None and not checkpoint_override.exists():
        raise FileNotFoundError(f"--checkpoint 경로가 존재하지 않습니다: {checkpoint_override}")

    ckpt = checkpoint_override or resolve_checkpoint(speaker, engine)
    if ckpt is None and ref_wav is None:
        raise RuntimeError(
            f"{speaker}/{engine} 체크포인트를 찾지 못했습니다. "
            "Phase 0 zero-shot 모드를 사용하려면 --ref-wav 를 함께 전달하세요."
        )
    return eng.synthesize(
        text=text,
        ckpt=ckpt,
        ref_wav=ref_wav,
        language=language,
        out_path=out_path,
        sample_rate=sample_rate,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="공통 voice-clone 추론 CLI")
    parser.add_argument("--text", help="합성할 문장")
    parser.add_argument("--speaker", help="speaker 식별자 (weights/voice-clone/user-<speaker>/...)")
    parser.add_argument("--engine", default="cosyvoice2", choices=list_engines(), help="합성 엔진")
    parser.add_argument("--ref-wav", type=Path, help="zero-shot 또는 prompt reference wav")
    parser.add_argument("--checkpoint", type=Path, help="체크포인트 경로 override")
    parser.add_argument("--out", type=Path, default=Path("out.wav"), help="출력 WAV 경로")
    parser.add_argument("--language", default="ko", help="합성 언어 코드")
    parser.add_argument("--sample-rate", type=int, default=24000, help="출력 샘플레이트")
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="등록된 엔진과 각 health_check 결과를 출력",
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_engines:
        return _print_engine_status()

    if not args.text or not args.speaker:
        parser.error("--text 와 --speaker 는 --list-engines 없이 실행할 때 필수입니다.")

    checkpoint = args.checkpoint or resolve_checkpoint(args.speaker, args.engine)
    if checkpoint is None and args.ref_wav is not None:
        print(
            "체크포인트를 찾지 못해 Phase 0 zero-shot 모드로 진행합니다. "
            "--ref-wav 를 speaker reference로 사용합니다."
        )
    elif checkpoint is None:
        print(
            "체크포인트를 찾지 못했습니다. "
            "Phase 0 zero-shot 모드를 사용하려면 --ref-wav 를 지정하세요."
        )

    result = synthesize(
        text=args.text,
        speaker=args.speaker,
        engine=args.engine,
        ref_wav=args.ref_wav,
        out_path=args.out,
        language=args.language,
        sample_rate=args.sample_rate,
        checkpoint_override=args.checkpoint,
    )

    summary = checkpoint if checkpoint is not None else "Phase 0 zero-shot"
    print(f"엔진: {args.engine}")
    print(f"체크포인트: {summary}")
    print(f"출력 파일: {result}")
    print(f"샘플레이트: {args.sample_rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
