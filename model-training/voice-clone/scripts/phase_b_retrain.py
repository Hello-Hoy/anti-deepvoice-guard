"""Phase B GPT-SoVITS retrain entrypoint."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B GPT-SoVITS retrain")
    parser.add_argument("--speaker", default="hyohee")
    parser.add_argument("--version", default="v2ProPlus")
    parser.add_argument("--min-snr", type=float, default=25.0)
    parser.add_argument("--s1-epochs", type=int, default=40)
    parser.add_argument("--s2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile", choices=("quality", "fast"), default="quality")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--ckpt-out", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser.parse_args()


def _resolve_default_manifest(root: Path, speaker: str) -> Path:
    requested = root / "model-training" / "voice-clone" / "data" / speaker / "manifest.json"
    actual_layout = root / "model-training" / "voice-clone" / "data" / f"user-{speaker}" / "manifest.json"
    if requested.exists() or not actual_layout.exists():
        return requested
    return actual_layout


def _resolve_path(path: Path | None, default: Path) -> Path:
    return (path if path is not None else default).expanduser().resolve()


def _archive_previous(ckpt_out: Path) -> Path | None:
    if not (ckpt_out / "config.json").exists():
        return None
    previous_root = ckpt_out.parent / "_previous"
    previous_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = previous_root / stamp
    suffix = 1
    while destination.exists():
        destination = previous_root / f"{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(ckpt_out), str(destination))
    return destination


def _set_default_env(root: Path, version: str) -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("GPT_SOVITS_DIR", str(root.parent / "GPT-SoVITS"))
    os.environ.setdefault("GPT_SOVITS_VERSION", version)
    os.environ["GPT_SOVITS_VERSION"] = version


def main() -> int:
    args = _parse_args()
    root = _repo_root()

    manifest = _resolve_path(args.manifest, _resolve_default_manifest(root, args.speaker))
    ckpt_out = _resolve_path(
        args.ckpt_out,
        root / "model-training" / "weights" / "voice-clone" / f"user-{args.speaker}" / "gpt_sovits",
    )
    data_dir = _resolve_path(
        args.data_dir,
        root / "model-training" / "voice-clone" / "data" / f"user-{args.speaker}" / "gpt_sovits_prepared",
    )

    archived = _archive_previous(ckpt_out)
    ckpt_out.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _set_default_env(root, args.version)

    from model_training.voice_clone.engines.gpt_sovits import GPTSoVITSEngine

    engine = GPTSoVITSEngine()
    prepare_sig = inspect.signature(engine.prepare_data)
    if "min_snr_db" not in prepare_sig.parameters:
        raise RuntimeError(
            "GPTSoVITSEngine.prepare_data does not accept min_snr_db. "
            "Apply the Phase B engine changes before running this script."
        )

    s1_epochs = args.s1_epochs
    s2_epochs = args.s2_epochs
    if args.profile == "fast":
        s1_epochs = max(1, s1_epochs // 2)
        s2_epochs = max(1, s2_epochs // 2)

    started = perf_counter()
    prepared = engine.prepare_data(manifest, data_dir, min_snr_db=args.min_snr)
    trained = engine.train(
        prepared,
        ckpt_out,
        s1_epochs=s1_epochs,
        s2_epochs=s2_epochs,
        batch_size=args.batch_size,
    )
    duration_sec = perf_counter() - started

    meta_path = prepared / "_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    skipped_low_snr = meta.get("skipped_low_snr", [])
    skipped_missing_snr = meta.get("skipped_missing_snr", [])

    print()
    print("Phase B retrain summary")
    print("=======================")
    print(f"speaker: {args.speaker}")
    print(f"version: {args.version}")
    print(f"profile: {args.profile}")
    print(f"manifest: {manifest}")
    print(f"prepared data: {prepared}")
    print(f"checkpoint dir: {trained}")
    print(f"s2 checkpoint: {trained / 's2G.pth'}")
    print(f"s1 checkpoint: {trained / 's1bert.pth'}")
    if archived is not None:
        print(f"archived previous: {archived}")
    print(f"duration seconds: {duration_sec:.1f}")
    print(f"min snr db: {args.min_snr}")
    print(f"skipped low snr ({len(skipped_low_snr)}): {', '.join(skipped_low_snr) or '(none)'}")
    print(f"missing snr kept ({len(skipped_missing_snr)}): {', '.join(skipped_missing_snr) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
