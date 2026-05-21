"""전현무 GPT-SoVITS 데이터셋 빌드 CLI.

서브커맨드: probe / rebuild-anchor / extract / montage / finalize
사용: ../.venv/bin/python build_gptsovits_dataset.py <cmd> [opts]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jhm.core import SR, cosine, demucs_vocal, embed_segments, log, vad_segments
from jhm.device import pick_device
from jhm import gptsovits as G

WORK = Path(__file__).resolve().parent
SRC_DIR = WORK / "전현무음성"
ANCHOR = WORK / "jhm_work" / "anchor" / "anchor.npy"
GS = WORK / "jhm_work" / "gptsovits"
STAGING = GS / "staging"
SEG_DIR = STAGING / "segments"
MANIFEST = STAGING / "segments.json"
EXPORT_SR = 32000
PREVIEW = STAGING / "preview"
REJECT = STAGING / "reject.txt"
CLEAN_REFS = ["montage_cluster_01.wav", "montage_cluster_05.wav",
              "montage_cluster_07.wav", "montage_cluster_16.wav"]


def _sources(limit: int | None) -> list[Path]:
    files = sorted(SRC_DIR.rglob("*.m4a"))
    return files[:limit] if limit else files


def _anchor() -> np.ndarray:
    if not ANCHOR.exists():
        raise RuntimeError("anchor.npy 없음 — 먼저 `rebuild-anchor` 실행")
    return np.load(ANCHOR)


def cmd_extract(args: argparse.Namespace) -> int:
    device = pick_device()
    anchor = _anchor()
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if MANIFEST.exists():
        entries = G.read_manifest(MANIFEST)
    done = {e["source"] for e in entries}
    next_idx = len(entries) + 1
    import librosa

    for src in _sources(args.limit):
        if src.name in done:
            log(f"[skip] {src.name} (이미 추출됨)")
            continue
        try:
            vocals, vsr = demucs_vocal(src, device=device)
            voc16 = librosa.resample(vocals, orig_sr=vsr, target_sr=SR).astype(np.float32)
            voc32 = librosa.resample(vocals, orig_sr=vsr, target_sr=EXPORT_SR).astype(np.float32)
            segs = vad_segments(voc16)
            embs, kept = embed_segments(voc16, segs)
            if len(kept) == 0:
                log(f"[extract] {src.name}: 발화 0")
                continue
            sims = cosine(embs, anchor)
            chosen = [{**kept[i], "sim": float(sims[i])} for i in range(len(kept)) if sims[i] >= args.threshold]
            new = G.save_segments(voc32, chosen, source=src.name, out_dir=SEG_DIR,
                                  start_index=next_idx, sr16=SR, sr32=EXPORT_SR)
            next_idx += len(new)
            entries.extend(new)
            G.write_manifest(MANIFEST, entries)
            log(f"[extract] {src.name}: {len(chosen)}/{len(kept)} 채택 (T={args.threshold})")
        except Exception as exc:  # noqa: BLE001 — 한 파일 실패가 전체를 막지 않도록
            log(f"[extract] {src.name} 실패: {type(exc).__name__}: {exc}")
    total = sum(e["dur"] for e in entries)
    log(f"[extract] 총 {len(entries)}개 세그먼트 / {total/60:.1f}분 → {MANIFEST}")
    return 0


def cmd_rebuild_anchor(args: argparse.Namespace) -> int:
    from resemblyzer import VoiceEncoder, preprocess_wav

    ref_dir = WORK / "jhm_work" / "anchor"
    enc = VoiceEncoder(device="cpu", verbose=False)
    embs = []
    for name in CLEAN_REFS:
        p = ref_dir / name
        if not p.exists():
            raise RuntimeError(f"클린 ref 없음: {p}")
        embs.append(enc.embed_utterance(preprocess_wav(str(p))))
    anchor = np.mean(embs, axis=0)
    anchor = (anchor / (np.linalg.norm(anchor) + 1e-9)).astype(np.float32)
    np.save(ANCHOR, anchor)
    log(f"[anchor] 클린 {len(embs)}-ref 평균 → {ANCHOR} (norm={np.linalg.norm(anchor):.3f})")
    return 0


def cmd_montage(args: argparse.Namespace) -> int:
    entries = G.read_manifest(MANIFEST)
    entries_sorted = sorted(entries, key=lambda e: e["sim"], reverse=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    placed = G.build_montage(entries_sorted, SEG_DIR, PREVIEW / "montage_all.wav", sr=EXPORT_SR)
    (PREVIEW / "timeline.txt").write_text("\n".join(G.timeline_lines(placed)) + "\n", encoding="utf-8")
    by_src: dict[str, list[dict]] = {}
    for e in entries_sorted:
        by_src.setdefault(e["source"], []).append(e)
    for src, es in by_src.items():
        safe = "".join(c if c.isalnum() else "_" for c in src)[:60]
        G.build_montage(es, SEG_DIR, PREVIEW / f"montage_{safe}.wav", sr=EXPORT_SR)
    if not REJECT.exists():
        REJECT.write_text(
            "# 제외할 세그먼트 id를 한 줄에 하나씩 적으세요 (예: jhm_0003).\n"
            "# '#' 뒤는 주석. preview/timeline.txt로 id↔시각 확인.\n",
            encoding="utf-8",
        )
    log(f"[montage] {len(placed)}개 → {PREVIEW}/montage_all.wav, timeline.txt, reject.txt")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--threshold", type=float, required=True)
    pe.add_argument("--limit", type=int, default=None)
    pe.set_defaults(func=cmd_extract)
    sub.add_parser("rebuild-anchor").set_defaults(func=cmd_rebuild_anchor)
    sub.add_parser("montage").set_defaults(func=cmd_montage)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
