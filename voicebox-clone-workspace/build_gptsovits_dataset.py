"""전현무 GPT-SoVITS 데이터셋 빌드 CLI.

서브커맨드: probe / rebuild-anchor / extract / montage / finalize
사용: ../.venv/bin/python build_gptsovits_dataset.py <cmd> [opts]
"""
from __future__ import annotations

import argparse
import shutil
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
PROCESSED = STAGING / "processed.json"
EXPORT_SR = 32000
PREVIEW = STAGING / "preview"
REJECT = STAGING / "reject.txt"
DATASET = GS / "dataset"
WAVS = DATASET / "wavs"
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
    import json
    import librosa

    device = pick_device()
    anchor = _anchor()
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = G.read_manifest(MANIFEST) if MANIFEST.exists() else []
    processed: list[str] = json.loads(PROCESSED.read_text(encoding="utf-8")) if PROCESSED.exists() else []
    done = set(processed) | {e["source"] for e in entries}
    next_idx = len(entries) + 1

    for src in _sources(args.limit):
        rel = str(src.relative_to(SRC_DIR))
        if rel in done:
            log(f"[skip] {rel} (이미 처리됨)")
            continue
        try:
            vocals, vsr = demucs_vocal(src, device=device)
            voc16 = librosa.resample(vocals, orig_sr=vsr, target_sr=SR).astype(np.float32)
            voc32 = librosa.resample(vocals, orig_sr=vsr, target_sr=EXPORT_SR).astype(np.float32)
            segs = vad_segments(voc16)
            embs, kept = embed_segments(voc16, segs)
            chosen = []
            if len(kept):
                sims = cosine(embs, anchor)
                chosen = [{**kept[i], "sim": float(sims[i])} for i in range(len(kept)) if sims[i] >= args.threshold]
            new = G.save_segments(voc32, chosen, source=rel, out_dir=SEG_DIR,
                                  start_index=next_idx, sr16=SR, sr32=EXPORT_SR)
            next_idx += len(new)
            entries.extend(new)
            processed.append(rel)
            G.write_manifest(MANIFEST, entries)
            PROCESSED.write_text(json.dumps(processed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            log(f"[extract] {rel}: {len(chosen)}/{len(kept)} 채택 (T={args.threshold})")
        except Exception as exc:  # noqa: BLE001 — 한 파일 실패가 전체를 막지 않도록
            log(f"[extract] {rel} 실패: {type(exc).__name__}: {exc}")
    total = sum(e["dur"] for e in entries)
    log(f"[extract] 총 {len(entries)}개 세그먼트 / {total/60:.1f}분 (처리 {len(processed)}파일) → {MANIFEST}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    import librosa

    device = pick_device()
    anchor = _anchor()
    thresholds = args.thresholds or [0.70, 0.75, 0.80, 0.85]
    grand: dict[float, float] = {t: 0.0 for t in thresholds}
    for src in _sources(args.n):
        try:
            vocals, vsr = demucs_vocal(src, device=device)
            voc16 = librosa.resample(vocals, orig_sr=vsr, target_sr=SR).astype(np.float32)
            segs = vad_segments(voc16)
            embs, kept = embed_segments(voc16, segs)
            if len(kept) == 0:
                log(f"[probe] {src.name}: 발화 0"); continue
            sims = cosine(embs, anchor)
            log(f"[probe] {src.name}: 발화 {len(kept)}개 "
                f"sim mean={sims.mean():.3f} p75={np.percentile(sims,75):.3f} max={sims.max():.3f}")
            for t in thresholds:
                grand[t] += sum(kept[i]["dur"] for i in range(len(kept)) if sims[i] >= t)
        except Exception as exc:  # noqa: BLE001
            log(f"[probe] {src.name} 실패: {type(exc).__name__}: {exc}")
    log("=== 임계별 누적 채택 분량 (probe " + f"{args.n}개 파일) ===")
    for t in thresholds:
        log(f"  T={t:.2f} → {grand[t]/60:.1f}분")
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
    if not MANIFEST.exists():
        log("[montage] segments.json 없음 — 먼저 extract 실행"); return 1
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


def cmd_finalize(args: argparse.Namespace) -> int:
    if not MANIFEST.exists():
        log("[finalize] segments.json 없음 — 먼저 extract 실행"); return 1
    entries = G.read_manifest(MANIFEST)
    rejected = G.parse_reject(REJECT.read_text(encoding="utf-8")) if REJECT.exists() else set()
    kept = [e for e in entries if e["id"] not in rejected]
    if not kept:
        log("[finalize] 남은 세그먼트 0 — 중단")
        return 1
    WAVS.mkdir(parents=True, exist_ok=True)
    for e in kept:
        shutil.copyfile(SEG_DIR / f"{e['id']}.wav", WAVS / f"{e['id']}.wav")
    device = pick_device()
    texts = G.transcribe_segments([WAVS / f"{e['id']}.wav" for e in kept],
                                  device="cuda" if device == "cuda" else "cpu")
    list_entries = [{"relpath": f"wavs/{e['id']}.wav", "text": texts.get(e["id"], "")} for e in kept]
    lines = G.build_list_lines(list_entries, speaker="jhm", lang="ko")
    (DATASET / "jhm.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DATASET / "relocate_list.py").write_text(G.relocate_script_text(), encoding="utf-8")
    (DATASET / "requirements.txt").write_text(G.requirements_text(), encoding="utf-8")
    total_min = sum(e["dur"] for e in kept) / 60
    (DATASET / "README.md").write_text(G.readme_text(len(lines), total_min, args.threshold), encoding="utf-8")
    log(f"[finalize] {len(lines)}개 라벨 / {total_min:.1f}분 → {DATASET}/jhm.list "
        f"(제외 {len(rejected)}, 빈전사 {len(kept)-len(lines)})")
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
    pf = sub.add_parser("finalize")
    pf.add_argument("--threshold", type=float, default=0.0, help="README 기록용")
    pf.set_defaults(func=cmd_finalize)
    pp = sub.add_parser("probe")
    pp.add_argument("--n", type=int, default=6)
    pp.add_argument("--thresholds", type=float, nargs="*", default=None)
    pp.set_defaults(func=cmd_probe)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
