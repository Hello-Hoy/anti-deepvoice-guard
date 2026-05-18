from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from .anchor import nearest_distinct_files, pick_target_cluster
from .decode import decode_to_wav16k
from .segments import sliding_window_embeddings

SR = 16000
WIN_S = 1.6
STEP_S = 0.4

WORK = Path(__file__).resolve().parents[1]
SHOW_DIR = WORK / "[전현무계획3] 無계획 노포 맛집 탐방!   매주 금요일 밤 9시 10분"
CACHE = WORK / "_cache"
ANCHOR_DIR = WORK / "_anchor"
ANCHOR_CAND = WORK / "_anchor_candidates"
OUT_DIR = WORK / "jeonhyunmoo_clean"
ANCHOR_NPY = ANCHOR_DIR / "anchor.npy"


def _encoder():
    from resemblyzer import VoiceEncoder

    return VoiceEncoder(device="cpu", verbose=False)


def _list_sources() -> list[Path]:
    return sorted(SHOW_DIR.glob("*.m4a"))


def _short_clips(srcs: list[Path], limit: int) -> list[Path]:
    """파일 크기 작은 순(예고편/쇼츠) = 전현무 내레이션 비중 큼."""
    return sorted(srcs, key=lambda p: p.stat().st_size)[:limit]


def run_bootstrap(sample_n: int = 30, k: int = 6, force: bool = False) -> None:
    if ANCHOR_NPY.exists() and not force:
        print(f"anchor 이미 존재: {ANCHOR_NPY} (재생성하려면 --rebootstrap)")
        return
    ANCHOR_CAND.mkdir(parents=True, exist_ok=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    srcs = _short_clips(_list_sources(), sample_n)
    print(f"부트스트랩 샘플 {len(srcs)}개")
    enc = _encoder()

    all_emb: list[np.ndarray] = []
    all_fid: list[int] = []
    win_meta: list[tuple[int, float, Path]] = []  # (file_idx, t0, wav)
    for fi, src in enumerate(srcs):
        try:
            wav = decode_to_wav16k(src, CACHE)
            y, _ = sf.read(str(wav), dtype="float32")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {src.name}: {e}")
            continue
        starts, embs = sliding_window_embeddings(y, enc, WIN_S, STEP_S)
        for s, e in zip(starts, embs):
            all_emb.append(e)
            all_fid.append(fi)
            win_meta.append((fi, s, wav))

    if not all_emb:
        raise RuntimeError("부트스트랩: 임베딩 0개 — 소스 디코드 확인")

    X = np.stack(all_emb)
    fid = np.array(all_fid)

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    target = pick_target_cluster(km.labels_, fid)
    centroid = km.cluster_centers_[target]
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    mask = km.labels_ == target
    picks = nearest_distinct_files(X[mask], fid[mask], centroid, n=3)
    cand_idx = np.where(mask)[0]

    print(f"\n전현무 후보 클러스터={target} "
          f"(파일 {len(np.unique(fid[mask]))}개 커버, 윈도우 {mask.sum()}개)")
    print("아래 3개 클립을 들어보고 전현무가 맞는지 확인하세요:\n")
    for n, (f_, local_i) in enumerate(picks):
        gi = cand_idx[local_i]
        _, t0, wav = win_meta[gi]
        y, _ = sf.read(str(wav), dtype="float32")
        s = int(t0 * SR)
        seg = y[s:s + int(10 * SR)]
        out = ANCHOR_CAND / f"cand_{n + 1:02d}.wav"
        sf.write(str(out), seg, SR, subtype="PCM_16")
        print(f"  {out}")

    np.save(ANCHOR_NPY, centroid.astype(np.float32))
    print(f"\n임시 anchor 저장: {ANCHOR_NPY}")
    print("→ 후보가 전현무가 맞으면 그대로 --pilot 진행.")
    print("→ 아니면 --rebootstrap --k <다른값> 으로 재시도.")


def main() -> None:
    ap = argparse.ArgumentParser(description="전현무 단독 음성 추출")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--rebootstrap", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pilot-n", type=int, default=25)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sim-thr", type=float, default=0.80)
    args = ap.parse_args()

    if args.bootstrap or args.rebootstrap:
        run_bootstrap(k=args.k, force=args.rebootstrap)
    elif args.pilot:
        from .runner import run_extract

        run_extract(_short_clips(_list_sources(), args.pilot_n),
                    args.sim_thr, tag="pilot")
    elif args.full:
        from .runner import run_extract

        run_extract(_list_sources(), args.sim_thr, tag="full")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
