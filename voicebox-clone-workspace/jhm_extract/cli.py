from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .anchor import nearest_distinct_files, rank_clusters
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


def run_bootstrap(sample_n: int = 30, k: int = 12, top_n: int = 6,
                  force: bool = False) -> None:
    if ANCHOR_NPY.exists() and not force:
        print(f"anchor 이미 존재: {ANCHOR_NPY} (재생성하려면 --rebootstrap)")
        return
    ANCHOR_CAND.mkdir(parents=True, exist_ok=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    srcs = _short_clips(_list_sources(), sample_n)
    print(f"부트스트랩 샘플 {len(srcs)}개, k={k}, 상위 {top_n} 클러스터 제시")
    enc = _encoder()

    all_emb: list[np.ndarray] = []
    all_fid: list[int] = []
    win_meta: list[tuple[int, float, Path]] = []
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
    ranked = rank_clusters(km.labels_, fid)[:top_n]

    print(f"\n상위 {len(ranked)}개 클러스터 — 각 폴더를 듣고 전현무인 번호를 찾으세요:\n")
    for r, lab in enumerate(ranked, start=1):
        mask = km.labels_ == lab
        Xc = X[mask]
        fidc = fid[mask]
        cand_idx = np.where(mask)[0]
        centroid = Xc.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims_c = Xc @ centroid
        topk = np.argsort(sims_c)[-min(200, len(sims_c)):]
        refined = Xc[topk].mean(axis=0)
        refined = refined / (np.linalg.norm(refined) + 1e-9)

        cdir = ANCHOR_CAND / f"cluster_{r:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        picks = nearest_distinct_files(Xc, fidc, refined, n=3)
        for n, (f_, local_i) in enumerate(picks):
            gi = cand_idx[local_i]
            _, t0, wav = win_meta[gi]
            y, _ = sf.read(str(wav), dtype="float32")
            s = int(t0 * SR)
            seg = y[s:s + int(8 * SR)]
            sf.write(str(cdir / f"cand_{n + 1:02d}.wav"), seg, SR,
                     subtype="PCM_16")
        np.save(ANCHOR_DIR / f"cluster_{r:02d}.npy", refined.astype(np.float32))
        print(f"  [{r}] 파일 {len(np.unique(fidc))}개 커버, "
              f"윈도우 {int(mask.sum())}개 → {cdir}/")

    print("\n→ 전현무인 클러스터 번호 N 확인 후: "
          "extract_jhm.py --use-cluster N")
    print("→ 적합한 게 없으면: extract_jhm.py --rebootstrap --k <다른값>")


def run_use_cluster(n: int) -> None:
    src = ANCHOR_DIR / f"cluster_{n:02d}.npy"
    if not src.exists():
        raise RuntimeError(f"{src} 없음 — 먼저 --bootstrap 실행 또는 번호 확인")
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ANCHOR_NPY, np.load(src))
    print(f"클러스터 {n} → 정식 anchor 채택: {ANCHOR_NPY}")
    print("→ 다음: extract_jhm.py --pilot")


def main() -> None:
    ap = argparse.ArgumentParser(description="전현무 단독 음성 추출")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--rebootstrap", action="store_true")
    ap.add_argument("--use-cluster", type=int, default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pilot-n", type=int, default=25)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--sim-thr", type=float, default=0.80)
    args = ap.parse_args()

    if args.bootstrap or args.rebootstrap:
        run_bootstrap(k=args.k, top_n=args.top_n, force=args.rebootstrap)
    elif args.use_cluster is not None:
        run_use_cluster(args.use_cluster)
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
