from __future__ import annotations

import numpy as np


def rank_clusters(labels: np.ndarray, file_ids: np.ndarray) -> list[int]:
    """클러스터를 (서로 다른 소스 파일 수, 윈도우 수) 내림차순 정렬한 라벨 리스트."""
    scored: list[tuple[int, int, int]] = []
    for lab in np.unique(labels):
        mask = labels == lab
        scored.append((len(np.unique(file_ids[mask])), int(mask.sum()), int(lab)))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [lab for _, _, lab in scored]


def pick_target_cluster(labels: np.ndarray, file_ids: np.ndarray) -> int:
    """가장 많은 서로 다른 파일에 걸친 클러스터(동률 시 윈도우 최다)."""
    return rank_clusters(labels, file_ids)[0]


def nearest_distinct_files(
    embs: np.ndarray,
    file_ids: np.ndarray,
    centroid: np.ndarray,
    n: int,
) -> list[tuple[int, int]]:
    """centroid에 가까운 순서로, 서로 다른 file_id에서 하나씩 (file_id, window_idx) n개."""
    c = centroid / (np.linalg.norm(centroid) + 1e-9)
    sims = embs @ c
    order = np.argsort(sims)[::-1]
    picks: list[tuple[int, int]] = []
    seen: set[int] = set()
    for idx in order:
        fid = int(file_ids[idx])
        if fid in seen:
            continue
        seen.add(fid)
        picks.append((fid, int(idx)))
        if len(picks) >= n:
            break
    return picks
