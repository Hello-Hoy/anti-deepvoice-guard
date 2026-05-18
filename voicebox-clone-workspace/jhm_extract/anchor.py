from __future__ import annotations

import numpy as np


def pick_target_cluster(labels: np.ndarray, file_ids: np.ndarray) -> int:
    """가장 많은 '서로 다른 소스 파일'에 걸친 클러스터 선택.
    동률 시 윈도우 수가 많은 클러스터. (전현무 = 메인 MC = 최다 파일 커버리지)"""
    best_label = -1
    best_key = (-1, -1)
    for lab in np.unique(labels):
        mask = labels == lab
        n_files = len(np.unique(file_ids[mask]))
        n_win = int(mask.sum())
        key = (n_files, n_win)
        if key > best_key:
            best_key = key
            best_label = int(lab)
    return best_label


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
