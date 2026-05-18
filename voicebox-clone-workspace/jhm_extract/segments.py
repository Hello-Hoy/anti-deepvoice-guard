from __future__ import annotations

import numpy as np


def group_turns(
    is_target: np.ndarray,
    starts: list[float],
    win_s: float,
    step_s: float,
    gap_merge_s: float,
    min_turn_s: float,
) -> list[dict]:
    """연속 타깃 윈도우를 턴으로 묶고, 짧은 dip은 병합. min_turn_s 미만 제거."""
    turns: list[dict] = []
    n = len(is_target)
    i = 0
    max_gap_windows = int(round(gap_merge_s / step_s)) + 1
    while i < n:
        if not is_target[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if is_target[j + 1]:
                j += 1
                continue
            nxt = None
            for k in range(j + 2, min(j + 2 + max_gap_windows, n)):
                if is_target[k]:
                    nxt = k
                    break
            if nxt is not None and (starts[nxt] - starts[j]) <= (step_s + gap_merge_s):
                j = nxt
                continue
            break
        t0 = starts[i]
        t1 = starts[j] + win_s
        dur = t1 - t0
        if dur >= min_turn_s:
            turns.append({"t0": t0, "t1": t1, "dur": dur, "i0": i, "i1": j})
        i = j + 1
    return turns


def clip_from_turn(
    turn: dict,
    min_clip_s: float = 10.0,
    max_clip_s: float = 20.0,
) -> tuple[float, float] | None:
    """단일 턴 → 10~20초 클립 (t0, t1). min 미만은 None, max 초과는 중앙 절취."""
    dur = turn["dur"]
    if dur < min_clip_s:
        return None
    if dur > max_clip_s:
        mid = (turn["t0"] + turn["t1"]) / 2.0
        return (mid - max_clip_s / 2.0, mid + max_clip_s / 2.0)
    return (turn["t0"], turn["t1"])


def select_clips(
    turns: list[dict],
    min_clip_s: float = 10.0,
    max_clip_s: float = 20.0,
) -> list[tuple[float, float]]:
    """여러 턴에서 클립 절취 (clip_from_turn 위에 구현).

    min_clip_s 미만 턴은 제외되므로 len(결과) <= len(turns) 일 수 있다."""
    out: list[tuple[float, float]] = []
    for t in turns:
        c = clip_from_turn(t, min_clip_s, max_clip_s)
        if c is not None:
            out.append(c)
    return out


def sliding_window_embeddings(
    y: np.ndarray,
    encoder,
    win_s: float = 1.6,
    step_s: float = 0.4,
    sr: int = 16000,
):
    """16k mono y에 슬라이딩 윈도우 화자 임베딩.

    반환: (starts: list[float] 윈도우 시작초, embs: ndarray[n,256] L2정규화).
    임베딩이 없으면 ([], ndarray shape (0,256))."""
    win = int(win_s * sr)
    step = int(step_s * sr)
    starts: list[float] = []
    embs: list[np.ndarray] = []
    for s in range(0, max(0, len(y) - win + 1), step):
        emb = encoder.embed_utterance(y[s:s + win])
        embs.append(emb)
        starts.append(s / sr)
    if not embs:
        return [], np.zeros((0, 256), dtype=np.float32)
    arr = np.stack(embs).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return starts, arr
