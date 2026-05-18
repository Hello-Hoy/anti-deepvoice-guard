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
