import numpy as np

from jhm_extract.segments import group_turns

WIN_S = 1.6
STEP_S = 0.4


def _starts(n: int) -> list[float]:
    return [i * STEP_S for i in range(n)]


def test_single_contiguous_turn():
    is_t = np.zeros(30, dtype=bool)
    is_t[5:25] = True  # 20 윈도우 연속
    turns = group_turns(is_t, _starts(30), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert len(turns) == 1
    t = turns[0]
    assert abs(t["t0"] - 5 * STEP_S) < 1e-6
    assert abs(t["t1"] - (24 * STEP_S + WIN_S)) < 1e-6
    assert t["dur"] > 6.0


def test_short_turn_dropped():
    is_t = np.zeros(20, dtype=bool)
    is_t[2:6] = True  # 4 윈도우 → dur ~ 1.6+1.2=2.8s < 6
    turns = group_turns(is_t, _starts(20), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert turns == []


def test_small_gap_merged():
    is_t = np.zeros(40, dtype=bool)
    is_t[5:19] = True
    is_t[19] = False  # 정확히 한 윈도우(idx 19) dip
    is_t[20:33] = True
    turns = group_turns(is_t, _starts(40), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert len(turns) == 1  # 1-윈도우 gap 병합되어 단일 턴


def test_large_gap_splits():
    is_t = np.zeros(60, dtype=bool)
    is_t[2:20] = True
    is_t[40:58] = True  # 사이에 큰 무음
    turns = group_turns(is_t, _starts(60), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=6.0)
    assert len(turns) == 2


def test_just_over_gap_splits():
    # 3-윈도우 gap: starts[nxt]-starts[j]=1.6 > step+gap_merge=1.2 → 병합 금지
    is_t = np.zeros(40, dtype=bool)
    is_t[5:15] = True   # j는 idx 14에서 끝
    is_t[18:28] = True  # gap = idx 15,16,17 (3 윈도우)
    turns = group_turns(is_t, _starts(40), WIN_S, STEP_S,
                         gap_merge_s=0.8, min_turn_s=0.0)
    assert len(turns) == 2


from jhm_extract.segments import clip_from_turn, select_clips


def test_clip_from_turn_short_returns_none():
    assert clip_from_turn({"t0": 0.0, "t1": 8.0, "dur": 8.0}, 10.0, 20.0) is None


def test_clip_from_turn_in_range_whole():
    assert clip_from_turn({"t0": 3.0, "t1": 18.0, "dur": 15.0}, 10.0, 20.0) == (3.0, 18.0)


def test_clip_from_turn_long_center_cropped():
    c = clip_from_turn({"t0": 100.0, "t1": 130.0, "dur": 30.0}, 10.0, 20.0)
    assert c is not None
    t0, t1 = c
    assert abs((t1 - t0) - 20.0) < 1e-6
    assert abs(((t0 + t1) / 2) - 115.0) < 1e-6


def test_clip_from_turn_exact_bounds_kept_whole():
    # dur == min, dur == max 는 strict 비교(<,>)라 둘 다 절취 없이 그대로 반환
    assert clip_from_turn({"t0": 1.0, "t1": 11.0, "dur": 10.0}, 10.0, 20.0) == (1.0, 11.0)
    assert clip_from_turn({"t0": 2.0, "t1": 22.0, "dur": 20.0}, 10.0, 20.0) == (2.0, 22.0)


def test_select_clips_skips_short():
    turns = [
        {"t0": 0.0, "t1": 8.0, "dur": 8.0},
        {"t0": 3.0, "t1": 18.0, "dur": 15.0},
    ]
    assert select_clips(turns, 10.0, 20.0) == [(3.0, 18.0)]


def test_sliding_window_embeddings_shapes():
    from resemblyzer import VoiceEncoder

    from jhm_extract.segments import sliding_window_embeddings

    rng = np.random.default_rng(0)
    y = (rng.standard_normal(16000 * 5).astype(np.float32) * 0.1)
    enc = VoiceEncoder(device="cpu", verbose=False)
    starts, embs = sliding_window_embeddings(y, enc, win_s=1.6, step_s=0.4)
    assert embs.ndim == 2
    assert embs.shape[0] == len(starts)
    assert embs.shape[0] > 5
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)  # 정규화됨
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))
