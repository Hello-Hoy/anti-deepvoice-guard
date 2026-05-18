from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .cleanliness import cleanliness_gate
from .decode import cache_key, decode_to_wav16k
from .manifest import Manifest
from .segments import clip_from_turn, group_turns, sliding_window_embeddings

SR = 16000
WIN_S = 1.6
STEP_S = 0.4
HOP_MS = 50

from .cli import ANCHOR_NPY, CACHE, OUT_DIR, _encoder


def _noise_floor(y: np.ndarray) -> float:
    hop = int(SR * HOP_MS / 1000)
    frame = hop
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    return float(np.percentile(rms, 10)) if rms.size else 1e-4


def run_extract(srcs: list[Path], sim_thr: float, tag: str) -> None:
    if not ANCHOR_NPY.exists():
        raise RuntimeError("anchor 없음 — 먼저 `--bootstrap` 실행")
    anchor = np.load(ANCHOR_NPY)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man = Manifest(OUT_DIR / f"manifest_{tag}.json")
    enc = _encoder()
    print(f"[{tag}] 소스 {len(srcs)}개, sim_thr={sim_thr}")

    for si, src in enumerate(srcs):
        key = None
        try:
            key = cache_key(src)
            if man.done(key):
                continue
            wav = decode_to_wav16k(src, CACHE)
            y, _ = sf.read(str(wav), dtype="float32")
        except Exception as e:  # noqa: BLE001
            man.add_error(src.name, f"decode: {e}")
            print(f"  [{si}] ERROR {src.name}: {e}")
            continue

        starts, embs = sliding_window_embeddings(y, enc, WIN_S, STEP_S)
        if embs.shape[0] == 0:
            if key:
                man.mark_done(key, 0)
            continue
        sims = embs @ anchor
        is_t = sims >= sim_thr
        turns = group_turns(is_t, starts, WIN_S, STEP_S,
                            gap_merge_s=0.8, min_turn_s=8.0)

        nf = _noise_floor(y)
        n_kept = 0
        for turn in turns:
            clip = clip_from_turn(turn, 10.0, 20.0)
            if clip is None:
                continue
            t0, t1 = clip
            seg = y[int(t0 * SR):int(t1 * SR)]
            cm = cleanliness_gate(seg, noise_floor=nf)
            if not cm.passed:
                continue
            dur = t1 - t0
            sim = float(sims[turn["i0"]:turn["i1"] + 1].mean())
            clean_score = max(0.0, 1.0 - cm.gap_rms_ratio / 4.0)
            score = float(sim * np.sqrt(dur) * (0.5 + 0.5 * clean_score))
            simtag = int(round(sim * 100))
            name = f"jhm_{si:03d}_{int(t0)}s_{int(dur)}s_sim{simtag}.wav"
            sf.write(str(OUT_DIR / name), seg, SR, subtype="PCM_16")
            man.add_clip({
                "out": name, "src": src.name, "t0": float(t0),
                "dur": float(dur), "sim": sim,
                "gap_rms_ratio": cm.gap_rms_ratio,
                "voiced_ratio": cm.voiced_ratio,
                "spectral_flatness": cm.spectral_flatness,
                "score": score,
            })
            n_kept += 1
        if key:
            man.mark_done(key, n_kept)
        if si % 10 == 0:
            man.save()
        print(f"  [{si}] {src.name[:40]}  clips={n_kept}")

    # 점수순 정렬 후 매니페스트 최종 저장
    man.clips.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    man.save()
    s = man.stats()
    print(f"\n[{tag}] 완료: clips={s['n_clips']} "
          f"총 {s['total_seconds'] / 60:.1f}분 errors={s['n_errors']}")
    print(f"산출: {OUT_DIR}")
    print(f"매니페스트: {man.path}")
