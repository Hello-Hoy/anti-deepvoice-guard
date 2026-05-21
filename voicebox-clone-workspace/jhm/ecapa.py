"""ECAPA-TDNN(speechbrain) 화자 임베딩 — resemblyzer보다 변별력 높음.

클린 단독 소스(강연 등)에서 화자 클러스터링으로 전현무 단독 발화를 분리하는 데 사용.
ECAPA는 16kHz mono를 입력으로 받으며 192-d d-vector를 낸다.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import SR, WORK, log

_ENC = None


def _encoder(device: str = "cpu"):
    global _ENC
    if _ENC is None:
        from speechbrain.inference import EncoderClassifier

        _ENC = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(WORK / "ecapa_model"),
            run_opts={"device": device},
        )
    return _ENC


def ecapa_embed_segments(wav16k: np.ndarray, segs: list[dict], device: str = "cpu",
                         min_dur_s: float = 0.8) -> tuple[np.ndarray, list[dict]]:
    """각 발화 세그먼트 → L2 정규화 ECAPA d-vector(192). 짧은 건 제외.

    반환: (embeddings[N,192], kept_segments[N] with 'dur').
    """
    enc = _encoder(device)
    embs, kept = [], []
    for s in segs:
        dur = (s["end"] - s["start"]) / SR
        if dur < min_dur_s:
            continue
        clip = wav16k[s["start"]: s["end"]].astype(np.float32)
        try:
            with torch.no_grad():
                e = enc.encode_batch(torch.from_numpy(clip)[None]).squeeze().cpu().numpy()
        except Exception as ex:  # noqa: BLE001 — 한 세그먼트 실패가 전체를 막지 않도록
            log(f"    ecapa skip ({dur:.1f}s): {type(ex).__name__}")
            continue
        e = e / (np.linalg.norm(e) + 1e-9)
        embs.append(e)
        kept.append({**s, "dur": dur})
    if not embs:
        return np.zeros((0, 192), np.float32), []
    return np.stack(embs).astype(np.float32), kept
