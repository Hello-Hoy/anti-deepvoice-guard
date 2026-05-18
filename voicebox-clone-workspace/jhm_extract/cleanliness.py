from __future__ import annotations

from dataclasses import dataclass, field, asdict

import librosa
import numpy as np

SR = 16000
HOP_MS = 50
FRAME_MS = 50


@dataclass
class CleanMetrics:
    voiced_ratio: float
    longest_sil_s: float
    peak: float
    gap_rms_ratio: float
    pause_ratio: float
    spectral_flatness: float
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _rms(y: np.ndarray) -> np.ndarray:
    hop = int(SR * HOP_MS / 1000)
    frame = int(SR * FRAME_MS / 1000)
    if y.size < frame:
        return np.zeros(0, dtype=np.float32)
    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]


def cleanliness_gate(
    y: np.ndarray,
    noise_floor: float,
    voiced_mult: float = 8.0,
    gap_ratio_max: float = 3.0,
    min_voiced_ratio: float = 0.7,
    max_sil_s: float = 1.5,
    peak_max: float = 0.95,
    min_pause_ratio: float = 0.04,
    max_flatness: float = 0.6,
) -> CleanMetrics:
    """모든 조건 통과 시 passed=True. 핵심 판별기는 gap_rms_ratio + pause_ratio.

    불변식: gap_ratio_max < voiced_mult. pause = (rms < noise_floor*voiced_mult)로
    분류되므로, gap-energy 분기가 도달 가능하려면 BGM 판정 임계가 pause 임계보다
    낮아야 한다(같으면 gap_rms_ratio ≤ voiced_mult = gap_ratio_max 로 영원히 미발동).
    크게 깔린 BGM은 gap을 voiced로 만들어 no_pauses 분기가 잡고, 약하게 깔린
    BGM(noise_floor의 gap_ratio_max~voiced_mult배)은 gap-energy 분기가 잡는다."""
    reasons: list[str] = []
    rms = _rms(y)
    voiced_thr = max(noise_floor * voiced_mult, 1e-6)
    voiced = rms > voiced_thr if rms.size else np.zeros(0, dtype=bool)
    pause = ~voiced if rms.size else np.zeros(0, dtype=bool)

    voiced_ratio = float(voiced.mean()) if rms.size else 0.0
    pause_ratio = float(pause.mean()) if rms.size else 0.0
    gap_rms = float(rms[pause].mean()) if pause.any() else 0.0
    gap_rms_ratio = gap_rms / noise_floor if noise_floor > 0 else float("inf")

    longest = cur = 0
    for v in pause:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    longest_sil_s = longest * HOP_MS / 1000.0

    peak = float(np.abs(y).max()) if y.size else 0.0
    flat = (
        float(np.median(librosa.feature.spectral_flatness(y=y)[0]))
        if y.size
        else 1.0
    )

    if voiced_ratio < min_voiced_ratio:
        reasons.append(f"voiced_ratio<{min_voiced_ratio}")
    if longest_sil_s > max_sil_s:
        reasons.append(f"silence>{max_sil_s}s")
    if peak > peak_max:
        reasons.append("clipping")
    if pause_ratio < min_pause_ratio:
        reasons.append("no_pauses(music-bed?)")
    if gap_rms_ratio > gap_ratio_max:
        reasons.append(f"gap_energy>{gap_ratio_max}x(BGM?)")
    if flat > max_flatness:
        reasons.append(f"flatness>{max_flatness}(noisy)")

    return CleanMetrics(
        voiced_ratio=voiced_ratio,
        longest_sil_s=longest_sil_s,
        peak=peak,
        gap_rms_ratio=gap_rms_ratio,
        pause_ratio=pause_ratio,
        spectral_flatness=flat,
        passed=not reasons,
        reasons=reasons,
    )
