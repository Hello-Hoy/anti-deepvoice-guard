"""실행 디바이스 자동 선택 — Mac(MPS)/Windows·Linux(CUDA)/CPU 이식성."""
from __future__ import annotations


def pick_device(prefer: str | None = None, *, available: dict[str, bool] | None = None) -> str:
    """cuda > mps > cpu 순으로 사용 가능한 디바이스명 반환.

    prefer가 주어지고 사용 가능하면 그것을 우선한다.
    available을 주면(테스트용) torch 조회를 건너뛴다.
    """
    if available is None:
        import torch

        available = {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        }
    if prefer and available.get(prefer):
        return prefer
    for d in ("cuda", "mps", "cpu"):
        if available.get(d):
            return d
    return "cpu"
