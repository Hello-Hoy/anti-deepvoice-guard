"""모델 버전 확인 v2 엔드포인트 (스텁)."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v2", tags=["v2-model"])


@router.get("/model/latest")
async def get_latest_model():
    """현재 최신 모델 버전 정보를 반환한다 (현재는 고정 응답)."""
    return {
        "version": "aasist-v5.1",
        "checksum": "a1b2c3d4e5f67890",
        "download_url": None,  # 모델 업데이트 미구현 (CONCEPT ONLY)
        "release_date": "2026-04-04",
        "notes": "v5.1: FFT rolloff fix, clamp, gain cap, MPS contiguous",
    }
