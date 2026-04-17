"""텔레메트리 v2 엔드포인트 — 익명 메타데이터 수집.

보안 가드:
- X-API-Key 헤더 인증 필수 (환경변수 TELEMETRY_API_KEY)
- 클라이언트 IP 기반 rate limit (분당 60건)
- 요청 body 크기 제한 (4KB)
- Bounded in-memory storage (최대 1000건, FIFO 삭제)
- /stats 엔드포인트는 동일 API key 필요 + 내부용
"""

import os
import time
from collections import deque
from datetime import datetime
from typing import Deque

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v2", tags=["v2-telemetry"])

# 설정 (환경변수로 오버라이드 가능)
_API_KEY = os.environ.get("TELEMETRY_API_KEY", "")
_MAX_STORE_SIZE = int(os.environ.get("TELEMETRY_MAX_STORE", "1000"))
_RATE_LIMIT_PER_MIN = int(os.environ.get("TELEMETRY_RATE_LIMIT", "60"))
_MAX_BODY_BYTES = int(os.environ.get("TELEMETRY_MAX_BODY", "4096"))

# Bounded 인메모리 스토리지 (deque) — 최대 크기 초과 시 가장 오래된 것부터 자동 삭제.
_telemetry_store: Deque[dict] = deque(maxlen=_MAX_STORE_SIZE)

# IP → [timestamps...] — rate limit 추적 (분 단위 슬라이딩 윈도우)
_rate_buckets: dict[str, Deque[float]] = {}


class TelemetryEvent(BaseModel):
    deepfake_score: float = Field(ge=0.0, le=1.0)
    phishing_score: float = Field(ge=0.0, le=1.0)
    phishing_keywords: list[str] = Field(default_factory=list, max_length=20)
    combined_threat_level: str = Field(max_length=16)
    model_version: str = Field(default="aasist-v5.1", max_length=32)
    inference_mode: str = Field(default="on_device", max_length=16)
    stt_available: bool = True


def require_api_key(request: Request) -> str:
    """X-API-Key 헤더 검증. 서버 키 미설정 시 명시적으로 비활성화."""
    if not _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telemetry endpoint disabled (TELEMETRY_API_KEY not set)",
        )
    provided = request.headers.get("x-api-key", "")
    if provided != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )
    return provided


def check_body_size(request: Request):
    """요청 body 크기 제한."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Body exceeds {_MAX_BODY_BYTES} bytes",
        )


def check_rate_limit(request: Request):
    """클라이언트 IP 기준 분당 요청 수 제한."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60.0
    bucket = _rate_buckets.setdefault(client_ip, deque(maxlen=_RATE_LIMIT_PER_MIN + 1))
    # 오래된 타임스탬프 제거.
    while bucket and bucket[0] < window_start:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT_PER_MIN:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (per minute)",
        )
    bucket.append(now)


@router.post("/telemetry", status_code=201)
async def receive_telemetry(
    event: TelemetryEvent,
    request: Request,
    _auth: str = Depends(require_api_key),
):
    """인증된 클라이언트로부터 탐지 메타데이터를 수신한다."""
    check_body_size(request)
    check_rate_limit(request)

    entry = event.model_dump()
    entry["received_at"] = datetime.utcnow().isoformat()
    # deque.maxlen이 자동으로 오래된 항목을 제거.
    _telemetry_store.append(entry)
    return {"status": "ok", "stored_count": len(_telemetry_store)}


@router.get("/telemetry/stats")
async def telemetry_stats(
    request: Request,
    _auth: str = Depends(require_api_key),
):
    """운영용 통계 — 인증 필수. 원시 이벤트는 노출하지 않고 집계만 반환."""
    check_rate_limit(request)
    total = len(_telemetry_store)
    if total == 0:
        return {"total_events": 0, "by_threat_level": {}}

    # 집계 (원시 데이터 미노출)
    by_level: dict[str, int] = {}
    for entry in _telemetry_store:
        level = entry.get("combined_threat_level", "UNKNOWN")
        by_level[level] = by_level.get(level, 0) + 1

    return {
        "total_events": total,
        "max_capacity": _MAX_STORE_SIZE,
        "by_threat_level": by_level,
    }
