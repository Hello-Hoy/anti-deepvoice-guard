"""피싱 규칙 동기화 v2 엔드포인트."""

import hashlib
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status

router = APIRouter(prefix="/api/v2", tags=["v2-rules"])


def _resolve_rules_dir() -> Path:
    """규칙 JSON 파일이 있는 디렉토리를 반환.
    환경변수 PHISHING_RULES_DIR이 우선, 없으면 프로젝트의 android-app assets 경로.
    """
    env_dir = os.environ.get("PHISHING_RULES_DIR")
    if env_dir:
        return Path(env_dir)
    # Fallback: dev/monorepo 경로.
    return Path(__file__).resolve().parents[4] / "android-app/app/src/main/assets"


def _load_rules() -> dict:
    """피싱 규칙 파일을 로드하고 ETag용 체크섬을 계산한다.

    **Fail-closed (H76)**: 두 파일 중 하나라도 없거나 파싱 실패면 HTTPException(503)을 던져
    빈 ruleset으로 silently serve되지 않도록 한다.
    """
    rules_dir = _resolve_rules_dir()
    keywords_path = rules_dir / "phishing_keywords.json"
    phrases_path = rules_dir / "phishing_phrases.json"

    if not keywords_path.exists() or not phrases_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Phishing rules assets missing on server. "
                f"Set PHISHING_RULES_DIR env or place files under {rules_dir}."
            ),
        )
    try:
        keywords = json.loads(keywords_path.read_text(encoding="utf-8"))
        phrases = json.loads(phrases_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Phishing rules read/parse failed: {exc}",
        ) from exc

    kw_list = keywords.get("keywords", [])
    ph_list = phrases.get("phrases", [])
    # **각 파일 독립 검증 (H78)** — 한쪽이 비어 있거나 list가 아니면 fail-closed.
    if not isinstance(kw_list, list) or not kw_list:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phishing keywords list is missing, empty, or malformed.",
        )
    if not isinstance(ph_list, list) or not ph_list:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phishing phrases list is missing, empty, or malformed.",
        )

    # MEDIUM 8: sort_keys=True는 dict key 순서만 안정화한다.
    # nested list 순서는 그대로 checksum에 반영되므로, 클라이언트 refetch 의도면 현 상태 유지가 맞다.
    combined = json.dumps({"keywords": keywords, "phrases": phrases}, sort_keys=True)
    checksum = hashlib.sha256(combined.encode()).hexdigest()[:16]

    return {
        "version": keywords.get("version", 1),
        "checksum": checksum,
        "keywords": kw_list,
        "phrases": ph_list,
    }


@router.get("/phishing/rules")
async def get_phishing_rules(request: Request):
    """최신 피싱 키워드/구문 규칙을 반환한다. ETag 지원 + fail-closed."""
    rules = _load_rules()
    etag = f'"{rules["checksum"]}"'

    # If-None-Match: 304 Not Modified
    if_none_match = request.headers.get("if-none-match")
    if if_none_match == etag:
        return Response(status_code=304)

    return Response(
        content=json.dumps(rules, ensure_ascii=False),
        media_type="application/json",
        headers={"ETag": etag},
    )
