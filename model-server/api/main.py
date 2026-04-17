"""Anti-DeepVoice Guard — FastAPI 서버.

AASIST 모델을 사용하여 deepfake 음성을 탐지하는 REST API 서버.

Usage:
    uvicorn model_server.api.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from model_server.inference.aasist_model import AASISTDetector

# 전역 모델 인스턴스
_detector: AASISTDetector | None = None

# 기본 모델 경로
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "model-training"
    / "weights"
    / "official"
    / "AASIST.pth"
)


def get_detector() -> AASISTDetector | None:
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    config_path = os.environ.get("CONFIG_PATH", None)
    device = os.environ.get("DEVICE", "cpu")

    print(f"Loading AASIST model from {model_path} (device={device})...")
    _detector = AASISTDetector(model_path=model_path, config_path=config_path, device=device)
    print("Model loaded successfully.")

    yield

    _detector = None


app = FastAPI(
    title="Anti-DeepVoice Guard API",
    version="2.0.0",
    description="Deepfake voice detection + voice phishing analysis API",
    lifespan=lifespan,
)

# v2 라우터 등록
from model_server.api.routes.v2.telemetry import router as telemetry_router
from model_server.api.routes.v2.rules import router as rules_router
from model_server.api.routes.v2.model import router as model_router

app.include_router(telemetry_router)
app.include_router(rules_router)
app.include_router(model_router)


# 설정: 엔드포인트별 body size 하드 캡 (bytes).
# v2 telemetry는 작은 JSON만 허용; v1 detect는 오디오 업로드 경로라 별도 제한.
_TELEMETRY_BODY_LIMIT = int(os.environ.get("TELEMETRY_MAX_BODY", "4096"))
# /api/v1/detect는 multipart/form-data 업로드. multipart boundary/헤더 overhead가 Content-Length에
# 포함되므로 10MB 오디오가 실제로는 10MB보다 약간 크게 보고된다. 12MB로 여유 확보.
_DETECT_BODY_LIMIT = int(os.environ.get("DETECT_MAX_BODY", "12582912"))  # 12MB (10MB audio + multipart overhead)
_DEFAULT_BODY_LIMIT = int(os.environ.get("DEFAULT_MAX_BODY", "65536"))  # 64KB


def _resolve_body_limit(path: str) -> int:
    """요청 경로에 따라 body size 캡을 결정한다."""
    if path.startswith("/api/v2/telemetry"):
        return _TELEMETRY_BODY_LIMIT
    if path.startswith("/api/v1/detect"):
        return _DETECT_BODY_LIMIT
    return _DEFAULT_BODY_LIMIT


# 오디오 업로드 경로(큰 body)는 미들웨어 사전 버퍼링을 건너뛰고 Content-Length 기반 사전 검사 +
# 스트리밍 통과로 처리. 작은 JSON 경로(telemetry/default)는 chunked 대응을 위해
# 점증 카운트하며 downstream으로 즉시 forward — 전체를 미리 버퍼링하지 않는다.
_STREAMING_PASSTHROUGH_PATHS = ("/api/v1/detect",)


async def _send_error_once(send: Send, status_code: int, detail: str) -> None:
    """downstream 호출을 건너뛰고 단일 JSON 에러 응답을 발신.

    413을 포함해 미들웨어가 직접 거부하는 모든 에러 응답에 `Connection: close`를 실어,
    클라이언트가 오버사이즈 body를 계속 흘려 보내면서 keep-alive 타이머를 리셋해 워커를
    장악하는 것을 차단한다. ASGI 서버(uvicorn/h11)가 이 헤더를 인식해 전송 완료 후 소켓을
    닫는다.
    """
    payload = JSONResponse({"detail": detail}, status_code=status_code)
    try:
        await send(
            {
                "type": "http.response.start",
                "status": payload.status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"connection", b"close"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": payload.body,
                "more_body": False,
            }
        )
    except Exception:
        pass


class BodySizeLimitMiddleware:
    """
    Body size 제한 미들웨어.

    엔드포인트별로 두 가지 모드:
    1. **스트리밍 통과(기본: /api/v1/detect 등 대용량 업로드)**: Content-Length 헤더를
       사전 검증해 초과면 즉시 413. 헤더가 없으면 receive를 래핑해 바이트를 집계만
       하고 downstream으로 즉시 forward — 전체 버퍼링 없음. 한도 초과 시점에는
       downstream이 이미 응답을 시작했을 수 있어 mid-stream 중단은 불가하므로
       `_STREAMING_HARD_CAP` 기준으로 추가 안전망을 둔다.
    2. **작은 JSON(telemetry/default)**: Content-Length 사전 검증 + chunked 대응을 위해
       점증 카운트하며 즉시 forward. chunk 단위로 통과시키므로 전체 버퍼링 없음.

    이전 버전의 "전체 body를 chunks 리스트에 쌓은 뒤 replay" 방식은 대용량 오디오를
    RAM에 이중 적재해 동시 업로드 시 메모리 압박을 유발했으므로 폐기.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()
        if method not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        limit = _resolve_body_limit(path)

        # Content-Length 사전 검증 (모든 경로 공통). 선언된 길이가 이미 초과면
        # downstream 호출 전 단일 413 응답.
        headers = dict(scope.get("headers", []))
        cl_header = headers.get(b"content-length")
        if cl_header is not None:
            try:
                declared = int(cl_header.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                await _send_error_once(send, status.HTTP_400_BAD_REQUEST, "Invalid Content-Length")
                return
            if declared < 0:
                await _send_error_once(send, status.HTTP_400_BAD_REQUEST, "Invalid Content-Length")
                return
            if declared > limit:
                await _send_error_once(
                    send,
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"Request body exceeds {limit} bytes",
                )
                return

        # **스트리밍 카운팅 + 명시적 413 오버라이드**:
        # - receive는 각 chunk를 downstream으로 즉시 forward하며 바이트를 카운트한다.
        #   전체 body 버퍼링 없음.
        # - send는 wrapping — downstream이 response.start를 보내기 전에 오버사이즈가
        #   감지되면 downstream의 응답 대신 413을 발신하고 이후 메시지를 drop한다.
        #   downstream이 먼저 응답을 시작했으면(경쟁 상황) 그 응답을 통과시킨다.
        counted = [0]
        exceeded = [False]
        response_started = [False]
        overridden = [False]
        body_fully_consumed = [False]

        async def counting_receive() -> Message:
            if exceeded[0]:
                return {"type": "http.request", "body": b"", "more_body": False}
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"") or b""
                counted[0] += len(body)
                more_body = message.get("more_body", False)
                if not more_body:
                    body_fully_consumed[0] = True
                if counted[0] > limit:
                    exceeded[0] = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def _drain_remaining_body() -> None:
            """오버사이즈로 EOF 합성 후 keep-alive 연결에 남은 body 바이트를 소비.
            - 오버플로 청크가 마지막이었다면 드레인 불필요 (EOF 이미 도달).
            - 전체 드레인에 1초 timeout을 걸어 슬로우 클라이언트가 워커를 붙잡지 못하게 한다."""
            if body_fully_consumed[0]:
                return
            async def _drain_loop() -> None:
                while True:
                    msg = await receive()
                    msg_type = msg.get("type")
                    if msg_type == "http.request":
                        if not msg.get("more_body", False):
                            return
                        continue
                    return  # disconnect 등.
            try:
                await asyncio.wait_for(_drain_loop(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                # 타임아웃 또는 receive 오류 — ASGI 서버(uvicorn 등)가 연결 정리를 맡는다.
                return

        async def gated_send(message: Message) -> None:
            msg_type = message.get("type")
            if exceeded[0] and not response_started[0] and not overridden[0]:
                overridden[0] = True
                response_started[0] = True
                # 남은 body 바이트 드레인 — keep-alive 연결 오염 방지.
                await _drain_remaining_body()
                await _send_error_once(
                    send,
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"Request body exceeds {limit} bytes",
                )
                return
            if overridden[0]:
                return
            if msg_type == "http.response.start":
                response_started[0] = True
            await send(message)

        # downstream이 body 파싱 중 예외를 던지거나 응답을 전혀 보내지 않는 경로 모두에서
        # oversize 판정이 서면 명시적 413으로 마무리한다.
        try:
            await self.app(scope, counting_receive, gated_send)
        except Exception:
            if exceeded[0] and not response_started[0] and not overridden[0]:
                overridden[0] = True
                response_started[0] = True
                await _send_error_once(
                    send,
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"Request body exceeds {limit} bytes",
                )
                return
            raise
        # downstream이 silent return (예외 없이, send 호출 없이 종료)한 케이스 — 사후 보정.
        if exceeded[0] and not response_started[0] and not overridden[0]:
            overridden[0] = True
            response_started[0] = True
            await _drain_remaining_body()
            await _send_error_once(
                send,
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Request body exceeds {limit} bytes",
            )


app.add_middleware(BodySizeLimitMiddleware)

# CORS — 환경변수로 화이트리스트 오리진 지정. 미설정 시 안전한 로컬 기본값.
_cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "X-API-Version", "If-None-Match", "Content-Type"],
    allow_credentials=False,
    max_age=600,
)


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _detector is not None,
    }


@app.get("/api/v1/model/info")
async def model_info():
    if _detector is None:
        return {"error": "Model not loaded"}
    return {
        "version": _detector.model_version,
        "sample_rate": 16000,
        "input_samples": 64600,
        "input_duration_sec": 4.0375,
    }


@app.post("/api/v1/detect")
async def detect(audio: UploadFile):
    from model_server.api.routes.detect import detect_deepfake
    return await detect_deepfake(audio)
