"""BodySizeLimitMiddleware 회귀 테스트.

Codex adversarial review HIGH #2 (미들웨어가 전체 body를 RAM에 buffer) 수정 검증.
주요 검증 경로:
  1. Content-Length 명시적 > limit → 413 (downstream 호출 없이)
  2. Content-Length 명시적 <= limit → 정상 통과
  3. Content-Length 없는 chunked body > limit → 413 (downstream 응답 전 override)
  4. downstream이 response.start 보내기 전 예외 → 413 override 성공
  5. 정상 스트리밍 → body는 통과, downstream이 모두 수신

ASGI 레벨 단위 테스트 — FastAPI 인스턴스 전체를 올리지 않고 미들웨어만 직접 구동.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, List

import pytest

from model_server.api.main import BodySizeLimitMiddleware, _resolve_body_limit


async def _collect_sent(send_log: List[dict]) -> Callable[[dict], Awaitable[None]]:
    async def send(message: dict) -> None:
        send_log.append(message)
    return send


async def _body_count_app(scope, receive, send) -> None:
    """downstream: body를 모두 읽고 200/counting JSON 응답."""
    total = 0
    while True:
        msg = await receive()
        if msg.get("type") == "http.request":
            total += len(msg.get("body", b"") or b"")
            if not msg.get("more_body", False):
                break
        else:
            break
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": f'{{"received":{total}}}'.encode(),
            "more_body": False,
        }
    )


async def _raise_before_send_app(scope, receive, send) -> None:
    """downstream: body를 다 읽기도 전에 예외 — middleware의 exception 경로 테스트."""
    await receive()
    raise RuntimeError("parser failed before send")


def _scope(path: str, method: str = "POST", content_length: int | None = None) -> dict:
    headers: list[tuple[bytes, bytes]] = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers,
    }


def _receive_chunks(chunks: list[bytes]):
    idx = [0]

    async def receive():
        i = idx[0]
        if i < len(chunks):
            idx[0] = i + 1
            return {
                "type": "http.request",
                "body": chunks[i],
                "more_body": i + 1 < len(chunks),
            }
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


@pytest.mark.asyncio
async def test_content_length_over_limit_returns_413_without_calling_app():
    called = {"downstream": False}

    async def app(scope, receive, send):
        called["downstream"] = True

    mw = BodySizeLimitMiddleware(app)
    sent: list[dict] = []

    limit = _resolve_body_limit("/api/v2/telemetry")
    scope = _scope("/api/v2/telemetry", content_length=limit + 1)

    async def send(message):
        sent.append(message)

    await mw(scope, _receive_chunks([b"x" * (limit + 1)]), send)

    assert not called["downstream"], "downstream app must be skipped on pre-check 413"
    statuses = [m.get("status") for m in sent if m.get("type") == "http.response.start"]
    assert statuses == [413]


@pytest.mark.asyncio
async def test_within_limit_passes_through_to_downstream():
    mw = BodySizeLimitMiddleware(_body_count_app)
    sent: list[dict] = []
    limit = _resolve_body_limit("/api/v2/telemetry")
    body = b"y" * (limit - 100)
    scope = _scope("/api/v2/telemetry", content_length=len(body))

    async def send(message):
        sent.append(message)

    await mw(scope, _receive_chunks([body]), send)
    statuses = [m.get("status") for m in sent if m.get("type") == "http.response.start"]
    assert statuses == [200], "under-limit request should pass through"


@pytest.mark.asyncio
async def test_chunked_over_limit_overrides_to_413_when_downstream_did_not_start_response():
    """downstream이 body만 read하고 응답 전에 merge된 케이스."""

    async def app(scope, receive, send):
        # body를 chunk 단위로 소진하되 응답은 보내지 않는다 — override 경로 노출.
        while True:
            msg = await receive()
            if msg.get("type") == "http.request" and not msg.get("more_body", False):
                break

    mw = BodySizeLimitMiddleware(app)
    sent: list[dict] = []
    limit = _resolve_body_limit("/api/v2/telemetry")
    scope = _scope("/api/v2/telemetry")  # no Content-Length → chunked.
    over = limit + 500

    async def send(message):
        sent.append(message)

    await mw(scope, _receive_chunks([b"z" * over]), send)

    statuses = [m.get("status") for m in sent if m.get("type") == "http.response.start"]
    assert 413 in statuses, f"expected 413 on chunked over-limit, got {statuses}"


@pytest.mark.asyncio
async def test_downstream_exception_before_send_triggers_413_when_exceeded():
    """미들웨어는 downstream이 예외를 던지더라도 oversize 판정된 경우 413 응답을 보장."""
    mw = BodySizeLimitMiddleware(_raise_before_send_app)
    sent: list[dict] = []
    limit = _resolve_body_limit("/api/v2/telemetry")
    scope = _scope("/api/v2/telemetry")  # no CL

    async def send(message):
        sent.append(message)

    await mw(scope, _receive_chunks([b"q" * (limit + 100)]), send)

    statuses = [m.get("status") for m in sent if m.get("type") == "http.response.start"]
    assert 413 in statuses


@pytest.mark.asyncio
async def test_non_post_bypasses_middleware():
    called = {"downstream": False}

    async def app(scope, receive, send):
        called["downstream"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    mw = BodySizeLimitMiddleware(app)
    sent: list[dict] = []
    scope = {"type": "http", "method": "GET", "path": "/api/v2/rules", "headers": []}

    async def send(message):
        sent.append(message)

    await mw(scope, _receive_chunks([]), send)
    assert called["downstream"], "GET should pass through without middleware enforcement"


@pytest.mark.asyncio
async def test_streaming_wrapper_does_not_buffer_full_body():
    """downstream이 chunk를 incremental하게 수신해야 — replay 기반 모델이 아님을 확인.

    이전 구현은 전체를 chunks 리스트에 버퍼링한 뒤 replay_receive로 재주입했다. 현재 구현은
    forward 방식이므로 downstream이 빈 EOF를 만나기 전에 실제 chunk를 수신한다.
    """
    received_chunks: list[bytes] = []

    async def app(scope, receive, send):
        while True:
            msg = await receive()
            if msg.get("type") == "http.request":
                received_chunks.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    mw = BodySizeLimitMiddleware(app)
    sent: list[dict] = []
    scope = _scope("/api/v2/telemetry", content_length=9)  # "part1" + "part2" = 10? 조정.
    parts = [b"abc", b"defg", b"hi"]  # total 9 bytes.
    scope = _scope("/api/v2/telemetry", content_length=sum(len(p) for p in parts))

    async def send(message):
        sent.append(message)

    await mw(scope, _receive_chunks(parts), send)
    # downstream이 각 chunk를 별도로 받았는지 확인 — 전체 버퍼 후 단일 delivery가 아님.
    non_empty = [c for c in received_chunks if c]
    assert non_empty == parts, f"downstream should see streamed chunks, got {received_chunks}"
