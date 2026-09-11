"""ASGI middleware: request IDs, request logging, body-size limit, rate limiting."""

import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging_config import request_id_var

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = b"x-request-id"
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (
        b"content-security-policy",
        b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        b"connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    ),
)


def _error_body(code: str, message: str) -> bytes:
    return json.dumps({"error": {"code": code, "message": message}}).encode()


async def _send_json_error(send: Send, status: int, code: str, message: str, request_id: str) -> None:
    body = _error_body(code, message)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (REQUEST_ID_HEADER, request_id.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestContextMiddleware:
    """Assigns a server-generated request ID, logs completion, adds headers.

    Client-supplied ``X-Request-ID`` values are ignored on purpose: accepting
    them would let a caller inject arbitrary strings into structured logs.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_holder = {"status": 0}

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode()))
                headers.extend(SECURITY_HEADERS)
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            path = scope.get("path", "")
            if not path.startswith("/static"):
                logger.info(
                    "request completed",
                    extra={
                        "method": scope.get("method"),
                        "path": path,
                        "status": status_holder["status"],
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                    },
                )
            request_id_var.reset(token)


class BodySizeLimitMiddleware:
    """Rejects request bodies larger than ``max_bytes`` with a 413.

    Bodies are small (a single question), so buffering up to the limit before
    handing the request to the application is cheap and avoids unbounded reads.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        request_id = request_id_var.get() or "-"
        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await _send_json_error(
                send, 413, "payload_too_large", "The request body exceeds the allowed size.", request_id
            )
            return

        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                await _send_json_error(
                    send, 413, "payload_too_large", "The request body exceeds the allowed size.", request_id
                )
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        buffered = b"".join(chunks)
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": buffered, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def resolve_client_ip(scope: Scope, trust_proxy_headers: bool) -> str:
    """Best-effort client identity for rate limiting.

    Behind Render's proxy the socket peer is the load balancer, so the
    right-most ``X-Forwarded-For`` entry (the one appended by the proxy we
    trust) identifies the client. Without a trusted proxy the header is
    attacker-controlled and is ignored.
    """
    if trust_proxy_headers:
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-for":
                forwarded = value.decode("latin-1").split(",")
                candidate = forwarded[-1].strip()
                if candidate:
                    return candidate
    client = scope.get("client")
    return client[0] if client else "unknown"


class RateLimiter:
    """Sliding-window, per-key, in-memory limiter for a single process.

    Limitations (by design, for a demo): state is process-local, resets on
    restart and is not shared across instances. Production would move this
    to Redis or an API gateway.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._last_prune = clock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            if now - self._last_prune > self._window:
                self._prune(now)
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] >= self._window:
                bucket.popleft()
            if len(bucket) >= self._max:
                return False
            bucket.append(now)
            return True

    def _prune(self, now: float) -> None:
        stale = [key for key, bucket in self._hits.items() if not bucket or now - bucket[-1] >= self._window]
        for key in stale:
            del self._hits[key]
        self._last_prune = now
