"""One JSON line per request (D-014).

The middleware opens a dict of fields for each request and writes it out when the request
ends, whatever the outcome (200, 4xx, 503, crash). Layers that know something about the
request — the route (tokens), SerialBackend (queue and generation time) — add to it with
`annotate()`, so no layer has to pass timing data through return values.
"""

import hashlib
import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Request, Response

REQUEST_ID_HEADER = "X-Request-ID"
# Docker's health check calls /health every 30 s; logging it would bury real traffic.
_UNLOGGED_PATHS = frozenset({"/health"})
# Caller-supplied ids end up in our logs; accept only short, plain ones.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")

logger = logging.getLogger("gateway.requests")

_fields: ContextVar[dict[str, Any] | None] = ContextVar("request_log_fields", default=None)


def annotate(**fields: Any) -> None:
    """Add fields to the current request's log line. A no-op outside a request."""
    current = _fields.get()
    if current is not None:
        current.update(fields)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def prompt_hash(messages: Iterable[tuple[str, str]]) -> str:
    """Stable hash of (role, content) pairs: shows repeated prompts without storing their text."""
    canonical = json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
    fields: dict[str, Any] = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client": None,  # no client identity until the rate limiter defines one
    }
    token = _fields.set(fields)
    start = time.perf_counter()
    status = 500  # if call_next raises, the server answers 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    except Exception as exc:
        fields["error"] = type(exc).__name__
        raise
    finally:
        if request.url.path not in _UNLOGGED_PATHS:
            line = {**fields, "status": status, "total_ms": elapsed_ms(start)}
            logger.info("request", extra={"fields": line})
        _fields.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        return json.dumps({"ts": ts, **getattr(record, "fields", {})}, ensure_ascii=False)


def configure_request_log(log_file: Path | None) -> None:
    """Send request lines to stdout and, if given, append them to `log_file`."""
    close_request_log()  # create_app can run many times in one process (tests)
    logger.setLevel(logging.INFO)
    # Own handlers only: don't also pass lines to the root/uvicorn loggers.
    logger.propagate = False
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)


def close_request_log() -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
