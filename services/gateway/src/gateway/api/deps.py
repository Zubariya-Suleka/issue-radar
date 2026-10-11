"""Request dependencies: who is calling (API key) and whether they are within their rate."""

import hmac
import math

from fastapi import Request

from gateway.api.errors import ApiError
from gateway.request_log import annotate

ANONYMOUS = "anonymous"


def client_name(request: Request) -> str:
    """Map the Bearer key to a client name. With no keys configured, auth is off (D-016)."""
    keys: dict[str, str] = request.app.state.settings.api_keys  # client name -> key
    if not keys:
        annotate(client=ANONYMOUS)
        return ANONYMOUS

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() == "bearer" and presented:
        for name, key in keys.items():
            # Constant-time comparison: response timing doesn't reveal how much of a key matched.
            if hmac.compare_digest(presented.encode(), key.encode()):
                annotate(client=name)
                return name

    raise ApiError(401, "Invalid or missing API key", "invalid_request_error", "invalid_api_key")


def enforce_rate_limit(request: Request, client: str) -> None:
    limiter = request.app.state.rate_limiter
    if limiter is None:
        return
    wait_s = limiter.check(client)
    if wait_s > 0:
        annotate(rate_limited=True)
        raise ApiError(
            429,
            f"Rate limit exceeded for client '{client}'; retry in {wait_s:.1f}s",
            "rate_limit_error",
            "rate_limit_exceeded",
            headers={"Retry-After": str(max(1, math.ceil(wait_s)))},
        )
