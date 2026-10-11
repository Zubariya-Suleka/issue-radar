from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """An error returned to the client in OpenAI's format, which OpenAI SDKs parse and surface."""

    def __init__(
        self,
        status: int,
        message: str,
        type: str,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.headers = headers


async def api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"message": exc.message, "type": exc.type, "code": exc.code}},
        headers=exc.headers,
    )
