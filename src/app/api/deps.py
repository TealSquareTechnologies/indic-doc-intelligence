"""Shared dependencies for the API routes."""

from __future__ import annotations

from fastapi import Header, UploadFile

from app.core.config import get_settings
from app.core.errors import AppError, PayloadTooLargeError, ValidationError

READ_CHUNK_BYTES = 1024 * 1024


class UnauthorisedError(AppError):
    """The caller did not present the configured shared secret."""

    status_code = 401
    default_message = "A valid X-API-Key header is required."


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce the shared secret when one is configured."""
    expected = get_settings().api_key.strip()
    if not expected:
        return
    if x_api_key != expected:
        raise UnauthorisedError()


async def read_upload(file: UploadFile) -> bytes:
    """Read an upload into memory, stopping as soon as it exceeds the size limit."""
    limit = get_settings().max_upload_bytes
    buffer = bytearray()

    while True:
        block = await file.read(READ_CHUNK_BYTES)
        if not block:
            break
        buffer.extend(block)
        if len(buffer) > limit:
            raise PayloadTooLargeError(
                f"The upload exceeds the {limit // (1024 * 1024)} MB limit."
            )

    if not buffer:
        raise ValidationError("The uploaded file is empty.")
    return bytes(buffer)
