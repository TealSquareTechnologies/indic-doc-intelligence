"""Application error types, mapped to HTTP status codes by the API layer."""

from __future__ import annotations


class AppError(Exception):
    """Base class for errors that carry a meaningful HTTP status."""

    status_code = 500
    default_message = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message


class ValidationError(AppError):
    """The caller supplied input the application cannot work with."""

    status_code = 400
    default_message = "The request payload is invalid."


class NotFoundError(AppError):
    """The requested resource does not exist."""

    status_code = 404
    default_message = "The requested resource was not found."


class PayloadTooLargeError(AppError):
    """The upload exceeds the configured size limit."""

    status_code = 413
    default_message = "The uploaded file is too large."


class ConfigurationError(AppError):
    """A required setting is missing or malformed."""

    status_code = 500
    default_message = "The application is not configured correctly."


class DependencyUnavailableError(AppError):
    """An external dependency such as Tesseract or the model provider is unreachable."""

    status_code = 503
    default_message = "A required dependency is unavailable."
