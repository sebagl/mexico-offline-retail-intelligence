"""Controlled application errors mapped to the public error format."""


class AppError(Exception):
    """Base class for errors that are safe to surface to API clients."""

    status_code = 500
    code = "internal_error"
    message = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        if message is not None:
            self.message = message


class InvalidQuestionError(AppError):
    status_code = 400
    code = "invalid_question"
    message = "The question is empty or invalid after normalization."


class RateLimitExceededError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "Too many requests. Please wait a moment and try again."


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"
    message = "The request body exceeds the allowed size."


class DatasetUnavailableError(AppError):
    status_code = 503
    code = "dataset_unavailable"
    message = "The DENUE dataset is temporarily unavailable."


class DatasetError(Exception):
    """Raised while loading or validating the generated data artifacts.

    Internal: the message may reference file structure and is logged at
    startup, but it is never returned to clients verbatim.
    """
