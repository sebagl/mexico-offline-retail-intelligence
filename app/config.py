"""Application settings loaded from environment variables and ``.env``."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class Settings(BaseSettings):
    """Runtime configuration.

    ``INEGI_API_TOKEN`` (ingestion only) and ``GEMINI_API_KEY`` are secrets.
    Every other value is safe to expose in deployment manifests. Numeric
    limits are validated at startup so a bad environment fails fast.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    inegi_api_token: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    gemini_model: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    fastembed_cache_path: Path | None = None
    data_dir: Path = Path("data")
    allowed_origins: str = "http://localhost:8000"
    max_question_length: int = Field(default=500, ge=1, le=5000)
    max_request_body_bytes: int = Field(default=16_384, ge=1024, le=1_048_576)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    # bge-small cosine scores are compressed: unrelated questions still score
    # ~0.45-0.50 against this corpus while real evidence scores >= ~0.65.
    # 0.57 sits in the measured gap (see scripts/evaluate.py).
    retrieval_min_score: float = Field(default=0.57, ge=-1.0, le=1.0)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    rate_limit_requests: int = Field(default=20, ge=1, le=10_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=86_400)
    # Only enable behind a proxy that overwrites/appends X-Forwarded-For
    # (Render does). Never enable when clients can reach Uvicorn directly.
    trust_proxy_headers: bool = False
    log_level: str = "INFO"
    port: int = Field(default=8000, ge=1, le=65_535)

    @field_validator("gemini_model", "embedding_model", "log_level", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
        return level

    @property
    def gemini_configured(self) -> bool:
        """True only when both a non-blank key and a model name are present."""
        key = self.gemini_api_key.get_secret_value().strip() if self.gemini_api_key else ""
        return bool(key) and bool(self.gemini_model)

    @property
    def inegi_token_value(self) -> str:
        return self.inegi_api_token.get_secret_value().strip() if self.inegi_api_token else ""

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
