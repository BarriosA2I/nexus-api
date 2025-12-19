"""
Nexus Assistant Unified - Configuration
Environment-based configuration with sensible defaults
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Nexus Assistant Unified"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # CORS
    CORS_ORIGINS: str = "*"

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    KNOWLEDGE_DIR: Optional[str] = None
    RAGNAROK_V6_PATH: Optional[str] = None
    RAGNAROK_BAR_PATH: Optional[str] = None

    # RAG Configuration
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 50
    RAG_SIMILARITY_THRESHOLD: float = 0.15
    RAG_MAX_RESULTS: int = 5

    # Circuit Breaker
    CB_FAILURE_THRESHOLD: int = 5
    CB_RECOVERY_TIMEOUT: int = 30
    CB_HALF_OPEN_MAX_CALLS: int = 3

    # Cache
    CACHE_ENABLED: bool = False
    REDIS_URL: Optional[str] = None
    CACHE_TTL_SECONDS: int = 300

    # LLM (for future real LLM integration)
    LLM_PROVIDER: str = "mock"  # mock, openai, anthropic
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024

    # Ragnarok
    RAGNAROK_MODE: str = "mock"  # mock, path, api
    RAGNAROK_API_URL: Optional[str] = None
    RAGNAROK_TIMEOUT: int = 300

    # Job Store
    JOB_RETENTION_HOURS: int = 24
    JOB_MAX_CONCURRENT: int = 5

    @property
    def knowledge_path(self) -> Path:
        """Get knowledge directory path"""
        if self.KNOWLEDGE_DIR:
            return Path(self.KNOWLEDGE_DIR)
        return self.BASE_DIR / "knowledge"

    @property
    def ragnarok_path(self) -> Optional[Path]:
        """Get Ragnarok installation path"""
        if self.RAGNAROK_V6_PATH:
            return Path(self.RAGNAROK_V6_PATH)
        if self.RAGNAROK_BAR_PATH:
            return Path(self.RAGNAROK_BAR_PATH)
        return None

    @property
    def cors_origins_list(self) -> list:
        """Parse CORS origins string to list"""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience accessor
settings = get_settings()
