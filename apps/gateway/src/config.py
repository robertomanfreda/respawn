from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "0.0.0.0"
    app_port: int = 8080
    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    redis_url: str | None = None
    model_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    default_model: str = "gpt-oss:120b"
    auth_disabled: bool = True
    local_openai_api_keys: str = "local-dev-key:tenant-local"
    store_default: bool = True
    max_chain_depth: int = 50
    max_tool_iterations: int = 8
    tool_timeout_seconds: float = 15.0
    backend_timeout_seconds: float = 120.0
    stream_heartbeat_seconds: float = 15.0
    max_output_tokens_default: int = 2048
    prompt_cache_enabled: bool = True
    prompt_cache_min_tokens: int = 1024
    prompt_cache_max_entries: int = 256
    prompt_cache_in_memory_ttl_seconds: int = 3600
    prompt_cache_extended_ttl_seconds: int = 86400
    prompt_cache_chunk_tokens: int = 128
    auto_create_tables: bool = Field(default=True, description="Convenience for local dev/tests; use Alembic in production.")

    def tenant_for_key(self, api_key: str | None) -> str | None:
        if self.auth_disabled:
            return None
        if not api_key:
            return None
        for entry in self.local_openai_api_keys.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                key, tenant = entry.split(":", 1)
                if api_key == key:
                    return tenant or key
            elif api_key == entry:
                return entry
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
