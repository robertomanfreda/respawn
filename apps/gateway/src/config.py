from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "0.0.0.0"
    app_port: int = 8080
    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    model_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    default_model: str = "gpt-oss:120b"
    vision_model: str = "moondream:latest"
    model_capabilities: str = "gpt-oss:120b=text,file-text,reasoning,tools;gpt-oss-120b=text,file-text,reasoning,tools;moondream:latest=text,file-text,vision"
    multimodal_download_timeout_seconds: float = 10.0
    multimodal_max_image_bytes: int = 5_000_000
    multimodal_max_file_bytes: int = 2_000_000
    file_storage_backend: str = "database"
    file_storage_path: str = "./data/files"
    file_upload_max_bytes: int = 20_000_000
    file_storage_quota_bytes: int = 100_000_000
    file_default_ttl_seconds: int = 0
    file_cleanup_interval_seconds: float = 300.0
    file_malware_scan_enabled: bool = True
    idempotency_cache_max_entries: int = 1024
    auth_disabled: bool = True
    local_openai_api_keys: str = "local-dev-key:tenant-local"
    store_default: bool = True
    max_chain_depth: int = 50
    context_window_default_tokens: int = 131072
    model_context_windows: str = "gpt-oss:120b=131072;gpt-oss-120b=131072;gpt-oss:20b=131072;gpt-oss-20b=131072;moondream:latest=4096"
    context_token_margin: int = 256
    backend_timeout_seconds: float = 120.0
    background_job_timeout_seconds: float = 300.0
    background_job_heartbeat_seconds: float = 1.0
    stream_heartbeat_seconds: float = 15.0
    max_output_tokens_default: int = 2048
    prompt_cache_enabled: bool = True
    prompt_cache_min_tokens: int = 1024
    prompt_cache_max_entries: int = 256
    prompt_cache_in_memory_ttl_seconds: int = 3600
    prompt_cache_extended_ttl_seconds: int = 86400
    prompt_cache_chunk_tokens: int = 128
    web_search_enabled: bool = False
    web_search_backend: str = "searxng"
    web_search_base_url: str = "http://searxng:8080"
    web_search_timeout_seconds: float = 10.0
    web_search_max_results: int = 5
    web_search_max_result_chars: int = 12_000
    web_search_cache_ttl_seconds: int = 3600
    web_search_allowed_domains: str = ""
    web_search_blocked_domains: str = ""
    web_search_user_agent: str = "RespawnWebSearch/1.0"
    image_generation_enabled: bool = False
    image_generation_backend: str = "comfyui"
    image_generation_base_url: str = "http://comfyui:8188"
    image_generation_model: str = "sd-v1-5.safetensors"
    image_generation_timeout_seconds: float = 180.0
    image_generation_default_size: str = "512x512"
    image_generation_max_pixels: int = 1_048_576
    image_generation_default_steps: int = 8
    image_generation_max_steps: int = 30
    image_generation_default_cfg_scale: float = 7.0
    image_generation_sampler: str = "Euler a"
    image_generation_negative_prompt: str = ""
    image_generation_output_format: str = "png"
    reasoning_encryption_key: str = "respawn-local-dev-reasoning-key-change-me"
    reasoning_heavy_token_threshold: int = 128
    include_expansion_max_bytes: int = 4_000_000
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
